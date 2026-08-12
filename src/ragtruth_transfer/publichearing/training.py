from __future__ import annotations

import copy
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..config import ExperimentConfig
from ..io_utils import sha256_file, write_json
from ..modeling import build_model, head_state_dict
from .config import PublicHearingConfig
from .tokenization import make_loader


def _model_config(config: PublicHearingConfig) -> ExperimentConfig:
    return ExperimentConfig.from_mapping({
        "run_name": config.run_name, "model_id": config.model_id, "model_revision": config.model_revision,
        "architecture": "gated_attention", "encoder_mode": "lora", "max_length": config.max_length,
        "projection_size": config.projection_size, "attention_size": config.attention_size, "dropout": config.dropout, "gradient_checkpointing": config.gradient_checkpointing,
        "lora": {"r": config.lora_r, "alpha": config.lora_alpha, "dropout": config.lora_dropout, "target_modules": list(config.lora_target_modules)},
        "training": {"train_batch_size": config.train_batch_size, "eval_batch_size": config.eval_batch_size, "gradient_accumulation_steps": config.gradient_accumulation_steps, "head_learning_rate": config.head_lr, "encoder_learning_rate": config.lora_lr, "weight_decay": config.weight_decay, "warmup_ratio": config.warmup_ratio, "max_epochs": config.max_epochs, "early_stopping_patience": config.early_stopping_patience, "max_grad_norm": config.max_grad_norm, "use_class_weight": config.use_class_weight, "task_balanced_sampler": False, "num_workers": config.num_workers},
    })


def _seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def predict(model: nn.Module, loader, device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    amp = device.type == "cuda"
    for batch in loader:
        labels = batch.pop("labels")
        indices = batch.pop("row_index")
        tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits, attention = model(**tensors)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        weights = attention.float().cpu().numpy()
        for position, probability in enumerate(probabilities):
            row = {"row_index": int(indices[position]), "label": int(labels[position]), "probability": float(probability)}
            row.update({f"attention_chunk_{chunk}": float(weights[position, chunk]) for chunk in range(4)})
            rows.append(row)
    output = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    attention_columns = [f"attention_chunk_{value}" for value in range(4)]
    if output.empty or not np.isfinite(output["probability"]).all() or not np.isfinite(output[attention_columns].to_numpy()).all():
        raise RuntimeError("Predições ou pesos de atenção não finitos.")
    if not np.allclose(output[attention_columns].sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Pesos de atenção não somam 1.")
    if ((output[attention_columns] < -1e-7) | (output[attention_columns] > 1 + 1e-7)).any().any():
        raise RuntimeError("Pesos de atenção fora de [0, 1].")
    return output


def _ranking(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {"AUPRC": float(average_precision_score(labels, probabilities)), "AUROC": float(roc_auc_score(labels, probabilities)), "Brier": float(brier_score_loss(labels, probabilities))}


def _optimizer(model: nn.Module, config: PublicHearingConfig) -> torch.optim.Optimizer:
    lora, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lora if "lora_" in name else head).append(parameter)
    if not lora or not head:
        raise RuntimeError("LoRA e cabeça devem possuir parâmetros treináveis.")
    return torch.optim.AdamW([
        {"params": lora, "lr": config.lora_lr, "weight_decay": config.weight_decay},
        {"params": head, "lr": config.head_lr, "weight_decay": config.weight_decay},
    ])


def seed_signature(experiment_signature: str, fold: int, seed: int, frame: pd.DataFrame, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> str:
    import hashlib
    payload = {"experiment_signature": experiment_signature, "fold": fold, "seed": seed, "train": frame.iloc[train].sample_id.tolist(), "validation": frame.iloc[validation].sample_id.tolist(), "test": frame.iloc[test].sample_id.tolist()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _complete_seed_dir(path: Path, signature: str) -> bool:
    marker = path / "SEED_RUN_COMPLETE.json"
    required = ("validation_predictions.csv", "test_predictions.csv", "training_history.csv", "metadata.json", "head.pt")
    if not marker.is_file() or not all((path / name).is_file() for name in required):
        return False
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    if marker_value.get("seed_signature") != signature:
        raise RuntimeError(f"Resultado existente incompatível em {path}; use outro output_root.")
    hashes = marker_value.get("files")
    if not isinstance(hashes, dict) or not hashes:
        return False
    for relative_name, expected_hash in hashes.items():
        candidate = path / relative_name
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            return False
    try:
        for name in ("validation_predictions.csv", "test_predictions.csv", "training_history.csv"):
            if pd.read_csv(path / name).empty:
                return False
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        torch.load(path / "head.pt", map_location="cpu", weights_only=True)
    except (OSError, ValueError, TypeError, RuntimeError):
        return False
    return True


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SEED_RUN_COMPLETE.json"}


def train_seed(config: PublicHearingConfig, cache: dict[str, Any], frame: pd.DataFrame, experiment_signature: str, fold: int, seed: int, train: np.ndarray, validation: np.ndarray, test: np.ndarray, seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    signature = seed_signature(experiment_signature, fold, seed, frame, train, validation, test)
    if _complete_seed_dir(seed_dir, signature):
        print(f"fold={fold} seed={seed}: resultado compatível reutilizado")
        return (pd.read_csv(seed_dir / "validation_predictions.csv"), pd.read_csv(seed_dir / "test_predictions.csv"), pd.read_csv(seed_dir / "training_history.csv"), json.loads((seed_dir / "metadata.json").read_text(encoding="utf-8")), True)
    if seed_dir.exists():
        raise RuntimeError(f"Diretório parcial não é resultado concluído: {seed_dir}. Remova-o manualmente após inspecionar.")
    stage = seed_dir.parent / f".{seed_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        _seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"fold={fold} seed={seed} device={device} train={len(train)} ({frame.iloc[train].label.mean():.3%}) validation={len(validation)} ({frame.iloc[validation].label.mean():.3%}) test={len(test)} ({frame.iloc[test].label.mean():.3%})")
        model, model_metadata = build_model(_model_config(config))
        model.to(device)
        train_loader = make_loader(cache, train, config.train_batch_size, True, config.num_workers)
        validation_loader = make_loader(cache, validation, config.eval_batch_size, False, config.num_workers)
        test_loader = make_loader(cache, test, config.eval_batch_size, False, config.num_workers)
        optimizer = _optimizer(model, config)
        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * config.max_epochs * config.warmup_ratio), max(1, steps_per_epoch * config.max_epochs))
        labels = frame.iloc[train].label.to_numpy(dtype=int)
        positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
        if not positives or not negatives:
            raise RuntimeError("Treino sem as duas classes.")
        pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device) if config.use_class_weight else None
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
        best_score, best_epoch, without_improvement = -float("inf"), 0, 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, Any]] = []
        started = time.time()
        for epoch in range(1, config.max_epochs + 1):
            epoch_started, losses = time.time(), []
            model.train(); optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                labels_batch = batch.pop("labels").to(device, non_blocking=True)
                batch.pop("row_index")
                tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.mixed_precision and device.type == "cuda"):
                    logits, _ = model(**tensors)
                    raw_loss = loss_function(logits, labels_batch)
                    loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward(); losses.append(float(raw_loss.detach().cpu()))
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], config.max_grad_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= scale_before:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            predicted = predict(model, validation_loader, device)
            ranking = _ranking(predicted.label.to_numpy(dtype=int), predicted.probability.to_numpy(dtype=float))
            record = {"outer_fold": fold, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "epoch_seconds": time.time() - epoch_started, "lora_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"], **{f"validation_{key}": value for key, value in ranking.items()}}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            if ranking["AUPRC"] > best_score + 1e-8:
                best_score, best_epoch, without_improvement = ranking["AUPRC"], epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in {parameter_name for parameter_name, parameter in model.named_parameters() if parameter.requires_grad}}
            else:
                without_improvement += 1
                if without_improvement >= config.early_stopping_patience:
                    print(f"fold={fold} seed={seed}: early stopping na época {epoch}; melhor época={best_epoch}")
                    break
        if best_state is None:
            raise RuntimeError("Nenhum checkpoint selecionado.")
        model.load_state_dict(best_state, strict=False)
        validation_predictions, test_predictions = predict(model, validation_loader, device), predict(model, test_loader, device)
        validation_predictions.to_csv(stage / "validation_predictions.csv", index=False); test_predictions.to_csv(stage / "test_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(stage / "training_history.csv", index=False)
        torch.save(head_state_dict(model), stage / "head.pt")
        if config.save_checkpoints:
            model.encoder.save_pretrained(stage / "adapter")
            AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision, use_fast=True).save_pretrained(stage / "tokenizer")
        metadata = {"seed_signature": signature, "experiment_signature": experiment_signature, "outer_fold": fold, "seed": seed, "best_epoch": best_epoch, "best_validation_AUPRC": best_score, "pos_weight": float(negatives / positives) if config.use_class_weight else None, "train_examples": len(train), "validation_examples": len(validation), "test_examples": len(test), "train_prevalence": float(labels.mean()), "seconds": time.time() - started, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0, "model": model_metadata, "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}}
        write_json(stage / "metadata.json", metadata)
        hashes = _artifact_hashes(stage)
        write_json(stage / "SEED_RUN_COMPLETE.json", {"seed_signature": signature, "files": hashes})
        os.replace(stage, seed_dir)
        return validation_predictions, test_predictions, pd.DataFrame(history), metadata, False
    except Exception:
        # Kept for audit, but never accepted by _complete_seed_dir or aggregation.
        raise
