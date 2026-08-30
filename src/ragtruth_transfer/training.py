from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .dataset import BagCollator, EvidenceBagDataset
from .io_utils import seed_everything, sha256_file, write_json
from .metrics import binary_metrics, select_threshold
from .modeling import build_model, head_state_dict, load_head_state
from .ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet

CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_REQUIRED_FILES = ("head.pt", "training_state.pt", "validation_predictions.csv")


def _write_split_assignments(output_dir: Path, assignments: pd.DataFrame) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split_assignments.parquet"
    temporary = output_dir / ".split_assignments.parquet.tmp"
    assignments.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)
    return sha256_file(path)


def prepare_training_data(
    config: ExperimentConfig,
    data_dir: Path | None = None,
    *,
    output_dir: Path | None = None,
    max_train_sources: int | None = None,
    max_validation_sources: int | None = None,
) -> tuple[EvidenceBagDataset, EvidenceBagDataset, EvidenceBagDataset, dict[str, Any], dict[str, str]]:
    if config.dataset.format == "jsonl":
        if data_dir is None:
            data_dir = config.dataset.path
        if data_dir is None:
            raise ValueError("--data-dir é obrigatório para dataset.format=jsonl")
        data_hashes = _data_hashes(data_dir)
        train_dataset = EvidenceBagDataset(data_dir / "train.jsonl")
        validation_dataset = EvidenceBagDataset(data_dir / "validation.jsonl")
        test_dataset = EvidenceBagDataset(data_dir / "test.jsonl")
        manifest_path = data_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        metadata = {
            "dataset_format": "jsonl",
            "data_dir": str(data_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "signature": manifest.get("run_signature"),
            "schema_version": manifest.get("schema_version"),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "split_signature": manifest.get("split_signature"),
            "split": {
                "strategy": "precomputed_jsonl_split",
                "group_key": "source_id",
                "signature": manifest.get("split_signature"),
            },
        }
        return train_dataset, validation_dataset, test_dataset, metadata, data_hashes

    parquet_path = config.dataset.path or data_dir
    if parquet_path is None:
        raise ValueError("dataset.path é obrigatório para dataset.format=parquet")
    if max_train_sources is None:
        max_train_sources = config.dataset.max_train_sources
    if max_validation_sources is None:
        max_validation_sources = config.dataset.max_validation_sources
    rows, parquet_metadata = load_ragtruth_parquet(
        parquet_path,
        manifest_path=config.dataset.manifest_path,
        expected_signature=config.dataset.expected_signature,
        expected_schema=config.dataset.expected_schema,
        claim_column=config.dataset.claim_column,
        chunk_columns=config.dataset.chunk_columns,
        evidence_mask_column=config.dataset.evidence_mask_column,
        label_column=config.dataset.label_column,
        group_column=config.dataset.group_column,
        split_column=config.dataset.split_column,
    )
    split = split_ragtruth_parquet(
        rows,
        parquet_metadata,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        max_train_sources=max_train_sources,
        max_validation_sources=max_validation_sources,
        max_test_sources=config.dataset.max_test_sources,
    )
    train_dataset = EvidenceBagDataset.from_rows(split.train_rows)
    validation_dataset = EvidenceBagDataset.from_rows(split.validation_rows)
    test_dataset = EvidenceBagDataset.from_rows(split.test_rows, allow_empty=not config.dataset.evaluate_test)
    split_metadata = dict(split.metadata)
    if output_dir is not None:
        split_metadata["assignment_sha256"] = _write_split_assignments(output_dir, split.assignments)
        split_metadata["assignment_path"] = str((output_dir / "split_assignments.parquet").resolve())
    metadata = {**parquet_metadata, "split": split_metadata, "feature_contract": {"used": ["claim", "evidence", "evidence_mask", "label", "source_id", "example_id"], "excluded": parquet_metadata["excluded_columns"]}}
    data_hashes = {"dataset": parquet_metadata["dataset_sha256"], "manifest": sha256_file(Path(parquet_metadata["manifest_path"])) if parquet_metadata.get("manifest_path") else "", "split": str(split_metadata["signature"]), "assignments": str(split_metadata.get("assignment_sha256", ""))}
    return train_dataset, validation_dataset, test_dataset, metadata, data_hashes


def validate_training_data(
    config: ExperimentConfig,
    data_dir: Path | None = None,
    *,
    output_dir: Path | None = None,
    max_train_sources: int | None = None,
    max_validation_sources: int | None = None,
) -> dict[str, Any]:
    train, validation, test, metadata, hashes = prepare_training_data(
        config,
        data_dir,
        output_dir=output_dir,
        max_train_sources=max_train_sources,
        max_validation_sources=max_validation_sources,
    )
    positives = int(train.labels.sum())
    negatives = int((~train.labels).sum())
    mode = config.training.class_weight_mode
    if mode == "auto_pos_weight":
        if positives == 0 or negatives == 0:
            raise ValueError("Treino efetivo precisa conter pelo menos um positivo e um negativo.")
        pos_weight = negatives / positives
    elif mode == "fixed":
        pos_weight = float(config.training.fixed_pos_weight)
    else:
        pos_weight = None
    result = {
        "status": "valid",
        "model_loaded": False,
        "cuda_initialized": False,
        "forward_executed": False,
        "backward_executed": False,
        "dataset": metadata,
        "data_hashes": hashes,
        "partitions": {
            "train": {"examples": len(train), "sources": len(set(train.sources)), "positives": positives, "negatives": negatives, "prevalence": float(train.labels.mean())},
            "validation": {"examples": len(validation), "sources": len(set(validation.sources)), "positives": int(validation.labels.sum()), "negatives": int((~validation.labels).sum()), "prevalence": float(validation.labels.mean())},
            "test": {"examples": len(test), "sources": len(set(test.sources)), "positives": int(test.labels.sum()), "negatives": int((~test.labels).sum()), "prevalence": float(test.labels.mean()) if len(test) else 0.0},
        },
        "pos_weight": {"mode": mode, "value": pos_weight, "positives_train_effective": positives, "negatives_train_effective": negatives},
        "test_used_for_selection": False,
        "feature_contract": metadata.get("feature_contract", {"used": ["claim", "evidence", "evidence_mask", "label", "source_id", "example_id"], "excluded": []}),
    }
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _data_hashes(data_dir: Path) -> dict[str, str]:
    result = {
        "manifest": sha256_file(data_dir / "manifest.json"),
        "train": sha256_file(data_dir / "train.jsonl"),
        "validation": sha256_file(data_dir / "validation.jsonl"),
        "test": sha256_file(data_dir / "test.jsonl"),
    }
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        for key in ("dataset_sha256", "split_signature"):
            value = manifest.get(key)
            if value:
                result[key if key != "split_signature" else "split"] = str(value)
    return result


def scientific_fingerprint(
    config: ExperimentConfig,
    data_hashes: dict[str, str],
    seed: int,
    limits: dict[str, int | None] | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved = copy.deepcopy(config.to_dict())
    resolved.pop("run_name", None)
    training = resolved["training"]
    training.pop("stop_after_epoch", None)
    training.pop("save_epoch_checkpoints", None)
    payload = {
        "config": resolved,
        "data_split_sha256": data_hashes,
        "seed": seed,
        "limits": limits or {"train": None, "validation": None, "test": None},
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def make_loader(
    dataset: EvidenceBagDataset,
    tokenizer,
    config: ExperimentConfig,
    train: bool,
    seed: int,
    generator: torch.Generator | None = None,
    sampler_generator: torch.Generator | None = None,
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and config.training.task_balanced_sampler:
        sampler = dataset.task_balanced_sampler(seed=seed, generator=sampler_generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=(config.training.train_batch_size if train else config.training.eval_batch_size),
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=config.training.num_workers,
        collate_fn=BagCollator(tokenizer, config.max_length, config.truncation),
        pin_memory=torch.cuda.is_available(),
    )


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inferência", leave=False):
            labels = batch.pop("labels").numpy().astype(bool)
            example_ids = batch.pop("example_ids")
            source_ids = batch.pop("source_ids")
            task_types = batch.pop("task_types")
            tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, weights = model(**tensors)
            scores = torch.sigmoid(logits.float()).cpu().numpy()
            weights_np = weights.float().cpu().numpy()
            for index, example_id in enumerate(example_ids):
                rows.append({
                    "example_id": example_id,
                    "source_id": source_ids[index],
                    "task_type": task_types[index],
                    "label": bool(labels[index]),
                    "score": float(scores[index]),
                    "evidence_weights_json": json.dumps(weights_np[index].tolist()),
                })
    return pd.DataFrame(rows)


def learning_rate_group_parameter_names(model: nn.Module) -> tuple[list[str], list[str]]:
    """Return trainable (head, encoder) parameter names.

    Everything outside ``encoder.*`` belongs to the trainable aggregation head.
    This preserves the historical grouping for mean/max/Gated Attention while
    ensuring new aggregators cannot accidentally receive the encoder/LoRA LR.
    """

    head_names, encoder_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder_names if name.startswith("encoder.") else head_names).append(name)
    return head_names, encoder_names


def optimizer_for_model(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    head_names, encoder_names = learning_rate_group_parameter_names(model)
    parameters = dict(model.named_parameters())
    head_parameters = [parameters[name] for name in head_names]
    encoder_parameters = [parameters[name] for name in encoder_names]
    groups = [{"params": head_parameters, "lr": config.training.head_learning_rate}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config.training.encoder_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root))] = sha256_file(path)
    return result


def _save_epoch_checkpoint(
    checkpoint_dir: Path,
    config: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    seed: int,
    epoch: int,
    next_epoch: int,
    global_step: int,
    train_loss: float,
    validation_auprc: float,
    validation_predictions: pd.DataFrame,
    best_validation_auprc: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    data_hashes: dict[str, str],
    train_sampler_generator: torch.Generator,
    train_loader_generator: torch.Generator,
    best_head_state: dict[str, torch.Tensor],
    best_adapter_state: dict[str, torch.Tensor] | None,
    model_metadata: dict[str, Any],
    total_steps: int,
    warmup_steps: int,
    steps_per_epoch: int,
) -> Path:
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint já existe: {checkpoint_dir}")
    stage = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        if config.encoder_mode == "lora":
            model.encoder.save_pretrained(stage / "adapter")
        torch.save(head_state_dict(model), stage / "head.pt")
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "seed": seed,
            "epoch": epoch,
            "next_epoch": next_epoch,
            "global_step": global_step,
            "micro_step": 0,
            "best_validation_AUPRC": best_validation_auprc,
            "best_epoch": best_epoch,
            "history": history,
            "best_head_state": best_head_state,
            "best_adapter_state": best_adapter_state,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_sampler_generator_state": train_sampler_generator.get_state(),
            "train_loader_generator_state": train_loader_generator.get_state(),
        }, stage / "training_state.pt")
        validation_predictions.to_csv(stage / "validation_predictions.csv", index=False)
        artifacts = _file_hashes(stage)
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "resumable": True,
            "checkpoint_policy": "atomic_complete_epoch",
            "config": config.to_dict(),
            "pooling_type": config.canonical_pooling_type,
            "config_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "model_metadata": model_metadata,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "peft": __import__("peft").__version__,
            },
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "lora": config.to_dict()["lora"],
            "data_split_sha256": data_hashes,
            "seed": seed,
            "epoch": epoch,
            "next_epoch": next_epoch,
            "global_step": global_step,
            "micro_step": 0,
            "planned_total_epochs": config.training.planned_total_epochs,
            "total_training_steps": total_steps,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "train_loss": train_loss,
            "validation_AUPRC": validation_auprc,
            "thresholds": None,
            "validation_metrics": None,
            "test_metrics": None,
            "best_validation_AUPRC": best_validation_auprc,
            "best_epoch": best_epoch,
            "history": history,
            "artifacts": artifacts,
        }
        write_json(stage / "checkpoint_manifest.json", manifest)
        (stage / "CHECKPOINT_COMPLETE").write_text(
            hashlib.sha256((stage / "checkpoint_manifest.json").read_bytes()).hexdigest() + "\n",
            encoding="ascii",
        )
        os.replace(stage, checkpoint_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return checkpoint_dir


def validate_checkpoint(
    checkpoint_dir: Path,
    expected_fingerprint: str | None = None,
    expected_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = checkpoint_dir / "CHECKPOINT_COMPLETE"
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Checkpoint incompleto: marcador/manifesto ausente em {checkpoint_dir}")
    expected_manifest_hash = marker.read_text(encoding="ascii").strip()
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise RuntimeError("Marcador CHECKPOINT_COMPLETE não corresponde ao manifesto")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION or not manifest.get("resumable"):
        raise RuntimeError("Formato de checkpoint não retomável ou incompatível")
    for relative, expected_hash in manifest.get("artifacts", {}).items():
        path = checkpoint_dir / relative
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Artefato ausente, vazio ou corrompido: {relative}")
    for required in CHECKPOINT_REQUIRED_FILES:
        path = checkpoint_dir / required
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Arquivo obrigatório ausente: {required}")
    state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
    for key in (
        "optimizer", "scheduler", "scaler", "epoch", "next_epoch", "global_step", "history",
        "python_random_state", "numpy_random_state", "torch_random_state",
        "train_sampler_generator_state", "train_loader_generator_state",
    ):
        if key not in state:
            raise RuntimeError(f"Estado obrigatório ausente: {key}")
    if expected_fingerprint is not None and manifest.get("config_fingerprint") != expected_fingerprint:
        differences: list[str] = []
        if expected_payload is not None:
            def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
                if isinstance(value, dict):
                    result: dict[str, Any] = {}
                    for key, item in value.items():
                        result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
                    return result
                return {prefix: value}
            saved = flatten(manifest.get("fingerprint_payload", {}))
            current = flatten(expected_payload)
            differences = sorted(
                key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
            )
        raise RuntimeError(
            "Fingerprint incompatível: "
            f"checkpoint={manifest.get('config_fingerprint')} atual={expected_fingerprint}; "
            f"campos divergentes={differences or ['fingerprint sem payload comparável']}"
        )
    return manifest, state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])


def _write_stopped_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    seed: int,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_metric: float,
    checkpoint_epochs: list[int],
    fingerprint: str,
    data_hashes: dict[str, str],
    started: float,
    stop_after_epoch: int,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "status": "stopped_after_epoch",
        "voluntary_stop": True,
        "stop_after_epoch": stop_after_epoch,
        "config": config.to_dict(),
        "pooling_type": config.canonical_pooling_type,
        "seed": seed,
        "config_fingerprint": fingerprint,
        "best_epoch": best_epoch,
        "best_validation_AUPRC": best_metric,
        "selection_metric": "validation_AUPRC",
        "best_checkpoint": f"checkpoints/epoch_{best_epoch:02d}",
        "checkpoint_policy": "atomic_complete_epoch",
        "checkpoint_epochs": checkpoint_epochs,
        "history": history,
        "seconds": time.time() - started,
        "model_metadata": model_metadata,
        "data_split_sha256": data_hashes,
        "thresholds": None,
        "validation_metrics": None,
        "test_metrics": None,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return manifest


def train_run(
    config: ExperimentConfig,
    data_dir: Path | None,
    output_dir: Path,
    seed: int,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
    max_test_examples: int | None = None,
    resume_from_checkpoint: Path | None = None,
    stop_after_epoch: int | None = None,
    max_train_sources: int | None = None,
    max_validation_sources: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.dataset.format == "jsonl" and data_dir is None:
        data_dir = config.dataset.path
    planned_total_epochs = config.training.planned_total_epochs
    stop_after = stop_after_epoch if stop_after_epoch is not None else config.training.stop_after_epoch
    if stop_after is None:
        stop_after = planned_total_epochs
    if not 1 <= stop_after <= planned_total_epochs:
        raise ValueError("stop_after_epoch deve estar entre 1 e planned_total_epochs")
    if (resume_from_checkpoint is not None or stop_after < planned_total_epochs) and not config.training.save_epoch_checkpoints:
        raise ValueError("Retomada/parada operacional exige training.save_epoch_checkpoints=true")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    limits = {"train": max_train_examples, "validation": max_validation_examples, "test": max_test_examples}
    train_dataset, validation_dataset, test_dataset, dataset_metadata, data_hashes = prepare_training_data(
        config,
        data_dir,
        output_dir=output_dir,
        max_train_sources=max_train_sources,
        max_validation_sources=max_validation_sources,
    )
    fingerprint, fingerprint_payload = scientific_fingerprint(config, data_hashes, seed, limits)
    if config.dataset.format == "parquet" and any(value is not None for value in (max_train_examples, max_validation_examples, max_test_examples)):
        raise ValueError("Formato parquet deve ser limitado por source_id, não por número de linhas.")
    if config.dataset.format == "jsonl":
        train_dataset = EvidenceBagDataset(data_dir / "train.jsonl", max_train_examples)  # type: ignore[arg-type]
        validation_dataset = EvidenceBagDataset(data_dir / "validation.jsonl", max_validation_examples)  # type: ignore[arg-type]
        test_dataset = EvidenceBagDataset(data_dir / "test.jsonl", max_test_examples)  # type: ignore[arg-type]
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision, use_fast=True)
    sampler_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = make_loader(train_dataset, tokenizer, config, True, seed, loader_generator, sampler_generator)
    validation_loader = make_loader(validation_dataset, tokenizer, config, False, seed)
    test_loader = make_loader(test_dataset, tokenizer, config, False, seed) if config.dataset.evaluate_test else None
    resume_manifest: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    if resume_from_checkpoint is not None:
        expected_run_dir = resume_from_checkpoint.parent.parent.resolve()
        if output_dir.resolve() != expected_run_dir:
            raise RuntimeError(
                "Retomada deve usar o mesmo output-dir do checkpoint; "
                f"esperado={expected_run_dir} atual={output_dir.resolve()}"
            )
        resume_manifest, resume_state = validate_checkpoint(
            resume_from_checkpoint, fingerprint, fingerprint_payload
        )
        if resume_manifest["planned_total_epochs"] != planned_total_epochs:
            raise RuntimeError("total planejado do checkpoint difere da configuração atual")
        if int(resume_manifest["next_epoch"]) > planned_total_epochs:
            # Um checkpoint com ``next_epoch = planned_total_epochs + 1`` é o
            # estado normal de uma execução já concluída.  ``--resume`` deve
            # ser idempotente nesse caso, sem tentar iniciar uma época extra.
            run_manifest_path = output_dir / "run_manifest.json"
            if run_manifest_path.exists():
                completed_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                if completed_manifest.get("status") == "completed":
                    return completed_manifest
            raise RuntimeError("Checkpoint já ultrapassou o total planejado")
        adapter_path = resume_from_checkpoint / "adapter"
        model, model_metadata = build_model(config, adapter_path=adapter_path, adapter_trainable=True)
    else:
        model, model_metadata = build_model(config)
    model.to(device)
    optimizer = optimizer_for_model(model, config)
    steps_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * planned_total_epochs)
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    labels = train_dataset.labels
    pos_weight = None
    class_weight_mode = config.training.class_weight_mode
    if class_weight_mode == "auto_pos_weight":
        positives = int(labels.sum())
        negatives = int((~labels).sum())
        if positives == 0:
            raise ValueError("Treino sem exemplos positivos.")
        pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    elif class_weight_mode == "fixed":
        positives = int(labels.sum())
        negatives = int((~labels).sum())
        pos_weight = torch.tensor(float(config.training.fixed_pos_weight), dtype=torch.float32, device=device)
    else:
        positives = int(labels.sum())
        negatives = int((~labels).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_head_state: dict[str, torch.Tensor] | None = None
    best_adapter_state: dict[str, torch.Tensor] | None = None
    best_metric = -np.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    if resume_state is not None:
        load_head_state(model, resume_from_checkpoint / "head.pt")
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        global_step = int(resume_state["global_step"])
        start_epoch = int(resume_state["next_epoch"])
        history = list(resume_state["history"])
        best_metric = float(resume_state["best_validation_AUPRC"])
        best_epoch = int(resume_state["best_epoch"])
        best_head_state = resume_state["best_head_state"]
        best_adapter_state = resume_state.get("best_adapter_state")
        sampler_generator.set_state(resume_state["train_sampler_generator_state"])
        loader_generator.set_state(resume_state["train_loader_generator_state"])
        _restore_rng(resume_state)
        if len(history) != start_epoch - 1 or (history and int(history[-1]["epoch"]) != start_epoch - 1):
            raise RuntimeError("Histórico do checkpoint não corresponde à próxima época")

    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, stop_after + 1):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Treino época {epoch}", leave=False), start=1):
            batch.pop("example_ids")
            batch.pop("source_ids")
            batch.pop("task_types")
            labels_batch = batch.pop("labels").to(device, non_blocking=True)
            tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(**tensors)
                raw_loss = criterion(logits, labels_batch)
                loss = raw_loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.append(float(raw_loss.detach().cpu()))
            should_step = step % config.training.gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
        validation_predictions = predict(model, validation_loader, device)
        validation_auprc = float(average_precision_score(
            validation_predictions["label"].to_numpy(dtype=bool),
            validation_predictions["score"].to_numpy(dtype=float),
        ))
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_AUPRC": validation_auprc,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(epoch_record)
        if validation_auprc > best_metric:
            best_metric = validation_auprc
            best_epoch = epoch
            best_head_state = copy.deepcopy(head_state_dict(model))
            if config.encoder_mode == "lora":
                best_adapter_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.encoder.state_dict().items() if "lora_" in key
                }
        last_epoch = epoch
        if config.training.save_epoch_checkpoints:
            _save_epoch_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:02d}", config, model, optimizer, scheduler,
                scaler, seed, epoch, epoch + 1, global_step, epoch_record["train_loss"], validation_auprc,
                validation_predictions, best_metric, best_epoch, history, fingerprint, fingerprint_payload,
                data_hashes, sampler_generator, loader_generator, best_head_state or {}, best_adapter_state,
                model_metadata, total_steps, warmup_steps, steps_per_epoch,
            )
        print(f"época={epoch} loss={epoch_record['train_loss']:.6f} validation_AUPRC={validation_auprc:.6f}")
        if epoch == stop_after and stop_after < planned_total_epochs:
            return _write_stopped_manifest(
                output_dir, config, seed, history, best_epoch, best_metric,
                list(range(1, last_epoch + 1)), fingerprint, data_hashes, started, stop_after, model_metadata,
            )
        if config.training.early_stopping_patience > 0:
            no_improvement = sum(1 for item in reversed(history) if float(item["validation_AUPRC"]) < best_metric)
            if no_improvement >= config.training.early_stopping_patience:
                break

    if best_head_state is None:
        raise RuntimeError("Nenhum checkpoint válido foi produzido.")
    model.load_state_dict(best_head_state, strict=False)
    if best_adapter_state is not None:
        model.encoder.load_state_dict(best_adapter_state, strict=False)
    adapter_dir = output_dir / "adapter"
    if config.encoder_mode == "lora":
        model.encoder.save_pretrained(adapter_dir)
    torch.save(best_head_state, output_dir / "head.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    validation_predictions = predict(model, validation_loader, device)
    test_predictions = predict(model, test_loader, device) if test_loader is not None else pd.DataFrame(columns=["example_id", "source_id", "task_type", "label", "score", "evidence_weights_json"])
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    threshold_rows: dict[str, Any] = {}
    validation_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    y_validation = validation_predictions["label"].to_numpy(dtype=bool)
    score_validation = validation_predictions["score"].to_numpy(dtype=float)
    y_test = test_predictions["label"].to_numpy(dtype=bool)
    score_test = test_predictions["score"].to_numpy(dtype=float)
    validation_ranking = binary_metrics(
        y_validation, score_validation >= 0.5, score_validation
    )
    validation_ranking_metrics = {
        key: float(validation_ranking[key])
        for key in ("AUPRC", "AUROC")
    }
    validation_ranking_metrics["Brier"] = float(
        brier_score_loss(y_validation, score_validation)
    )
    for criterion_name in ("f1", "fpr10"):
        threshold, development_metrics, feasible = select_threshold(y_validation, score_validation, criterion_name)
        threshold_rows[criterion_name] = {"threshold": threshold, "development_metrics": development_metrics, "constraint_feasible": feasible}
        validation_metrics[criterion_name] = {
            **binary_metrics(y_validation, score_validation >= threshold, score_validation),
            "Brier": float(brier_score_loss(y_validation, score_validation)),
        }
        test_metrics[criterion_name] = (
            {
                **binary_metrics(y_test, score_test >= threshold, score_test),
                "Brier": float(brier_score_loss(y_test, score_test)),
            }
            if len(y_test)
            else None
        )
    run_manifest = {
        "status": "completed",
        "voluntary_stop": False,
        "effective_stop_after_epoch": stop_after,
        "config": config.to_dict(),
        "pooling_type": config.canonical_pooling_type,
        "seed": seed,
        "config_fingerprint": fingerprint,
        "best_epoch": best_epoch,
        "best_validation_AUPRC": best_metric,
        "selection_metric": "validation_AUPRC",
        "best_checkpoint": (
            f"checkpoints/epoch_{best_epoch:02d}"
            if config.training.save_epoch_checkpoints else "adapter + head.pt"
        ),
        "checkpoint_policy": (
            "atomic_complete_epoch" if config.training.save_epoch_checkpoints else "best_only"
        ),
        "checkpoint_epochs": [epoch for epoch in range(1, planned_total_epochs + 1) if (output_dir / "checkpoints" / f"epoch_{epoch:02d}" / "CHECKPOINT_COMPLETE").is_file()],
        "history": history,
        "seconds": time.time() - started,
        "model_metadata": model_metadata,
        "data_dir": str(data_dir.resolve()) if data_dir is not None else None,
        "dataset": dataset_metadata,
        "class_weight": {"mode": class_weight_mode, "positives_train_effective": positives, "negatives_train_effective": negatives, "pos_weight": float(pos_weight.detach().cpu()) if pos_weight is not None else None},
        "data_manifest_sha256": data_hashes["manifest"],
        "data_split_sha256": data_hashes,
        "limits": limits,
        "planned_total_epochs": planned_total_epochs,
        "total_training_steps": total_steps,
        "warmup_steps": warmup_steps,
        "steps_per_epoch": steps_per_epoch,
        "resumed_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "thresholds": threshold_rows,
        "validation_ranking_metrics": validation_ranking_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "repository_git_commit": os.environ.get("REPO_GIT_COMMIT"),
            "code_tree_sha256": os.environ.get("CODE_TREE_SHA256"),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    write_json(output_dir / "test_metrics.json", test_metrics)
    return run_manifest
