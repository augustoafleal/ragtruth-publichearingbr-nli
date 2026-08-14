from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from ragtruth_transfer.config import ExperimentConfig
from ragtruth_transfer.io_utils import seed_everything
from ragtruth_transfer.modeling import PMA, SAB
from ragtruth_transfer.training import (
    _save_epoch_checkpoint,
    optimizer_for_model,
    validate_checkpoint,
)
from transformers import get_linear_schedule_with_warmup


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter_model.bin")
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")


class TinyModel(nn.Module):
    def __init__(self, set_transformer: bool = False) -> None:
        super().__init__()
        self.set_transformer = set_transformer
        self.encoder = TinyEncoder()
        dimension = 128 if set_transformer else 2
        self.projection = nn.Linear(3, dimension)
        if set_transformer:
            self.sab = SAB(dimension=128, num_heads=4, ffn_dim=128, dropout=0.2)
            self.pma = PMA(dimension=128, num_heads=4, ffn_dim=128, num_seeds=1, dropout=0.2)
        self.classifier = nn.Linear(dimension, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder.linear(x)
        hidden = self.dropout(hidden)
        projected = torch.tanh(self.projection(hidden))
        if self.set_transformer:
            values = projected.unsqueeze(1).expand(-1, 4, -1)
            evidence_mask = torch.ones((len(values), 4), dtype=torch.bool, device=values.device)
            contextualized = self.sab(values, evidence_mask)
            projected, _ = self.pma(contextualized, evidence_mask)
            projected = projected.squeeze(1)
        return self.classifier(projected).squeeze(-1)


def _config(architecture: str = "mean"):
    raw = {
        "run_name": "checkpoint_resume_test",
        "model_id": "tiny-test-encoder",
        "model_revision": "test-revision",
        "architecture": architecture,
        "encoder_mode": "lora",
        "max_length": 16,
        "projection_size": 2,
        "attention_size": 2,
        "dropout": 0.2,
        "gradient_checkpointing": False,
        "lora": {
            "r": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["linear"],
        },
        "training": {
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "head_learning_rate": 5e-4,
            "encoder_learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.2,
            "max_epochs": 5,
            "planned_total_epochs": 5,
            "stop_after_epoch": 2,
            "early_stopping_patience": 0,
            "max_grad_norm": 1.0,
            "use_class_weight": True,
            "task_balanced_sampler": True,
            "num_workers": 0,
            "save_epoch_checkpoints": True,
        },
    }
    if architecture == "set_transformer":
        raw.update({
            "pooling_type": "set_transformer",
            "projection_size": 128,
            "set_transformer": {
                "num_sab_layers": 1,
                "num_heads": 4,
                "num_seeds": 1,
                "ffn_dim": 128,
            },
        })
    return ExperimentConfig.from_mapping(raw)


def _segment(model, config, out, start, stop, resume_state=None):
    out.mkdir(parents=True, exist_ok=True)
    optimizer = optimizer_for_model(model, config)
    scheduler = get_linear_schedule_with_warmup(optimizer, 2, 10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler_generator = torch.Generator().manual_seed(101)
    loader_generator = torch.Generator().manual_seed(102)
    history = []
    best_metric = -float("inf")
    best_epoch = 0
    best_head = None
    best_adapter = None
    global_step = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        history = list(resume_state["history"])
        best_metric = float(resume_state["best_validation_AUPRC"])
        best_epoch = int(resume_state["best_epoch"])
        best_head = resume_state["best_head_state"]
        best_adapter = resume_state["best_adapter_state"]
        global_step = int(resume_state["global_step"])
        sampler_generator.set_state(resume_state["train_sampler_generator_state"])
        loader_generator.set_state(resume_state["train_loader_generator_state"])
        import random
        import numpy as np
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
    x = torch.tensor([[1.0, -1.0, 0.5], [0.2, 0.3, -0.4]])
    y = torch.tensor([1.0, 0.0])
    for epoch in range(start, stop + 1):
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x, y), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
        metric = 1.0 - epoch * 0.01
        history.append({"epoch": epoch, "train_loss": float(loss), "validation_AUPRC": metric})
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_head = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            best_adapter = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
        _save_epoch_checkpoint(
            out / f"epoch_{epoch:02d}", config, model, optimizer, scheduler, scaler, 101,
            epoch, epoch + 1, global_step, float(loss), metric,
            pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8]}), best_metric, best_epoch,
            history, "fp", {"config": {}}, {"train": "a", "validation": "b", "test": "c", "manifest": "d"},
            sampler_generator, loader_generator, best_head, best_adapter, {"tiny": True}, 10, 2, 2,
        )
    return model, optimizer, scheduler, history, global_step


def _load_tiny(checkpoint: Path, set_transformer: bool = False):
    manifest, state = validate_checkpoint(checkpoint, "fp")
    model = TinyModel(set_transformer=set_transformer)
    model.encoder.load_state_dict(torch.load(checkpoint / "adapter/adapter_model.bin", weights_only=True))
    model.load_state_dict(torch.load(checkpoint / "head.pt", weights_only=True), strict=False)
    return model, state


def test_continuous_and_stop_resume_are_bitwise_equivalent(tmp_path):
    config = _config()
    seed_everything(7)
    continuous, continuous_opt, continuous_sched, continuous_history, continuous_step = _segment(
        TinyModel(), config, tmp_path / "continuous", 1, 3
    )
    seed_everything(7)
    first_model, _, _, _, _ = _segment(TinyModel(), config, tmp_path / "split", 1, 1)
    resumed_model, state = _load_tiny(tmp_path / "split/epoch_01")
    resumed, resumed_opt, resumed_sched, resumed_history, resumed_step = _segment(
        resumed_model, config, tmp_path / "split", 2, 3, state
    )
    assert all(torch.equal(continuous.state_dict()[k], resumed.state_dict()[k]) for k in continuous.state_dict())
    assert continuous_history == resumed_history
    assert continuous_step == resumed_step == 6
    assert continuous_sched.state_dict() == resumed_sched.state_dict()
    assert [g["lr"] for g in continuous_opt.param_groups] == [g["lr"] for g in resumed_opt.param_groups]
    checkpoint_manifest, _ = validate_checkpoint(tmp_path / "split/epoch_01", "fp")
    assert checkpoint_manifest["planned_total_epochs"] == 5
    assert checkpoint_manifest["total_training_steps"] == 10
    assert checkpoint_manifest["warmup_steps"] == 2
    assert checkpoint_manifest["next_epoch"] == 2


def test_checkpoint_rejects_missing_marker_and_corruption(tmp_path):
    config = _config()
    seed_everything(3)
    _segment(TinyModel(), config, tmp_path / "run", 1, 1)
    checkpoint = tmp_path / "run/epoch_01"
    validate_checkpoint(checkpoint, "fp")
    (checkpoint / "CHECKPOINT_COMPLETE").unlink()
    with pytest.raises(RuntimeError, match="incompleto"):
        validate_checkpoint(checkpoint, "fp")


def test_checkpoint_rejects_fingerprint_mismatch(tmp_path):
    config = _config()
    seed_everything(3)
    _segment(TinyModel(), config, tmp_path / "run", 1, 1)
    with pytest.raises(RuntimeError, match="Fingerprint incompatível"):
        validate_checkpoint(tmp_path / "run/epoch_01", "different", {"config": {"seed": 202}})


def test_set_transformer_checkpoint_resume_keeps_explicit_config_and_state(tmp_path):
    config = _config("set_transformer")
    seed_everything(43)
    _segment(TinyModel(set_transformer=True), config, tmp_path / "split", 1, 1)
    resumed_model, state = _load_tiny(tmp_path / "split/epoch_01", set_transformer=True)
    resumed, _, _, history, _ = _segment(
        resumed_model, config, tmp_path / "split", 2, 2, state
    )
    manifest, _ = validate_checkpoint(tmp_path / "split/epoch_01", "fp")
    assert manifest["pooling_type"] == "set_transformer"
    assert manifest["config"]["set_transformer"]["num_heads"] == 4
    assert len(history) == 2
    assert any(parameter.requires_grad for parameter in resumed.parameters())
