from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import SetTransformerSettings


@dataclass(frozen=True)
class PublicHearingConfig:
    run_name: str = "publichearing_lora_attention_mil"
    output_root: str = "results/publichearing_lora_attention_mil"
    mode: str = "screening"
    architecture: str = "gated_attention"
    set_transformer: SetTransformerSettings | None = None
    dataset_repo: str = "unicamp-dl/PublicHearingBR"
    dataset_filename: str = "PublicHearingBR_NLI.jsonl"
    dataset_revision: str = "2f84a44bc34df483e25c987f0ff86caad0ab3433"
    model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    model_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    split_seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 5
    seeds: tuple[int, ...] = (101,)
    max_epochs: int = 5
    max_length: int = 512
    tokenization_batch_size: int = 256
    projection_size: int = 128
    attention_size: int = 64
    dropout: float = 0.30
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.10
    lora_target_modules: tuple[str, ...] = ("query_proj", "value_proj")
    train_batch_size: int = 1
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    lora_lr: float = 2e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    early_stopping_patience: int = 2
    max_grad_norm: float = 1.0
    use_class_weight: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    fpr_target: float = 0.10
    bootstrap_repetitions: int = 1000
    num_workers: int = 0
    include_checkpoints_in_zip: bool = False
    save_checkpoints: bool = True
    smoke_max_hearings: int | None = None
    smoke_outer_splits: int | None = None
    smoke_fold: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"screening", "confirmatory", "smoke"}:
            raise ValueError("mode deve ser screening, confirmatory ou smoke")
        if self.architecture not in {"gated_attention", "set_transformer"}:
            raise ValueError("architecture deve ser gated_attention ou set_transformer")
        if self.architecture == "set_transformer" and self.set_transformer is None:
            raise ValueError("set_transformer é obrigatório para architecture=set_transformer")
        if self.architecture != "set_transformer" and self.set_transformer is not None:
            raise ValueError("set_transformer só é aceito com architecture=set_transformer")
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits e inner_splits devem ser >= 2")
        if not self.seeds:
            raise ValueError("seeds não pode ser vazio")
        if self.mode == "screening" and self.seeds != (101,):
            raise ValueError("screening reproduzível usa seeds: [101]")
        if self.mode == "confirmatory" and self.seeds != (101, 202, 303):
            raise ValueError("confirmatory reproduzível usa seeds: [101, 202, 303]")
        if self.mode == "screening" and self.max_epochs != 5:
            raise ValueError("screening reproduzível usa max_epochs: 5")
        if self.mode == "confirmatory" and self.max_epochs != 8:
            raise ValueError("confirmatory reproduzível usa max_epochs: 8")
        if not 0.0 <= self.fpr_target <= 1.0:
            raise ValueError("fpr_target deve estar entre 0 e 1")
        if self.mode == "smoke" and self.smoke_fold < 0:
            raise ValueError("smoke_fold deve ser não negativo")

    @property
    def effective_outer_splits(self) -> int:
        return self.smoke_outer_splits or self.outer_splits

    def to_dict(self) -> dict[str, Any]:
        serialized = asdict(self)
        if self.architecture == "gated_attention" and self.set_transformer is None:
            # Keep the historical representation used by existing campaign signatures.
            serialized.pop("architecture")
            serialized.pop("set_transformer")
        return serialized


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"dataset", "model", "training", "lora", "cross_validation", "operation"}:
            if not isinstance(value, dict):
                raise ValueError(f"{key} deve ser um mapa YAML")
            merged.update(value)
        else:
            merged[key] = value
    aliases = {
        "repo": "dataset_repo", "filename": "dataset_filename", "revision": "dataset_revision",
        "learning_rate": "lora_lr", "head_learning_rate": "head_lr",
        "outer_fold_count": "outer_splits", "inner_fold_count": "inner_splits",
    }
    for old, new in aliases.items():
        if old in merged and new not in merged:
            merged[new] = merged.pop(old)
    for key in ("seeds", "lora_target_modules"):
        if key in merged:
            merged[key] = tuple(merged[key])
    return merged


def load_publichearing_config(path: Path, overrides: list[str] | None = None) -> PublicHearingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração inválida: {path}")
    normalized = _normalize(raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override inválido (use chave=valor): {override}")
        key, value = override.split("=", 1)
        if "." in key:
            key = key.rsplit(".", 1)[-1]
        normalized[key] = yaml.safe_load(value)
    normalized = _normalize(normalized)
    set_transformer_raw = normalized.get("set_transformer")
    if set_transformer_raw is not None:
        if not isinstance(set_transformer_raw, dict):
            raise ValueError("set_transformer deve ser um mapa")
        normalized["set_transformer"] = SetTransformerSettings(
            num_sab_layers=int(set_transformer_raw.get("num_sab_layers", 1)),
            num_heads=int(set_transformer_raw.get("num_heads", 4)),
            num_seeds=int(set_transformer_raw.get("num_seeds", 1)),
            ffn_dim=int(set_transformer_raw.get("ffn_dim", 128)),
        )
    allowed = set(PublicHearingConfig.__dataclass_fields__)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"Chaves de configuração desconhecidas: {unknown}")
    return PublicHearingConfig(**normalized)
