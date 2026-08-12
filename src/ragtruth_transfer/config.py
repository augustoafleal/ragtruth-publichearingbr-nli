from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoRASettings:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSettings:
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_epochs: int
    planned_total_epochs: int
    stop_after_epoch: int | None
    early_stopping_patience: int
    max_grad_norm: float
    use_class_weight: bool
    task_balanced_sampler: bool
    num_workers: int
    save_epoch_checkpoints: bool
    class_weight_mode: str = "auto_pos_weight"
    fixed_pos_weight: float | None = None

    def __post_init__(self) -> None:
        if self.max_epochs != self.planned_total_epochs:
            raise ValueError("max_epochs deve ser igual a planned_total_epochs")
        if self.planned_total_epochs < 1:
            raise ValueError("planned_total_epochs deve ser positivo")
        if self.stop_after_epoch is not None and not 1 <= self.stop_after_epoch <= self.planned_total_epochs:
            raise ValueError("stop_after_epoch deve estar entre 1 e planned_total_epochs")
        if self.class_weight_mode not in {"none", "auto_pos_weight", "fixed"}:
            raise ValueError("class_weight_mode deve ser none, auto_pos_weight ou fixed")
        if self.class_weight_mode == "fixed" and (self.fixed_pos_weight is None or self.fixed_pos_weight <= 0):
            raise ValueError("fixed_pos_weight deve ser positivo no modo fixed")


@dataclass(frozen=True)
class DatasetSettings:
    format: str = "jsonl"
    path: Path | None = None
    manifest_path: Path | None = None
    expected_signature: str | None = None
    expected_schema: str | None = "ragtruth-qa-training-view-deduplicated-v1"
    validation_fraction: float = 0.15
    split_seed: int = 42
    max_train_sources: int | None = None
    max_validation_sources: int | None = None
    max_test_sources: int | None = None
    evaluate_test: bool = True
    claim_column: str = "claim"
    chunk_columns: tuple[str, ...] = ("chunk_1", "chunk_2", "chunk_3", "chunk_4")
    evidence_mask_column: str = "evidence_mask"
    label_column: str = "label"
    group_column: str = "source_id"
    split_column: str = "split"

    def __post_init__(self) -> None:
        if self.format not in {"jsonl", "parquet"}:
            raise ValueError("dataset.format deve ser jsonl ou parquet")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("dataset.validation_fraction deve estar entre 0 e 1")
        if self.format == "parquet" and self.path is None:
            raise ValueError("dataset.path é obrigatório para formato parquet")
        if len(self.chunk_columns) != 4:
            raise ValueError("dataset.chunk_columns deve conter exatamente quatro colunas")


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    model_id: str
    model_revision: str | None
    architecture: str
    encoder_mode: str
    max_length: int
    projection_size: int
    attention_size: int
    dropout: float
    gradient_checkpointing: bool
    lora: LoRASettings
    training: TrainingSettings
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    output_root: Path | None = None
    pooling_type: str | None = None

    @property
    def canonical_pooling_type(self) -> str:
        return self.pooling_type or (
            "attention" if self.architecture == "gated_attention" else self.architecture
        )

    @staticmethod
    def from_mapping(raw: dict[str, Any], base_dir: Path | None = None) -> "ExperimentConfig":
        architecture = str(raw.get("architecture", "gated_attention"))
        raw_pooling_type = raw.get("pooling_type")
        pooling_type: str | None = None
        if raw_pooling_type is not None:
            pooling_type = str(raw_pooling_type)
            if pooling_type not in {"attention", "mean", "max"}:
                raise ValueError(f"pooling_type inválido: {pooling_type}")
            expected_architecture = (
                "gated_attention" if pooling_type == "attention" else pooling_type
            )
            if "architecture" in raw and architecture not in {
                expected_architecture,
                "attention" if pooling_type == "attention" else expected_architecture,
            }:
                raise ValueError("architecture e pooling_type são incompatíveis")
            architecture = expected_architecture
        elif architecture == "attention":
            architecture = "gated_attention"
            pooling_type = "attention"
        elif architecture in {"mean", "max"}:
            pooling_type = architecture
        if architecture not in {"mean", "max", "gated_attention"}:
            raise ValueError(f"Arquitetura inválida: {architecture}")
        encoder_mode = str(raw.get("encoder_mode", "lora"))
        if encoder_mode not in {"lora", "frozen"}:
            raise ValueError(f"encoder_mode inválido: {encoder_mode}")

        lora_raw = raw.get("lora", {})
        training_raw = raw["training"]
        dataset_raw = raw.get("dataset", {})

        def resolve(value: Any) -> Path | None:
            if value is None:
                return None
            path = Path(str(value)).expanduser()
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            return path.resolve()

        class_weight_raw = training_raw.get("class_weight", {})
        if not isinstance(class_weight_raw, dict):
            raise ValueError("training.class_weight deve ser um mapa")
        legacy_use_class_weight = bool(training_raw.get("use_class_weight", True))
        class_weight_mode = str(class_weight_raw.get("mode", "auto_pos_weight" if legacy_use_class_weight else "none"))
        fixed_pos_weight = class_weight_raw.get("pos_weight")
        return ExperimentConfig(
            run_name=str(raw["run_name"]),
            model_id=str(raw["model_id"]),
            model_revision=(str(raw["model_revision"]) if raw.get("model_revision") else None),
            architecture=architecture,
            encoder_mode=encoder_mode,
            max_length=int(raw.get("max_length", 512)),
            projection_size=int(raw.get("projection_size", 128)),
            attention_size=int(raw.get("attention_size", 64)),
            dropout=float(raw.get("dropout", 0.3)),
            gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
            lora=LoRASettings(
                r=int(lora_raw.get("r", 8)),
                alpha=int(lora_raw.get("alpha", 16)),
                dropout=float(lora_raw.get("dropout", 0.1)),
                target_modules=tuple(lora_raw.get("target_modules", ["query_proj", "value_proj"])),
            ),
            training=TrainingSettings(
                train_batch_size=int(training_raw.get("train_batch_size", 1)),
                eval_batch_size=int(training_raw.get("eval_batch_size", 4)),
                gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 16)),
                head_learning_rate=float(training_raw.get("head_learning_rate", 5e-4)),
                encoder_learning_rate=float(training_raw.get("encoder_learning_rate", 2e-4)),
                weight_decay=float(training_raw.get("weight_decay", 0.01)),
                warmup_ratio=float(training_raw.get("warmup_ratio", 0.1)),
                max_epochs=int(training_raw.get("max_epochs", 5)),
                planned_total_epochs=int(
                    training_raw.get("planned_total_epochs", training_raw.get("max_epochs", 5))
                ),
                stop_after_epoch=(
                    int(training_raw["stop_after_epoch"])
                    if training_raw.get("stop_after_epoch") is not None
                    else None
                ),
                early_stopping_patience=int(training_raw.get("early_stopping_patience", 2)),
                max_grad_norm=float(training_raw.get("max_grad_norm", 1.0)),
                use_class_weight=bool(training_raw.get("use_class_weight", True)),
                task_balanced_sampler=bool(training_raw.get("task_balanced_sampler", True)),
                num_workers=int(training_raw.get("num_workers", 0)),
                save_epoch_checkpoints=bool(training_raw.get("save_epoch_checkpoints", False)),
                class_weight_mode=class_weight_mode,
                fixed_pos_weight=(float(fixed_pos_weight) if fixed_pos_weight is not None else None),
            ),
            dataset=DatasetSettings(
                format=str(dataset_raw.get("format", raw.get("dataset_format", "jsonl"))),
                path=resolve(dataset_raw.get("path", raw.get("dataset_path"))),
                manifest_path=resolve(dataset_raw.get("manifest_path")),
                expected_signature=(str(dataset_raw["expected_signature"]) if dataset_raw.get("expected_signature") else None),
                expected_schema=(str(dataset_raw["expected_schema"]) if dataset_raw.get("expected_schema") else None),
                validation_fraction=float(dataset_raw.get("validation_fraction", 0.15)),
                split_seed=int(dataset_raw.get("split_seed", 42)),
                max_train_sources=(int(dataset_raw["max_train_sources"]) if dataset_raw.get("max_train_sources") is not None else None),
                max_validation_sources=(int(dataset_raw["max_validation_sources"]) if dataset_raw.get("max_validation_sources") is not None else None),
                max_test_sources=(int(dataset_raw["max_test_sources"]) if dataset_raw.get("max_test_sources") is not None else None),
                evaluate_test=bool(dataset_raw.get("evaluate_test", True)),
                claim_column=str(dataset_raw.get("claim_column", "claim")),
                chunk_columns=tuple(dataset_raw.get("chunk_columns", ["chunk_1", "chunk_2", "chunk_3", "chunk_4"])),
                evidence_mask_column=str(dataset_raw.get("evidence_mask_column", "evidence_mask")),
                label_column=str(dataset_raw.get("label_column", "label")),
                group_column=str(dataset_raw.get("group_column", "source_id")),
                split_column=str(dataset_raw.get("split_column", "split")),
            ),
            output_root=resolve(raw.get("output_root")),
            pooling_type=pooling_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "output_root": str(self.output_root) if self.output_root else None,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "architecture": self.architecture,
            **(
                {"pooling_type": self.canonical_pooling_type}
                if self.pooling_type is not None
                else {}
            ),
            "encoder_mode": self.encoder_mode,
            "max_length": self.max_length,
            "projection_size": self.projection_size,
            "attention_size": self.attention_size,
            "dropout": self.dropout,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora": {
                "r": self.lora.r,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": list(self.lora.target_modules),
            },
            "training": {
                "train_batch_size": self.training.train_batch_size,
                "eval_batch_size": self.training.eval_batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "head_learning_rate": self.training.head_learning_rate,
                "encoder_learning_rate": self.training.encoder_learning_rate,
                "weight_decay": self.training.weight_decay,
                "warmup_ratio": self.training.warmup_ratio,
                "max_epochs": self.training.max_epochs,
                "planned_total_epochs": self.training.planned_total_epochs,
                "stop_after_epoch": self.training.stop_after_epoch,
                "early_stopping_patience": self.training.early_stopping_patience,
                "max_grad_norm": self.training.max_grad_norm,
                "use_class_weight": self.training.use_class_weight,
                "task_balanced_sampler": self.training.task_balanced_sampler,
                "num_workers": self.training.num_workers,
                "save_epoch_checkpoints": self.training.save_epoch_checkpoints,
                "class_weight_mode": self.training.class_weight_mode,
                "fixed_pos_weight": self.training.fixed_pos_weight,
            },
            "dataset": {
                "format": self.dataset.format,
                "path": str(self.dataset.path) if self.dataset.path else None,
                "manifest_path": str(self.dataset.manifest_path) if self.dataset.manifest_path else None,
                "expected_signature": self.dataset.expected_signature,
                "expected_schema": self.dataset.expected_schema,
                "validation_fraction": self.dataset.validation_fraction,
                "split_seed": self.dataset.split_seed,
                "max_train_sources": self.dataset.max_train_sources,
                "max_validation_sources": self.dataset.max_validation_sources,
                "max_test_sources": self.dataset.max_test_sources,
                "evaluate_test": self.dataset.evaluate_test,
                "claim_column": self.dataset.claim_column,
                "chunk_columns": list(self.dataset.chunk_columns),
                "evidence_mask_column": self.dataset.evidence_mask_column,
                "label_column": self.dataset.label_column,
                "group_column": self.dataset.group_column,
                "split_column": self.dataset.split_column,
            },
        }


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração inválida: {path}")
    return ExperimentConfig.from_mapping(raw, base_dir=path.parent)
