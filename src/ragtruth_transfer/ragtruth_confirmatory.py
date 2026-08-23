
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .io_utils import sha256_file, write_json
from .metrics import binary_metrics, select_threshold
from .modeling import build_model, load_head_state
from .ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet
from .ragtruth_zero_shot import ZeroShotConfig, load_zero_shot_config, run_zero_shot
from .training import make_loader, predict, prepare_training_data, train_run, validate_training_data

CONFIRMATORY_SCHEMA = "ragtruth-confirmatory-three-seed-v1"


@dataclass(frozen=True)
class ConfirmatoryConfig:
    experiment: ExperimentConfig
    seeds: tuple[int, ...] = (0, 1, 2)
    split_seed: int = 42
    selection_metric: str = "validation_AUPRC"
    selection_mode: str = "max"
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 4242
    expected_dataset_sha256: str | None = "357e05b08cdcc22b766dce432fd8ed5caa7703ddf144dc02da24ef63e7ff0a7c"
    expected_dataset_signature: str = "0cdf598fa866741d"
    expected_schema: str = "ragtruth-qa-training-view-deduplicated-v1"
    expected_split_signature: str | None = "525edec2966a4fac"
    zero_shot_config_path: Path | None = None
    reference_dir: Path | None = None
    protocol_name: str = "ragtruth_confirmatory"

    @classmethod
    def from_yaml(cls, path: Path) -> "ConfirmatoryConfig":
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")
        base = path.parent.resolve()
        campaign = raw.get("campaign", {})
        if not isinstance(campaign, dict):
            raise ValueError("campaign deve ser um mapa")
        evaluation = raw.get("evaluation", {})
        if evaluation.get("evaluate_test_during_training", False) or evaluation.get("evaluate_publichearing_during_training", False):
            raise ValueError("O treino confirmatório não pode avaliar test/PublicHearing.")
        experiment = ExperimentConfig.from_mapping(raw, base_dir=base)
        if experiment.dataset.format not in {"parquet", "jsonl"} or experiment.dataset.evaluate_test:
            raise ValueError("A configuração confirmatória deve usar Parquet/JSONL e evaluate_test=false.")
        seeds = tuple(int(seed) for seed in campaign.get("seeds", [0, 1, 2]))
        if seeds != (0, 1, 2):
            raise ValueError("A campanha confirmatória exige exatamente seeds [0, 1, 2].")
        if experiment.dataset.split_seed != int(campaign.get("split_seed", experiment.dataset.split_seed)):
            raise ValueError("split_seed da campanha difere do dataset.")
        if experiment.training.planned_total_epochs != int(campaign.get("max_epochs", experiment.training.planned_total_epochs)):
            raise ValueError("max_epochs da campanha difere do training.planned_total_epochs.")
        if experiment.training.early_stopping_patience != int(campaign.get("early_stopping_patience", experiment.training.early_stopping_patience)):
            raise ValueError("early_stopping_patience da campanha difere do training.")
        def resolve(value: Any) -> Path | None:
            if value is None:
                return None
            value_path = Path(str(value)).expanduser()
            return (base / value_path).resolve() if not value_path.is_absolute() else value_path.resolve()
        return cls(
            experiment=experiment,
            seeds=seeds,
            split_seed=int(campaign.get("split_seed", experiment.dataset.split_seed)),
            selection_metric=str(campaign.get("selection_metric", "validation_AUPRC")),
            selection_mode=str(campaign.get("selection_mode", "max")),
            bootstrap_repetitions=int(campaign.get("bootstrap_repetitions", 2000)),
            bootstrap_seed=int(campaign.get("bootstrap_seed", 4242)),
            expected_dataset_sha256=(str(raw["expected_dataset_sha256"]) if raw.get("expected_dataset_sha256") else (None if "expected_dataset_sha256" in raw else cls.expected_dataset_sha256)),
            expected_dataset_signature=str(raw.get("expected_dataset_signature", cls.expected_dataset_signature)),
            expected_schema=str(raw.get("expected_schema", cls.expected_schema)),
            expected_split_signature=(str(raw["expected_split_signature"]) if raw.get("expected_split_signature") else (None if "expected_split_signature" in raw else cls.expected_split_signature)),
            zero_shot_config_path=resolve(raw.get("zero_shot_config")),
            reference_dir=resolve(raw.get("reference_dir")),
            protocol_name=str(raw.get("protocol_name", cls.protocol_name)),
        )

    @property
    def output_root(self) -> Path:
        return (self.experiment.output_root or Path("runs"))

    def protocol_payload(self, split_signature: str, dataset_sha256: str | None = None) -> dict[str, Any]:
        scientific_config = copy.deepcopy(self.experiment.to_dict())
        # Absolute filesystem paths are operational, not scientific.  Removing
        # them keeps the protocol signature identical across cluster/local
        # machines while dataset SHA/lineage remain explicit below.
        scientific_config["output_root"] = None
        # New ablation configs use the canonical pooling field, so their
        # run_name remains operational rather than a scientific difference.
        # Legacy attention keeps its established signature unchanged.
        if self.experiment.pooling_type is not None:
            scientific_config.pop("run_name", None)
        dataset_config = scientific_config.get("dataset", {})
        if isinstance(dataset_config, dict):
            dataset_config["path"] = "dataset.parquet" if self.experiment.dataset.format == "parquet" else "dataset.jsonl"
            dataset_config["manifest_path"] = "manifest.json"
        payload = {
            "schema": CONFIRMATORY_SCHEMA,
            "protocol_name": self.protocol_name,
            "dataset_sha256": dataset_sha256 or self.expected_dataset_sha256,
            "dataset_signature": self.expected_dataset_signature,
            "dataset_schema": self.expected_schema,
            "split_signature": split_signature,
            "split_seed": self.split_seed,
            "model_seeds": list(self.seeds),
            "experiment_config": scientific_config,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "screening_excluded": True,
            "test_used_during_training": False,
            "publichearing_used_during_training": False,
        }
        # Keep the legacy attention protocol signature stable while making the
        # new ablation protocols explicit and independently identifiable.
        if self.experiment.pooling_type is not None:
            payload["pooling_type"] = self.experiment.canonical_pooling_type
        return payload


def _signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:16]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _campaign_dir(config: ConfirmatoryConfig, protocol_signature: str) -> Path:
    return (config.output_root / protocol_signature).resolve()


def _validate_data(config: ConfirmatoryConfig) -> tuple[dict[str, Any], str]:
    path = config.experiment.dataset.path
    if path is None:
        raise ValueError("dataset.path ausente")
    if config.experiment.dataset.format == "jsonl":
        if not path.is_dir():
            raise ValueError(f"dataset.path JSONL deve ser um diretório: {path}")
        manifest_path = config.experiment.dataset.manifest_path or (path / "manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifesto JSONL não encontrado: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            raise ValueError("Manifesto JSONL ausente, inválido ou não concluído.")
        if manifest.get("schema_version") != config.expected_schema:
            raise ValueError("Schema do dataset JSONL não coincide com o protocolo confirmatório.")
        if manifest.get("run_signature") != config.expected_dataset_signature:
            raise ValueError("Assinatura do dataset JSONL não coincide com o protocolo confirmatório.")
        output_hashes = manifest.get("output_hashes")
        if not isinstance(output_hashes, dict):
            raise ValueError("Manifesto JSONL não contém output_hashes.")
        for split in ("train", "validation", "test"):
            split_path = path / f"{split}.jsonl"
            if not split_path.is_file() or sha256_file(split_path) != str(output_hashes.get(split)):
                raise ValueError(f"Hash do split JSONL divergente: {split}")
        dataset_sha256 = str(manifest.get("dataset_sha256", ""))
        split_signature = str(manifest.get("split_signature", ""))
        if config.expected_dataset_sha256 and dataset_sha256 != config.expected_dataset_sha256:
            raise ValueError("SHA-256 agregado do dataset JSONL não coincide com o protocolo confirmatório.")
        if split_signature != config.expected_split_signature:
            raise ValueError(f"Assinatura da divisão divergente: {split_signature}")
        validation = validate_training_data(config.experiment, path)
        expected_counts = manifest.get("counts", {}).get("per_split", {})
        for split in ("train", "validation", "test"):
            expected = expected_counts.get(split, {}).get("remaining_examples")
            actual = validation["partitions"][split]["examples"]
            if expected is None or int(expected) != int(actual):
                raise ValueError(f"Contagem do split JSONL divergente em {split}: {actual} != {expected}")
        metadata = {
            "dataset_format": "jsonl",
            "path": str(path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "dataset_sha256": dataset_sha256,
            "signature": str(manifest["run_signature"]),
            "schema_version": str(manifest["schema_version"]),
            "output_hashes": output_hashes,
        }
        audit = {
            "metadata": metadata,
            "split": {"strategy": "precomputed_jsonl_split", "signature": split_signature, "seed": config.split_seed},
            "partitions": validation["partitions"],
            "rows": sum(item["examples"] for item in validation["partitions"].values()),
        }
        return audit, split_signature
    rows, metadata = load_ragtruth_parquet(path, manifest_path=config.experiment.dataset.manifest_path, expected_signature=config.expected_dataset_signature, expected_schema=config.expected_schema, claim_column=config.experiment.dataset.claim_column, chunk_columns=config.experiment.dataset.chunk_columns, evidence_mask_column=config.experiment.dataset.evidence_mask_column, label_column=config.experiment.dataset.label_column, group_column=config.experiment.dataset.group_column, split_column=config.experiment.dataset.split_column)
    if config.expected_dataset_sha256 and metadata["dataset_sha256"] != config.expected_dataset_sha256:
        raise ValueError("SHA-256 do dataset não coincide com o protocolo confirmatório.")
    split = split_ragtruth_parquet(rows, metadata, validation_fraction=config.experiment.dataset.validation_fraction, split_seed=config.split_seed, max_test_sources=None)
    if config.expected_split_signature and split.metadata["signature"] != config.expected_split_signature:
        raise ValueError(f"Assinatura da divisão divergente: {split.metadata['signature']}")
    return {"metadata": metadata, "split": split.metadata, "rows": len(rows)}, split.metadata["signature"]


def _freeze_protocol(config: ConfirmatoryConfig, protocol_signature: str, data_audit: dict[str, Any]) -> Path:
    campaign_dir = _campaign_dir(config, protocol_signature)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    payload = config.protocol_payload(data_audit["split"]["signature"], data_audit["metadata"]["dataset_sha256"])
    frozen = {"schema_version": CONFIRMATORY_SCHEMA, "signature": protocol_signature, "payload": payload, "data_audit": data_audit}
    path = campaign_dir / "frozen_protocol.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise RuntimeError("Protocolo congelado divergente; não altere uma campanha iniciada.")
    _atomic_json(path, frozen)
    _atomic_json(campaign_dir / "resolved_config.json", config.experiment.to_dict())
    return campaign_dir


def _seed_state(path: Path, state: str, **extra: Any) -> None:
    _atomic_json(path / "state.json", {"state": state, "updated_at": time.time(), **extra})


def _seed_manifest_valid(seed_dir: Path) -> bool:
    manifest_path = seed_dir / "run_manifest.json"
    if not manifest_path.is_file() or not (seed_dir / "validation_predictions.csv").is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return False
    checkpoint = seed_dir / str(manifest.get("best_checkpoint", ""))
    marker = checkpoint / "CHECKPOINT_COMPLETE"
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    if not checkpoint.is_dir() or not marker.is_file() or not checkpoint_manifest_path.is_file():
        return False
    if marker.read_text(encoding="ascii").strip() != sha256_file(checkpoint_manifest_path):
        return False
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    for relative, expected in dict(checkpoint_manifest.get("artifacts", {})).items():
        candidate = checkpoint / relative
        if not candidate.is_file() or sha256_file(candidate) != str(expected):
            return False
    return True


def validate_only(config: ConfirmatoryConfig) -> dict[str, Any]:
    audit, split_signature = _validate_data(config)
    payload = config.protocol_payload(split_signature, audit["metadata"]["dataset_sha256"])
    return {"status": "valid", "model_loaded": False, "cuda_initialized": False, "training_executed": False, "evaluation_executed": False, "protocol_signature": _signature(payload), "seeds": list(config.seeds), "data_audit": audit, "commands": {"train": "--phase train", "evaluate": "--phase evaluate", "aggregate": "--phase aggregate"}}


def train_phase(config: ConfirmatoryConfig, *, resume: bool = False) -> dict[str, Any]:
    data_audit, split_signature = _validate_data(config)
    protocol_signature = _signature(config.protocol_payload(split_signature, data_audit["metadata"]["dataset_sha256"]))
    campaign_dir = _freeze_protocol(config, protocol_signature, data_audit)
    # Persist the one canonical source-group assignment for the campaign.
    prepare_training_data(config.experiment, None, output_dir=campaign_dir)
    _atomic_json(campaign_dir / "manifest.json", {"status": "training", "signature": protocol_signature, "pooling_type": config.experiment.canonical_pooling_type, "seeds": list(config.seeds), "data_audit": data_audit})
    for seed in config.seeds:
        seed_dir = campaign_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if resume and _seed_manifest_valid(seed_dir):
            _seed_state(seed_dir, "trained", seed=seed, reused=True)
            continue
        _seed_state(seed_dir, "training", seed=seed)
        seed_config = replace(config.experiment, run_name=f"{config.experiment.run_name}_seed_{seed}")
        try:
            resume_checkpoint = None
            if resume:
                completed_checkpoints = sorted(seed_dir.glob("checkpoints/epoch_*/CHECKPOINT_COMPLETE"))
                if completed_checkpoints:
                    resume_checkpoint = completed_checkpoints[-1].parent
            train_run(seed_config, data_dir=None, output_dir=seed_dir, seed=seed, resume_from_checkpoint=resume_checkpoint)
            if not _seed_manifest_valid(seed_dir):
                raise RuntimeError(f"Artefatos incompletos para seed {seed}")
            validation_csv = seed_dir / "validation_predictions.csv"
            pd.read_csv(validation_csv).to_parquet(seed_dir / "validation_predictions.parquet", index=False)
            seed_run_manifest = json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8"))
            write_json(seed_dir / "validation_metrics.json", {
                "threshold_free": seed_run_manifest.get("validation_ranking_metrics", {}),
                "best_f1": seed_run_manifest.get("validation_metrics", {}).get("f1", {}),
                "fpr10": seed_run_manifest.get("validation_metrics", {}).get("fpr10", {}),
            })
            write_json(seed_dir / "manifest.json", {"status": "trained", "pooling_type": config.experiment.canonical_pooling_type, "seed": seed, "best_epoch": seed_run_manifest.get("best_epoch"), "best_checkpoint": seed_run_manifest.get("best_checkpoint"), "run_manifest_sha256": sha256_file(seed_dir / "run_manifest.json")})
            (seed_dir / "run_log.jsonl").write_text(json.dumps({"event": "trained", "seed": seed}, ensure_ascii=False) + "\n", encoding="utf-8")
            _seed_state(seed_dir, "trained", seed=seed)
        except Exception as error:
            _seed_state(seed_dir, "failed", seed=seed, error=repr(error))
            raise
    _atomic_json(campaign_dir / "manifest.json", {"status": "trained", "signature": protocol_signature, "pooling_type": config.experiment.canonical_pooling_type, "seeds": list(config.seeds), "data_audit": data_audit})
    return {"status": "trained", "protocol_signature": protocol_signature, "campaign_dir": str(campaign_dir), "seeds": list(config.seeds)}


def _thresholds(seed_dir: Path) -> dict[str, Any]:
    path = seed_dir / "validation_predictions.csv"
    frame = pd.read_csv(path)
    labels = frame["label"].astype(bool).to_numpy()
    scores = frame["score"].astype(float).to_numpy()
    result: dict[str, Any] = {}
    for criterion in ("f1", "fpr10"):
        threshold, metrics, feasible = select_threshold(labels, scores, criterion)
        result[criterion] = {"seed": int(seed_dir.name.split("_")[-1]), "threshold": float(threshold), "validation_examples": len(frame), "validation_sha256": sha256_file(path), "development_metrics": metrics, "constraint_feasible": feasible, "rule": "select_threshold existing project implementation"}
    _atomic_json(seed_dir / "thresholds.json", result)
    return result


def _metric(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    value = binary_metrics(labels, scores >= threshold, scores)
    value["Brier"] = float(brier_score_loss(labels, scores))
    value["threshold"] = float(threshold)
    return value


def _group_bootstrap(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, thresholds: dict[str, float], repetitions: int, seed: int) -> dict[str, Any]:
    unique = np.unique(groups.astype(str)); rng = np.random.default_rng(seed); output: dict[str, Any] = {}
    for regime, threshold in thresholds.items():
        rows: list[dict[str, float]] = []
        invalid = 0
        for _ in range(repetitions):
            selected = rng.choice(unique, size=len(unique), replace=True)
            positions = np.concatenate([np.flatnonzero(groups.astype(str) == group) for group in selected])
            try:
                m = _metric(scores[positions], labels[positions], threshold)
            except ValueError:
                invalid += 1
                continue
            rows.append({key: float(m[key]) for key in ("AUPRC", "AUROC", "F1", "Recall", "Precision", "MCC", "FPR")})
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[regime] = {"invalid": invalid}
        else:
            output[regime] = {metric: {"lower": float(frame[metric].quantile(0.025)), "upper": float(frame[metric].quantile(0.975)), "invalid": invalid} for metric in frame.columns}
    return output


def _evaluate_ragtruth_test(config: ConfirmatoryConfig, seed_dir: Path, seed: int, thresholds: dict[str, Any]) -> dict[str, Any]:
    eval_dataset_config = replace(config.experiment.dataset, evaluate_test=True)
    eval_config = replace(config.experiment, dataset=eval_dataset_config)
    _, _, test_dataset, _, _ = prepare_training_data(eval_config, None, output_dir=None)
    checkpoint = seed_dir / json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8"))["best_checkpoint"]
    model, _ = build_model(eval_config, adapter_path=checkpoint / "adapter", adapter_trainable=False)
    load_head_state(model, checkpoint / "head.pt")
    tokenizer_path = seed_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.is_dir() else eval_config.model_id, revision=None if tokenizer_path.is_dir() else eval_config.model_revision, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    predictions = predict(model, make_loader(test_dataset, tokenizer, eval_config, False, seed), device)
    scores = predictions["score"].to_numpy(float); labels = predictions["label"].to_numpy(bool)
    output = {"threshold_free": {"AUPRC": float(average_precision_score(labels, scores)), "AUROC": float(roc_auc_score(labels, scores)), "Brier": float(brier_score_loss(labels, scores))}, "best_f1": _metric(scores, labels, thresholds["f1"]["threshold"]), "fpr10": _metric(scores, labels, thresholds["fpr10"]["threshold"]), "N": len(labels), "prevalence": float(labels.mean())}
    target = seed_dir / "ragtruth_test"; target.mkdir(exist_ok=True)
    enriched = predictions.assign(probability=scores, seed=seed, best_epoch=json.loads((seed_dir / "run_manifest.json").read_text())["best_epoch"], checkpoint_hash=sha256_file(checkpoint / "checkpoint_manifest.json"))
    enriched.to_parquet(target / "predictions.parquet", index=False)
    write_json(target / "metrics.json", output)
    write_json(target / "bootstrap_grouped.json", _group_bootstrap(labels, scores, enriched["source_id"].to_numpy(), {"best_f1": thresholds["f1"]["threshold"], "fpr10": thresholds["fpr10"]["threshold"]}, config.bootstrap_repetitions, config.bootstrap_seed + seed))
    return output


def evaluate_phase(config: ConfirmatoryConfig, *, resume: bool = False) -> dict[str, Any]:
    data_audit, split_signature = _validate_data(config); protocol_signature = _signature(config.protocol_payload(split_signature, data_audit["metadata"]["dataset_sha256"])); campaign_dir = _campaign_dir(config, protocol_signature)
    if not all(_seed_manifest_valid(campaign_dir / f"seed_{seed}") for seed in config.seeds):
        raise RuntimeError("A avaliação exige as três seeds treinadas e válidas.")
    for seed in config.seeds:
        seed_dir = campaign_dir / f"seed_{seed}"; _seed_state(seed_dir, "frozen", seed=seed)
        thresholds = _thresholds(seed_dir)
        ragtruth_metrics = _evaluate_ragtruth_test(config, seed_dir, seed, thresholds)
        write_json(seed_dir / "ragtruth_test" / "thresholds_applied.json", thresholds)
        if config.zero_shot_config_path is None:
            raise ValueError("zero_shot_config é obrigatório para evaluate.")
        zero_base = load_zero_shot_config(config.zero_shot_config_path)
        run_manifest = json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8")); checkpoint_rel = run_manifest["best_checkpoint"]
        zero_cfg = replace(zero_base, ragtruth_run_dir=seed_dir, checkpoint_relative_path=checkpoint_rel, validation_predictions_relative_path=f"{checkpoint_rel}/validation_predictions.csv", expected_run_signature=run_manifest.get("config_fingerprint"), expected_source_run_name=run_manifest["config"]["run_name"], expected_best_epoch=int(run_manifest["best_epoch"]), expected_ragtruth_dataset_signature=config.expected_dataset_signature, expected_ragtruth_schema=config.expected_schema, expected_ragtruth_split_signature=config.expected_split_signature, output_root=seed_dir / "publichearing_zero_shot", run_name=f"confirmatory_seed_{seed}", seed=seed)
        zero_manifest = run_zero_shot(zero_cfg, resume=resume)
        zero_dir = seed_dir / "publichearing_zero_shot" / zero_manifest["signature"]
        write_json(zero_dir / "thresholds_applied.json", thresholds)
        zero_predictions = pd.read_parquet(zero_dir / "predictions.parquet")
        write_json(zero_dir / "bootstrap_grouped.json", _group_bootstrap(zero_predictions["label"].astype(bool).to_numpy(), zero_predictions["probability"].to_numpy(float), zero_predictions["hearing_id"].astype(str).to_numpy(), {"best_f1": thresholds["f1"]["threshold"], "fpr10": thresholds["fpr10"]["threshold"]}, config.bootstrap_repetitions, config.bootstrap_seed + seed))
        _seed_state(seed_dir, "externally_evaluated", seed=seed, ragtruth_test=ragtruth_metrics, publichearing_signature=zero_manifest["signature"])
    _atomic_json(campaign_dir / "manifest.json", {"status": "externally_evaluated", "signature": protocol_signature, "pooling_type": config.experiment.canonical_pooling_type, "seeds": list(config.seeds), "data_audit": data_audit})
    return {"status": "externally_evaluated", "protocol_signature": protocol_signature, "campaign_dir": str(campaign_dir)}


def aggregate_phase(config: ConfirmatoryConfig) -> dict[str, Any]:
    data_audit, split_signature = _validate_data(config); protocol_signature = _signature(config.protocol_payload(split_signature, data_audit["metadata"]["dataset_sha256"])); campaign_dir = _campaign_dir(config, protocol_signature)
    if not all((campaign_dir / f"seed_{seed}" / "state.json").is_file() and json.loads((campaign_dir / f"seed_{seed}" / "state.json").read_text())["state"] == "externally_evaluated" for seed in config.seeds):
        raise RuntimeError("A agregação exige avaliações externas válidas para as três seeds.")
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_dir = campaign_dir / f"seed_{seed}"; thresholds = json.loads((seed_dir / "thresholds.json").read_text()); run = json.loads((seed_dir / "run_manifest.json").read_text())
        rt = json.loads((seed_dir / "ragtruth_test" / "metrics.json").read_text())
        zero_dirs = sorted((seed_dir / "publichearing_zero_shot").glob("*/metrics.json")); zm = json.loads(zero_dirs[-1].read_text())
        checkpoint_rel = run["best_checkpoint"]
        checkpoint_hash = sha256_file(seed_dir / checkpoint_rel / "checkpoint_manifest.json")
        public_metrics = {
            "threshold_free": zm.get("threshold_free", {}),
            "best_f1": zm.get("ragtruth_validation_best_f1_threshold", {}),
            "fpr10": zm.get("ragtruth_validation_fpr10_threshold", {}),
        }
        for dataset, metrics in (("ragtruth_test", rt), ("publichearing_zero_shot", public_metrics)):
            for regime in ("threshold_free", "best_f1", "fpr10"):
                value = metrics.get(regime, {})
                if not value: continue
                rows.append({"dataset": dataset, "pooling_type": config.experiment.canonical_pooling_type, "seed": seed, "threshold_regime": regime, "best_epoch": run["best_epoch"], "validation_AUPRC": run.get("best_validation_AUPRC"), "threshold": value.get("threshold"), "checkpoint_hash": checkpoint_hash, "run_signature": run.get("config_fingerprint"), **{key: value.get(key) for key in ("AUPRC", "AUROC", "Brier", "Precision", "Recall", "F1", "MCC", "FPR", "Specificity", "BalancedAccuracy")}})
    frame = pd.DataFrame(rows); aggregate = frame.groupby(["dataset", "threshold_regime"], as_index=False).agg({column: ["mean", "std", "min", "max"] for column in ["AUPRC", "AUROC", "Brier", "Precision", "Recall", "F1", "MCC", "FPR", "Specificity", "BalancedAccuracy"] if column in frame}).reset_index(drop=True)
    target = campaign_dir / "aggregate"; target.mkdir(parents=True, exist_ok=True); frame.to_csv(target / "per_seed_metrics.csv", index=False); aggregate.to_csv(target / "aggregate_metrics.csv", index=False)
    bootstrap = {str(seed): {"ragtruth_test": json.loads((campaign_dir / f"seed_{seed}" / "ragtruth_test" / "bootstrap_grouped.json").read_text()), "publichearing_zero_shot": json.loads(next((campaign_dir / f"seed_{seed}" / "publichearing_zero_shot").glob("*/bootstrap_grouped.json")).read_text())} for seed in config.seeds}
    _atomic_json(target / "bootstrap_grouped.json", bootstrap)
    comparison: dict[str, Any] = {"interpretation": "descriptive_only", "screening_excluded": True, "in_domain_reference_dir": str(config.reference_dir) if config.reference_dir else None, "baseline_max_entailment": {"AUPRC": 0.343, "AUROC": 0.769, "F1": 0.371, "FPR": 0.076, "MCC": 0.291}}
    if config.reference_dir:
        reference_metrics = config.reference_dir / "outputs" / "overall_oof_metrics.csv"
        if reference_metrics.is_file():
            reference_frame = pd.read_csv(reference_metrics)
            comparison["in_domain_confirmatory"] = reference_frame.to_dict(orient="records")
            comparison["in_domain_metrics_source"] = str(reference_metrics)
    _atomic_json(target / "comparison.json", comparison)
    _atomic_json(target / "protocol_summary.json", {"protocol_signature": protocol_signature, "pooling_type": config.experiment.canonical_pooling_type, "seeds": list(config.seeds), "screening_excluded": True, "no_best_seed_selection": True, "data_audit": data_audit})
    (target / "report.md").write_text(f"# RAGTruth confirmatory three-seed campaign\n\nPooling: `{config.experiment.canonical_pooling_type}`. This report is confirmatory only after all three frozen seeds have been trained and externally evaluated. The prior screening and preliminary zero-shot run are excluded from aggregation.\n", encoding="utf-8")
    _atomic_json(target / "manifest.json", {"schema_version": CONFIRMATORY_SCHEMA, "status": "completed", "signature": protocol_signature, "pooling_type": config.experiment.canonical_pooling_type, "artifacts": {str(path.name): sha256_file(path) for path in target.iterdir() if path.is_file()}})
    return {"status": "completed", "protocol_signature": protocol_signature, "campaign_dir": str(campaign_dir), "rows": len(frame)}


def run_confirmatory(config: ConfirmatoryConfig, phase: str, *, resume: bool = False, validate: bool = False) -> dict[str, Any]:
    if validate:
        return validate_only(config)
    if phase == "train": return train_phase(config, resume=resume)
    if phase == "evaluate": return evaluate_phase(config, resume=resume)
    if phase == "aggregate": return aggregate_phase(config)
    raise ValueError(f"Fase desconhecida: {phase}")
