
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io_utils import sha256_file, write_json
from .metrics import binary_metrics, select_threshold
from .publichearing_off_the_shelf import MODEL_ID, MODEL_REVISION, _normalize_labels, _state_hash
from .ragtruth_confirmatory import ConfirmatoryConfig
from .ragtruth_parquet import load_ragtruth_parquet


SCHEMA_VERSION = "ragtruth-off-the-shelf-threshold-transfer-v1"
CODE_VERSION = "ragtruth-off-the-shelf-threshold-transfer-v2"
EXPECTED_SEEDS = (0, 1, 2)
REGIMES = ("best_f1", "fpr10")
OPERATOR = ">="


@dataclass(frozen=True)
class ThresholdTransferConfig:
    output_root: Path
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    max_length: int
    truncation: str
    padding: str
    batch_size: int
    device: str
    expected_label_mapping: tuple[tuple[str, int], ...]
    confirmatory_config_path: Path
    dataset_signature: str
    dataset_sha256: str
    split_signature: str
    split_seed: int
    expected_validation_examples: int
    expected_validation_sources: int
    expected_validation_positives: int
    target_fpr: float
    baseline_run: Path
    baseline_signature: str
    confirmatory_run: Path
    confirmatory_signature: str
    seeds: tuple[int, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "ThresholdTransferConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        protocol, model = raw.get("protocol", {}), raw.get("model", {})
        source, operating, comparison, output = (
            raw.get("source_domain", {}), raw.get("operating_points", {}), raw.get("comparison_inputs", {}), raw.get("output", {})
        )
        if protocol.get("analysis_type") != "source_domain_threshold_transfer":
            raise ValueError("protocol.analysis_type deve ser source_domain_threshold_transfer")
        if operating.get("best_f1_criterion") != "f1" or operating.get("fpr10_criterion") != "fpr10":
            raise ValueError("O protocolo exige os critérios confirmatórios f1 e fpr10")
        if operating.get("decision_operator") != OPERATOR or operating.get("threshold_selection_implementation") != "ragtruth_transfer.metrics.select_threshold":
            raise ValueError("A seleção deve reutilizar select_threshold e o operador >=")
        mapping = model.get("expected_label_mapping", {})
        config = cls(
            output_root=resolve(output.get("root", "../runs/ragtruth_off_the_shelf_threshold_transfer")),
            model_id=str(model.get("model_id")), model_revision=str(model.get("revision")),
            tokenizer_id=str(model.get("tokenizer_id")), tokenizer_revision=str(model.get("tokenizer_revision")),
            max_length=int(model.get("max_length", 512)), truncation=str(model.get("truncation", "only_first")),
            padding=str(model.get("padding", "longest")), batch_size=int(model.get("batch_size", 16)), device=str(model.get("device", "auto")),
            expected_label_mapping=tuple(sorted((str(key), int(value)) for key, value in mapping.items())),
            confirmatory_config_path=resolve(source.get("confirmatory_config")),
            dataset_signature=str(source.get("dataset_signature")), dataset_sha256=str(source.get("dataset_sha256")),
            split_signature=str(source.get("split_signature")), split_seed=int(source.get("split_seed", 42)),
            expected_validation_examples=int(source.get("expected_validation_examples", 4517)),
            expected_validation_sources=int(source.get("expected_validation_sources", 126)),
            expected_validation_positives=int(source.get("expected_validation_positives", 524)),
            target_fpr=float(operating.get("target_fpr", 0.10)),
            baseline_run=resolve(comparison.get("publichearing_baseline_run")), baseline_signature=str(comparison.get("publichearing_baseline_signature")),
            confirmatory_run=resolve(comparison.get("ragtruth_confirmatory_run")), confirmatory_signature=str(comparison.get("ragtruth_confirmatory_signature")),
            seeds=tuple(int(seed) for seed in comparison.get("lora_seeds", [])),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if (self.model_id, self.model_revision) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Modelo/revisão devem ser idênticos ao baseline NLI congelado")
        if (self.tokenizer_id, self.tokenizer_revision) != (self.model_id, self.model_revision):
            raise ValueError("Tokenizer deve usar o mesmo id/revisão do baseline")
        if self.max_length != 512 or self.truncation != "only_first" or self.padding not in {"longest", "max_length"} or self.batch_size < 1:
            raise ValueError("Contrato de tokenização inválido")
        if dict(self.expected_label_mapping) != {"entailment": 0, "neutral": 1, "contradiction": 2}:
            raise ValueError("Mapeamento NLI declarado incompatível")
        if self.dataset_signature != "0cdf598fa866741d" or self.split_signature != "525edec2966a4fac":
            raise ValueError("Linhas de dados/split congeladas divergentes")
        if self.expected_validation_examples != 4517 or self.expected_validation_sources != 126 or self.expected_validation_positives != 524:
            raise ValueError("Contagens de validation congeladas divergentes")
        if self.baseline_signature != "54d9c623f8685c39" or self.confirmatory_signature != "4e12933c51136624" or self.seeds != EXPECTED_SEEDS:
            raise ValueError("Runs de comparação congelados divergentes")
        if not np.isclose(self.target_fpr, 0.10):
            raise ValueError("O protocolo exige target_fpr=0.10")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {"schema": SCHEMA_VERSION, "analysis_type": "source_domain_threshold_transfer", "operator": OPERATOR},
            "model": {"model_id": self.model_id, "revision": self.model_revision, "tokenizer_id": self.tokenizer_id, "tokenizer_revision": self.tokenizer_revision, "loader": "AutoModelForSequenceClassification", "max_length": self.max_length, "truncation": self.truncation, "padding": self.padding, "batch_size": self.batch_size, "device": self.device, "premise": "chunk", "hypothesis": "claim", "aggregation": "max_entailment_over_valid_evidence", "score": "1 - max_entailment", "label_mapping_strategy": "dynamic_id2label_and_label2id", "expected_label_mapping": dict(self.expected_label_mapping), "lora_used": False},
            "source_domain": {"confirmatory_config": str(self.confirmatory_config_path), "dataset_signature": self.dataset_signature, "dataset_sha256": self.dataset_sha256, "split_signature": self.split_signature, "split_seed": self.split_seed, "expected_validation_examples": self.expected_validation_examples, "expected_validation_sources": self.expected_validation_sources, "expected_validation_positives": self.expected_validation_positives},
            "operating_points": {"best_f1_criterion": "f1", "fpr10_criterion": "fpr10", "target_fpr": self.target_fpr, "decision_operator": OPERATOR, "threshold_selection_implementation": "ragtruth_transfer.metrics.select_threshold"},
            "comparison_inputs": {"publichearing_baseline_run": str(self.baseline_run), "publichearing_baseline_signature": self.baseline_signature, "ragtruth_confirmatory_run": str(self.confirmatory_run), "ragtruth_confirmatory_signature": self.confirmatory_signature, "lora_seeds": list(self.seeds)},
            "output": {"root": str(self.output_root)},
        }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    write_json(temporary, value)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _file_hashes(directory: Path) -> dict[str, str]:
    return {item.name: sha256_file(item) for item in sorted(directory.iterdir()) if item.is_file() and item.name != "manifest.json"}


def _validate_manifest(directory: Path, signature: str) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto ausente: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("status") not in {"completed", "externally_evaluated"}:
        raise ValueError(f"Manifesto incompatível: {path}")
    for name, digest in manifest.get("artifacts", {}).items():
        artifact = directory / name
        if artifact.is_file() and sha256_file(artifact) != digest:
            raise ValueError(f"Hash divergente: {artifact}")
    return manifest


def _load_validation(config: ThresholdTransferConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confirmatory = ConfirmatoryConfig.from_yaml(config.confirmatory_config_path)
    if confirmatory.expected_dataset_signature != config.dataset_signature or confirmatory.expected_dataset_sha256 != config.dataset_sha256:
        raise ValueError("A configuração confirmatória não corresponde ao dataset congelado")
    if confirmatory.expected_split_signature != config.split_signature or confirmatory.split_seed != config.split_seed:
        raise ValueError("A configuração confirmatória não corresponde ao split congelado")
    dataset = confirmatory.experiment.dataset
    if dataset.path is None:
        raise ValueError("dataset.path ausente na configuração confirmatória")
    # This is the exact source assignment persisted by the confirmatory
    # campaign, rather than a new reconstruction of its search procedure.
    assignment_path = config.confirmatory_run / "split_assignments.parquet"
    if not assignment_path.is_file():
        raise FileNotFoundError(f"Atribuições de split congeladas ausentes: {assignment_path}")
    assignments = pd.read_parquet(assignment_path)
    required = {"source_id", "partition", "examples", "positives", "negatives", "split_seed", "split_signature", "algorithm_version"}
    if not required.issubset(assignments.columns) or assignments["source_id"].astype(str).duplicated().any():
        raise ValueError("split_assignments.parquet inválido")
    if set(assignments["partition"].astype(str)) != {"train", "validation"} or set(assignments["split_seed"].astype(int)) != {config.split_seed} or set(assignments["split_signature"].astype(str)) != {config.split_signature} or set(assignments["algorithm_version"].astype(str)) != {"group-split-v1"}:
        raise ValueError("Atribuições não correspondem ao split congelado")
    validation_assignments = assignments.loc[assignments["partition"].astype(str) == "validation"].copy()
    validation_sources = set(validation_assignments["source_id"].astype(str))
    validation, metadata = load_ragtruth_parquet(
        dataset.path, manifest_path=dataset.manifest_path, expected_signature=config.dataset_signature,
        expected_schema=confirmatory.expected_schema, claim_column=dataset.claim_column, chunk_columns=dataset.chunk_columns,
        evidence_mask_column=dataset.evidence_mask_column, label_column=dataset.label_column, group_column=dataset.group_column,
        split_column=dataset.split_column, source_ids=validation_sources,
    )
    if metadata["dataset_sha256"] != config.dataset_sha256:
        raise ValueError("SHA-256 do Parquet diverge da campanha confirmatória")
    if any(row["split"] != "train" for row in validation):
        raise ValueError("Validation confirmatório deve derivar somente do split train do RAGTruth")
    labels = np.asarray([row["label"] for row in validation], dtype=int)
    if len(validation) != config.expected_validation_examples or len({row["source_id"] for row in validation}) != config.expected_validation_sources or int(labels.sum()) != config.expected_validation_positives:
        raise ValueError("Validation não corresponde exatamente ao da campanha confirmatória")
    masks = np.asarray([row["evidence_mask"] for row in validation], dtype=bool)
    if masks.shape != (len(validation), 4) or not masks.any(axis=1).all():
        raise ValueError("Máscaras de evidência de validation inválidas")
    if int(validation_assignments["examples"].sum()) != len(validation) or int(validation_assignments["positives"].sum()) != int(labels.sum()) or int(validation_assignments["negatives"].sum()) != int((labels == 0).sum()):
        raise ValueError("Contagens do validation divergem de split_assignments.parquet")
    split_audit = {"strategy": "persisted_confirmatory_split_assignments", "assignment_path": str(assignment_path), "assignment_sha256": sha256_file(assignment_path), "signature": config.split_signature, "seed": config.split_seed, "algorithm_version": "group-split-v1", "validation_sources": len(validation_sources)}
    return validation, {"metadata": metadata, "split": split_audit, "validation_examples": len(validation), "validation_sources": len(validation_sources), "validation_positives": int(labels.sum()), "validation_example_ids_hash": _canonical_hash([row["example_id"] for row in validation]), "validation_evidence_masks_hash": _canonical_hash(masks.astype(int).tolist())}


def _threshold_regime(criterion: str) -> str:
    return "best_f1" if criterion == "f1" else "fpr10"


def _lora_threshold_audit(config: ThresholdTransferConfig, validation: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[int, dict[str, float]], dict[str, str]]:
    _validate_manifest(config.confirmatory_run, config.confirmatory_signature)
    expected = pd.DataFrame({"example_id": [str(row["example_id"]) for row in validation], "source_id": [str(row["source_id"]) for row in validation], "label": [int(row["label"]) for row in validation]}).set_index("example_id")
    audit: dict[str, Any] = {"selection_function": "ragtruth_transfer.metrics.select_threshold", "decision_operator": OPERATOR, "threshold_source": "RAGTruth validation only", "seeds": {}}
    thresholds: dict[int, dict[str, float]] = {}
    hashes: dict[str, str] = {}
    for seed in config.seeds:
        seed_dir = config.confirmatory_run / f"seed_{seed}"
        threshold_path, prediction_path = seed_dir / "thresholds.json", seed_dir / "validation_predictions.csv"
        if not threshold_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Threshold/predição de validation ausente para seed {seed}")
        frame = pd.read_csv(prediction_path)
        required = {"example_id", "source_id", "label", "score"}
        if not required.issubset(frame.columns) or len(frame) != len(expected) or not frame["example_id"].astype(str).is_unique:
            raise ValueError(f"Previsão validation inválida para seed {seed}")
        current = frame.assign(example_id=frame["example_id"].astype(str), source_id=frame["source_id"].astype(str)).set_index("example_id")
        if set(current.index) != set(expected.index) or not np.array_equal(current.loc[expected.index, "label"].astype(int), expected["label"].astype(int)) or not np.array_equal(current.loc[expected.index, "source_id"].astype(str), expected["source_id"].astype(str)):
            raise ValueError(f"Validation LoRA não corresponde ao split congelado: seed {seed}")
        score = current.loc[expected.index, "score"].to_numpy(dtype=float)
        if not np.isfinite(score).all() or not ((score >= 0).all() and (score <= 1).all()):
            raise ValueError(f"Scores LoRA inválidos na seed {seed}")
        frozen = json.loads(threshold_path.read_text(encoding="utf-8"))
        seed_thresholds: dict[str, float] = {}
        per_seed: dict[str, Any] = {"validation_predictions": str(prediction_path), "validation_predictions_sha256": sha256_file(prediction_path), "thresholds_path": str(threshold_path), "thresholds_sha256": sha256_file(threshold_path), "regimes": {}}
        for criterion in ("f1", "fpr10"):
            item = frozen.get(criterion, {})
            selected, metrics, feasible = select_threshold(expected["label"].to_numpy(dtype=bool), score, criterion, max_fpr=config.target_fpr)
            if not np.isclose(float(item.get("threshold", np.nan)), selected) or item.get("validation_sha256") != per_seed["validation_predictions_sha256"] or item.get("development_metrics") != metrics or bool(item.get("constraint_feasible")) != bool(feasible):
                raise ValueError(f"Threshold LoRA congelado não reproduz validation: seed {seed}/{criterion}")
            regime = _threshold_regime(criterion)
            seed_thresholds[regime] = float(selected)
            per_seed["regimes"][regime] = {"threshold": float(selected), "constraint_feasible": bool(feasible), "development_metrics": metrics, "frozen_development_metrics": item.get("development_metrics"), "rule": item.get("rule")}
        audit["seeds"][str(seed)] = per_seed
        thresholds[seed] = seed_thresholds
        hashes[f"seed_{seed}_validation_predictions.csv"] = per_seed["validation_predictions_sha256"]
        hashes[f"seed_{seed}_thresholds.json"] = per_seed["thresholds_sha256"]
    return audit, thresholds, hashes


def _load_publichearing_predictions(config: ThresholdTransferConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    baseline_manifest = _validate_manifest(config.baseline_run, config.baseline_signature)
    baseline_path = config.baseline_run / "predictions.parquet"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = pd.read_parquet(baseline_path)
    required = {"example_id", "hearing_id", "label", "hallucination_score"}
    if not required.issubset(baseline.columns):
        raise ValueError("Baseline PublicHearingBR sem colunas obrigatórias")
    baseline = baseline[list(required)].copy().rename(columns={"hallucination_score": "baseline_score"})
    baseline["example_id"] = baseline["example_id"].astype(str); baseline["hearing_id"] = baseline["hearing_id"].astype(str)
    if len(baseline) != 4235 or baseline["example_id"].duplicated().any() or baseline["hearing_id"].nunique() != 206 or int(baseline["label"].sum()) != 501 or not set(baseline["label"].unique()).issubset({0, 1}):
        raise ValueError("Baseline PublicHearingBR não corresponde ao run congelado")
    if not np.isfinite(baseline["baseline_score"].to_numpy(float)).all() or not ((baseline["baseline_score"] >= 0).all() and (baseline["baseline_score"] <= 1).all()):
        raise ValueError("Score do baseline PublicHearingBR inválido")
    base = baseline.set_index("example_id").sort_index()
    audit: dict[str, Any] = {"baseline_manifest_status": baseline_manifest.get("status"), "baseline_signature": config.baseline_signature, "seeds": {}}
    hashes = {"publichearing_baseline_predictions.parquet": sha256_file(baseline_path)}
    for seed in config.seeds:
        matches = sorted((config.confirmatory_run / f"seed_{seed}" / "publichearing_zero_shot").glob("*/predictions.parquet"))
        if len(matches) != 1:
            raise ValueError(f"Esperava uma previsão PublicHearingBR para seed {seed}, encontrei {len(matches)}")
        path = matches[0]; frame = pd.read_parquet(path)
        if not {"example_id", "hearing_id", "label", "probability"}.issubset(frame.columns):
            raise ValueError(f"Previsão PublicHearingBR inválida: {path}")
        frame = frame[["example_id", "hearing_id", "label", "probability"]].copy(); frame["example_id"] = frame["example_id"].astype(str); frame["hearing_id"] = frame["hearing_id"].astype(str)
        current = frame.set_index("example_id")
        base_only, seed_only = set(base.index) - set(current.index), set(current.index) - set(base.index)
        common = base.index.intersection(current.index)
        label_mismatch = int((base.loc[common, "label"].astype(int) != current.loc[common, "label"].astype(int)).sum())
        hearing_mismatch = int((base.loc[common, "hearing_id"].astype(str) != current.loc[common, "hearing_id"].astype(str)).sum())
        score = current.loc[base.index, "probability"].to_numpy(float) if not base_only and not seed_only else np.array([])
        status = {"path": str(path), "baseline_only_ids": len(base_only), "zero_shot_only_ids": len(seed_only), "duplicate_ids": int(frame["example_id"].duplicated().sum()), "label_mismatches": label_mismatch, "hearing_id_mismatches": hearing_mismatch, "pairable_examples": len(common)}
        if any(status[key] for key in ("baseline_only_ids", "zero_shot_only_ids", "duplicate_ids", "label_mismatches", "hearing_id_mismatches")) or len(common) != 4235 or not np.isfinite(score).all() or not ((score >= 0).all() and (score <= 1).all()):
            raise ValueError(f"Pareamento PublicHearingBR inválido para seed {seed}: {status}")
        base[f"seed_{seed}_score"] = score
        audit["seeds"][str(seed)] = status
        hashes[f"publichearing_seed_{seed}_predictions.parquet"] = sha256_file(path)
    return base.reset_index(), audit, hashes


def _signature(config: ThresholdTransferConfig, validation_audit: dict[str, Any], lora_hashes: dict[str, str], target_hashes: dict[str, str]) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": SCHEMA_VERSION, "code_version": CODE_VERSION, "model_id": config.model_id, "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer_id, "tokenizer_revision": config.tokenizer_revision, "ragtruth_dataset_signature": config.dataset_signature,
        "ragtruth_dataset_sha256": config.dataset_sha256, "split_signature": config.split_signature, "validation_example_ids_hash": validation_audit["validation_example_ids_hash"],
        "valid_evidence_masks_hash": validation_audit["validation_evidence_masks_hash"], "premise": "chunk", "hypothesis": "claim", "max_length": config.max_length,
        "truncation": config.truncation, "padding": config.padding, "nli_label_mapping_strategy": "dynamic_id2label_and_label2id", "expected_nli_label_mapping": dict(config.expected_label_mapping),
        "score_formula": "hallucination_score = 1 - max_entailment", "aggregation": "max_entailment_over_valid_evidence", "threshold_selection": "ragtruth_transfer.metrics.select_threshold",
        "threshold_candidates": "unique(clipped scores) plus 0, 1, nextafter(1, 2)", "best_f1_rule": "maximize F1; then Recall; then lower FPR; then higher threshold", "fpr10_rule": "among FPR<=0.10 maximize Recall, then F1, then lower FPR, then lower threshold; otherwise minimize FPR, then maximize Recall, then higher threshold", "target_fpr": config.target_fpr,
        "decision_operator": OPERATOR, "baseline_run_signature": config.baseline_signature, "confirmatory_run_signature": config.confirmatory_signature, "lora_seeds": list(config.seeds),
        "lora_artifact_hashes": lora_hashes, "publichearing_prediction_hashes": target_hashes,
    }
    return _canonical_hash(payload)[:16], payload


def _infer_validation(model: Any, tokenizer: Any, validation: list[dict[str, Any]], config: ThresholdTransferConfig, mapping: dict[str, int], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(validation)
    values = {name: np.full((n_rows, 4), np.nan, dtype=float) for name in ("entailment", "neutral", "contradiction")}
    with torch.inference_mode():
        for slot in range(4):
            valid_positions = [index for index, row in enumerate(validation) if bool(row["evidence_mask"][slot])]
            for start in range(0, len(valid_positions), config.batch_size):
                positions = valid_positions[start:start + config.batch_size]
                rows = [validation[index] for index in positions]
                encoded = tokenizer([row["evidence"][slot] for row in rows], [row["claim"] for row in rows], truncation=config.truncation, max_length=config.max_length, padding=True if config.padding == "longest" else "max_length", return_tensors="pt")
                tensors = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
                probabilities = torch.softmax(model(**tensors).logits.float(), dim=-1).cpu().numpy()
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                    raise RuntimeError("Probabilidades NLI inválidas")
                for name, label_index in mapping.items():
                    values[name][positions, slot] = probabilities[:, label_index]
    masks = np.asarray([row["evidence_mask"] for row in validation], dtype=bool)
    for value in values.values():
        if not np.isfinite(value[masks]).all() or not np.isnan(value[~masks]).all():
            raise RuntimeError("Contrato de evidências válidas violado")
    return values["entailment"], values["neutral"], values["contradiction"]


def _validation_predictions(validation: list[dict[str, Any]], entailment: np.ndarray, neutral: np.ndarray, contradiction: np.ndarray, config: ThresholdTransferConfig, signature: str) -> pd.DataFrame:
    masks = np.asarray([row["evidence_mask"] for row in validation], dtype=bool)
    support = np.nanmax(entailment, axis=1); best = np.nanargmax(entailment, axis=1) + 1; score = 1.0 - support
    if not np.isfinite(score).all() or not ((score >= 0).all() and (score <= 1).all()):
        raise RuntimeError("hallucination_score inválido")
    result: dict[str, Any] = {"example_id": [str(row["example_id"]) for row in validation], "source_id": [str(row["source_id"]) for row in validation], "label": [int(row["label"]) for row in validation], "number_of_valid_chunks": masks.sum(axis=1), "max_entailment": support, "max_entailment_chunk_index": best, "hallucination_score": score, "model_id": config.model_id, "model_revision": config.model_revision, "dataset_signature": config.dataset_signature, "split_signature": config.split_signature, "run_signature": signature}
    for slot in range(4):
        result[f"evidence_mask_chunk_{slot + 1}"] = masks[:, slot]
        for name, value in (("entailment", entailment), ("neutral", neutral), ("contradiction", contradiction)):
            result[f"{name}_probability_chunk_{slot + 1}"] = value[:, slot]
    return pd.DataFrame(result)


def _validate_validation_predictions(path: Path, validation: list[dict[str, Any]], config: ThresholdTransferConfig, signature: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    fixed = {"example_id", "source_id", "label", "number_of_valid_chunks", "max_entailment", "max_entailment_chunk_index", "hallucination_score", "model_id", "model_revision", "dataset_signature", "split_signature", "run_signature"}
    slots = {f"{kind}_probability_chunk_{slot}" for kind in ("entailment", "neutral", "contradiction") for slot in range(1, 5)} | {f"evidence_mask_chunk_{slot}" for slot in range(1, 5)}
    if not (fixed | slots).issubset(frame.columns) or len(frame) != len(validation) or not frame["example_id"].astype(str).is_unique:
        raise ValueError("ragtruth_validation_predictions.parquet incompleto")
    expected_ids = [str(row["example_id"]) for row in validation]
    if frame["example_id"].astype(str).tolist() != expected_ids or not np.array_equal(frame["label"].to_numpy(int), np.asarray([row["label"] for row in validation], dtype=int)) or not np.array_equal(frame["source_id"].astype(str).to_numpy(), np.asarray([str(row["source_id"]) for row in validation])):
        raise ValueError("Predições validation não correspondem ao split congelado")
    if set(frame["run_signature"].astype(str)) != {signature} or set(frame["model_id"].astype(str)) != {config.model_id} or set(frame["model_revision"].astype(str)) != {config.model_revision} or set(frame["dataset_signature"].astype(str)) != {config.dataset_signature} or set(frame["split_signature"].astype(str)) != {config.split_signature}:
        raise ValueError("Proveniência de prediction validation divergente")
    masks = np.asarray([row["evidence_mask"] for row in validation], dtype=bool)
    entailment = frame[[f"entailment_probability_chunk_{slot}" for slot in range(1, 5)]].to_numpy(float)
    neutral = frame[[f"neutral_probability_chunk_{slot}" for slot in range(1, 5)]].to_numpy(float)
    contradiction = frame[[f"contradiction_probability_chunk_{slot}" for slot in range(1, 5)]].to_numpy(float)
    if not all(np.array_equal(frame[f"evidence_mask_chunk_{slot}"].to_numpy(bool), masks[:, slot - 1]) for slot in range(1, 5)) or not np.isfinite(np.concatenate([entailment[masks], neutral[masks], contradiction[masks]])).all() or not (np.isnan(entailment[~masks]).all() and np.isnan(neutral[~masks]).all() and np.isnan(contradiction[~masks]).all()):
        raise ValueError("Máscara/probabilidades validation inválidas")
    if not np.allclose((entailment + neutral + contradiction)[masks], 1.0, atol=1e-6) or not np.allclose(frame["max_entailment"].to_numpy(float), np.nanmax(entailment, axis=1), atol=1e-7) or not np.array_equal(frame["max_entailment_chunk_index"].to_numpy(int), np.nanargmax(entailment, axis=1) + 1) or not np.allclose(frame["hallucination_score"].to_numpy(float), 1.0 - frame["max_entailment"].to_numpy(float), atol=1e-7):
        raise ValueError("Agregação/score validation inválidos")
    return frame


def _select_baseline_thresholds(predictions: pd.DataFrame, config: ThresholdTransferConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    labels, scores = predictions["label"].to_numpy(bool), predictions["hallucination_score"].to_numpy(float)
    thresholds: dict[str, Any] = {}; metrics: dict[str, Any] = {}
    for criterion in ("f1", "fpr10"):
        threshold, development, feasible = select_threshold(labels, scores, criterion, max_fpr=config.target_fpr)
        regime = _threshold_regime(criterion)
        thresholds[regime] = {
            "threshold": float(threshold),
            "selection_dataset": "RAGTruth validation",
            "selection_dataset_signature": config.dataset_signature,
            "selection_split_signature": config.split_signature,
            "selection_criterion": criterion,
            "selection_function": "ragtruth_transfer.metrics.select_threshold",
            "selection_implementation_version": "select-threshold-sorted-confusion-matrix-v1",
            "threshold_candidates": "unique(clipped scores) plus 0, 1, nextafter(1, 2)",
            "decision_operator": OPERATOR,
            "target_fpr": config.target_fpr if criterion == "fpr10" else None,
            "achieved_source_fpr": float(development["FPR"]) if criterion == "fpr10" else None,
            "constraint_feasible": bool(feasible),
            "source_metrics": development,
            "development_metrics": development,
        }
        metrics[regime] = _operating_metrics(labels, scores, threshold, "RAGTruth validation")
    return thresholds, metrics


def _operating_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, threshold_source: str) -> dict[str, Any]:
    result = binary_metrics(labels, np.asarray(scores) >= threshold)
    result.update({"threshold": float(threshold), "threshold_source": threshold_source, "decision_operator": OPERATOR, "secondary_metric_due_to_class_imbalance": True})
    return result


def _publichearing_metrics(target: pd.DataFrame, baseline_thresholds: dict[str, Any], lora_thresholds: dict[int, dict[str, float]]) -> dict[str, Any]:
    labels = target["label"].to_numpy(bool)
    result: dict[str, Any] = {"dataset": "PublicHearingBR", "examples": len(target), "hearings": int(target["hearing_id"].nunique()), "positives": int(labels.sum()), "negatives": int((~labels).sum()), "prevalence": float(labels.mean()), "off_the_shelf": {}, "lora": {}, "lora_mean_std": {}}
    for regime in REGIMES:
        result["off_the_shelf"][regime] = _operating_metrics(labels, target["baseline_score"].to_numpy(float), float(baseline_thresholds[regime]["threshold"]), "RAGTruth validation")
        values: list[dict[str, Any]] = []
        for seed, per_seed in lora_thresholds.items():
            item = _operating_metrics(labels, target[f"seed_{seed}_score"].to_numpy(float), per_seed[regime], "RAGTruth validation")
            result["lora"].setdefault(str(seed), {})[regime] = item; values.append(item)
        result["lora_mean_std"][regime] = {metric: {"mean": float(np.mean([float(value[metric]) for value in values])), "std": float(np.std([float(value[metric]) for value in values], ddof=1))} for metric in ("Precision", "Recall", "F1", "FPR", "Specificity", "MCC", "BalancedAccuracy", "Accuracy")}
        result["lora_mean_std"][regime]["threshold"] = "not averaged"
    return result


def _comparison(source_baseline: dict[str, Any], source_lora_audit: dict[str, Any], target_metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"protocol": "descriptive_point_comparison_without_bootstrap", "ensemble_used": False, "best_seed_selected": False, "threshold_selection_uses_publichearing_labels": False, "operating_point_transfer": {"off_the_shelf": {}, "lora": {}}, "lora_minus_baseline": {}}
    source_keys = {"FPR": "fpr", "Recall": "recall", "F1": "f1", "Precision": "precision"}
    for regime in REGIMES:
        baseline_source = source_baseline[regime]
        baseline_target = target_metrics["off_the_shelf"][regime]
        result["operating_point_transfer"]["off_the_shelf"][regime] = {name: {"ragtruth_validation": float(baseline_source[key]), "publichearing": float(baseline_target[key]), "delta": float(baseline_target[key] - baseline_source[key])} for key, name in source_keys.items()}
        for seed, audit in source_lora_audit["seeds"].items():
            source = audit["regimes"][regime]["development_metrics"]
            target = target_metrics["lora"][seed][regime]
            result["operating_point_transfer"]["lora"].setdefault(seed, {})[regime] = {name: {"ragtruth_validation": float(source[key]), "publichearing": float(target[key]), "delta": float(target[key] - source[key])} for key, name in source_keys.items()}
            result["lora_minus_baseline"].setdefault(seed, {})[regime] = {"delta_f1": float(target["F1"] - baseline_target["F1"]), "delta_recall": float(target["Recall"] - baseline_target["Recall"]), "delta_precision": float(target["Precision"] - baseline_target["Precision"]), "fpr_reduction": float(baseline_target["FPR"] - target["FPR"]), "delta_mcc": float(target["MCC"] - baseline_target["MCC"]), "delta_balanced_accuracy": float(target["BalancedAccuracy"] - baseline_target["BalancedAccuracy"])}
    return result


def _report(signature: str, thresholds: dict[str, Any], target: dict[str, Any], seconds: float) -> str:
    lines = ["# RAGTruth → PublicHearingBR threshold transfer", "", f"- Signature: `{signature}`", "- Off-the-shelf NLI scorer sem treinamento: `hallucination_score = 1 - max P(entailment)` apenas sobre evidências válidas.", "- Threshold selection ocorreu somente no RAGTruth validation; PublicHearingBR foi somente avaliação.", "- Sem LoRA/PEFT/adapter no baseline; sem ensemble ou seleção de melhor seed.", "", "## Thresholds off-the-shelf", ""]
    for regime in REGIMES:
        lines.append(f"- {regime}: {thresholds[regime]['threshold']:.9f} ({thresholds[regime]['selection_function']}; operador `{OPERATOR}`).")
    lines += ["", "## Métricas transferidas", ""]
    for regime in REGIMES:
        lines += [f"### {regime}", "", "| Método | Threshold | Precision | Recall | F1 | FPR | Specificity | MCC | Balanced accuracy | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = [("Off-the-shelf", target["off_the_shelf"][regime])]
        rows.extend((f"LoRA seed {seed}", target["lora"][str(seed)][regime]) for seed in EXPECTED_SEEDS)
        for name, metric in rows:
            lines.append(f"| {name} | {metric['threshold']:.9f} | {metric['Precision']:.6f} | {metric['Recall']:.6f} | {metric['F1']:.6f} | {metric['FPR']:.6f} | {metric['Specificity']:.6f} | {metric['MCC']:.6f} | {metric['BalancedAccuracy']:.6f} | {metric['Accuracy']:.6f} |")
        summary = target["lora_mean_std"][regime]
        lines.append(f"| LoRA mean ± std | not averaged | {summary['Precision']['mean']:.6f} ± {summary['Precision']['std']:.6f} | {summary['Recall']['mean']:.6f} ± {summary['Recall']['std']:.6f} | {summary['F1']['mean']:.6f} ± {summary['F1']['std']:.6f} | {summary['FPR']['mean']:.6f} ± {summary['FPR']['std']:.6f} | {summary['Specificity']['mean']:.6f} ± {summary['Specificity']['std']:.6f} | {summary['MCC']['mean']:.6f} ± {summary['MCC']['std']:.6f} | {summary['BalancedAccuracy']['mean']:.6f} ± {summary['BalancedAccuracy']['std']:.6f} | {summary['Accuracy']['mean']:.6f} ± {summary['Accuracy']['std']:.6f} |")
        lines.append("")
    lines += ["", f"Tempo de execução: {seconds:.3f} s.", "Precision varia com a prevalência e a mudança de domínio. Este experimento não é calibração probabilística, não demonstra causalidade e não generaliza além destes domínios e runs."]
    return "\n".join(lines) + "\n"


def _verify_completed(output_dir: Path, signature: str) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("status") != "completed":
        raise ValueError("Manifesto de threshold transfer incompatível")
    for name, digest in manifest.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Artefato ausente/corrompido: {path}")
    return manifest


def _rebuild_from_predictions(output_dir: Path, config: ThresholdTransferConfig, signature: str, payload: dict[str, Any], validation: list[dict[str, Any]], lora_audit: dict[str, Any], lora_thresholds: dict[int, dict[str, float]], target: pd.DataFrame, validation_audit: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    prediction_path = output_dir / "ragtruth_validation_predictions.parquet"
    source_predictions = _validate_validation_predictions(prediction_path, validation, config, signature)
    thresholds, source_metrics = _select_baseline_thresholds(source_predictions, config)
    target_metrics = _publichearing_metrics(target, thresholds, lora_thresholds)
    comparison = _comparison(source_metrics, lora_audit, target_metrics)
    _atomic_json(output_dir / "thresholds.json", thresholds); _atomic_json(output_dir / "ragtruth_validation_metrics.json", source_metrics); _atomic_json(output_dir / "publichearing_metrics.json", target_metrics); _atomic_json(output_dir / "lora_threshold_audit.json", lora_audit); _atomic_json(output_dir / "comparison.json", comparison); _atomic_json(output_dir / "resolved_config.json", config.to_dict())
    integrity = {"resumed_from_complete_predictions": True, "inference_reexecuted": False, "model_loaded": False, "cuda_initialized": False, "inputs": input_hashes, "validation": validation_audit, "publichearing_labels_used_only_for_frozen_threshold_application": True, "ensemble_used": False, "best_seed_selected": False}
    _atomic_json(output_dir / "integrity_audit.json", integrity); _atomic_text(output_dir / "run_log.jsonl", json.dumps({"event": "derived_artifacts_resumed", "inference_executed": False}, ensure_ascii=False) + "\n")
    _atomic_text(output_dir / "report.md", _report(signature, thresholds, target_metrics, 0.0))
    manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": payload, "inputs": input_hashes, "artifacts": {}}
    manifest["artifacts"] = _file_hashes(output_dir); _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def run_threshold_transfer(config: ThresholdTransferConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    validation, validation_audit = _load_validation(config)
    lora_audit, lora_thresholds, lora_hashes = _lora_threshold_audit(config, validation)
    target, pairing_audit, target_hashes = _load_publichearing_predictions(config)
    signature, payload = _signature(config, validation_audit, lora_hashes, target_hashes)
    output_dir = (config.output_root / signature).resolve()
    input_hashes = {**lora_hashes, **target_hashes, "ragtruth_dataset.parquet": validation_audit["metadata"]["dataset_sha256"]}
    if validate_only:
        return {"status": "valid", "inference_executed": False, "training_executed": False, "model_loaded": False, "cuda_initialized": False, "ragtruth_validation_examples": len(validation), "ragtruth_validation_sources": validation_audit["validation_sources"], "ragtruth_validation_positives": validation_audit["validation_positives"], "publichearing_examples": len(target), "publichearing_hearings": int(target["hearing_id"].nunique()), "publichearing_positives": int(target["label"].sum()), "lora_seeds": list(config.seeds), "threshold_regimes": list(REGIMES), "target_fpr": config.target_fpr, "ready_for_cluster_inference": True, "analysis_signature": signature, "output_dir": str(output_dir), "pairing": pairing_audit, "lora_threshold_audit": lora_audit}
    if output_dir.exists():
        if resume and (output_dir / "manifest.json").is_file():
            return _verify_completed(output_dir, signature)
        if resume and (output_dir / "ragtruth_validation_predictions.parquet").is_file():
            return _rebuild_from_predictions(output_dir, config, signature, payload, validation, lora_audit, lora_thresholds, target, validation_audit, input_hashes)
        raise FileExistsError(f"Output já existe; use --resume: {output_dir}")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id, revision=config.tokenizer_revision, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_id, revision=config.model_revision)
    if hasattr(model, "peft_config") or any("lora" in name.lower() or "adapter" in name.lower() for name, _ in model.named_modules()):
        raise RuntimeError("O baseline off-the-shelf não pode carregar LoRA/adapter/PEFT")
    mapping = _normalize_labels(model)
    if mapping != dict(config.expected_label_mapping) or int(getattr(model.config, "num_labels", 0)) != 3:
        raise ValueError("Mapeamento/classificação NLI incompatível")
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit is not None and str(resolved_commit) != config.model_revision:
        raise ValueError("Revisão efetivamente carregada diverge da revisão congelada")
    if not hasattr(model, "classifier") and not hasattr(model, "score"):
        raise ValueError("Cabeça de classificação NLI não identificável")
    model.eval(); [parameter.requires_grad_(False) for parameter in model.parameters()]; model.to(device)
    before = _state_hash(model)
    entailment, neutral, contradiction = _infer_validation(model, tokenizer, validation, config, mapping, device)
    after = _state_hash(model)
    if before != after:
        raise RuntimeError("Pesos do baseline mudaram durante inferência")
    predictions = _validation_predictions(validation, entailment, neutral, contradiction, config, signature)
    thresholds, source_metrics = _select_baseline_thresholds(predictions, config)
    target_metrics = _publichearing_metrics(target, thresholds, lora_thresholds)
    comparison = _comparison(source_metrics, lora_audit, target_metrics)
    input_hashes_after = {**lora_hashes, **target_hashes, "ragtruth_dataset.parquet": sha256_file(Path(validation_audit["metadata"]["path"]))}
    if input_hashes_after != input_hashes:
        raise RuntimeError("Input congelado alterado durante a análise")
    output_dir.parent.mkdir(parents=True, exist_ok=True); stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"; stage.mkdir(parents=True, exist_ok=False)
    try:
        predictions.to_parquet(stage / "ragtruth_validation_predictions.parquet", index=False)
        write_json(stage / "thresholds.json", thresholds); write_json(stage / "ragtruth_validation_metrics.json", source_metrics); write_json(stage / "publichearing_metrics.json", target_metrics); write_json(stage / "lora_threshold_audit.json", lora_audit); write_json(stage / "comparison.json", comparison); write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {"model_loaded": True, "tokenizer_loaded": True, "cuda_initialized": device.type == "cuda", "model_eval": not model.training, "all_parameters_requires_grad_false": not any(parameter.requires_grad for parameter in model.parameters()), "weights_hash_before": before, "weights_hash_after": after, "inputs": input_hashes, "inputs_reloaded_hashes": input_hashes_after, "validation": validation_audit, "publichearing_pairing": pairing_audit, "publichearing_labels_used_only_for_frozen_threshold_application": True, "optimizer_created": False, "backward_executed": False, "training_executed": False, "ensemble_used": False, "best_seed_selected": False}
        write_json(stage / "integrity_audit.json", integrity); _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "seconds": time.time() - started, "device": str(device), "inference_executed": True}, ensure_ascii=False) + "\n"); _atomic_text(stage / "report.md", _report(signature, thresholds, target_metrics, time.time() - started))
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": payload, "inputs": input_hashes, "model": {"id": config.model_id, "revision": config.model_revision, "label_mapping": mapping, "loader": "AutoModelForSequenceClassification", "lora_used": False}, "artifacts": {}}
        manifest["artifacts"] = _file_hashes(stage); write_json(stage / "manifest.json", manifest); os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
