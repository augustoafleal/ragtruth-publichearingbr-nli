
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .dataset import BagCollator
from .io_utils import read_jsonl, sha256_file, write_json
from .metrics import binary_metrics, select_threshold
from .modeling import build_model

ZERO_SHOT_SCHEMA = "ragtruth-to-publichearing-zero-shot-preliminary-v1"
CODE_VERSION = "ragtruth-to-publichearing-zero-shot-v1"


@dataclass(frozen=True)
class ZeroShotConfig:
    ragtruth_run_dir: Path
    checkpoint_relative_path: str = "checkpoints/epoch_03"
    validation_predictions_relative_path: str = "checkpoints/epoch_03/validation_predictions.csv"
    expected_run_signature: str | None = None
    expected_source_run_name: str = "ragtruth_lora_attention_mil_parquet"
    expected_ragtruth_dataset_signature: str = "0cdf598fa866741d"
    expected_ragtruth_schema: str = "ragtruth-qa-training-view-deduplicated-v1"
    expected_ragtruth_split_signature: str = "525edec2966a4fac"
    expected_best_epoch: int = 3
    expected_model_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    publichearing_path: Path = Path("data/PublicHearingBR_NLI.jsonl")
    publichearing_dataset_sha256: str | None = None
    publichearing_dataset_revision: str | None = "2f84a44bc34df483e25c987f0ff86caad0ab3433"
    expected_publichearing_examples: int = 4235
    expected_publichearing_positives: int | None = 501
    batch_size: int = 4
    dtype: str = "float32"
    device: str = "auto"
    output_root: Path = Path("results/ragtruth_to_publichearing_zero_shot/preliminary")
    run_name: str = "ragtruth_en_to_publichearingbr_zero_shot_preliminary"
    expected_chunks: int = 4
    indomain_reference_dir: Path | None = None
    seed: int = 42

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], base_dir: Path) -> "ZeroShotConfig":
        def resolve(value: Any, default: Path | None = None) -> Path | None:
            if value is None:
                return default
            path = Path(str(value)).expanduser()
            return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(
            ragtruth_run_dir=resolve(raw.get("ragtruth_run_dir"), base_dir / "../runs/ragtruth_lora_attention_mil_parquet"),  # type: ignore[arg-type]
            checkpoint_relative_path=str(raw.get("checkpoint_relative_path", cls.checkpoint_relative_path)),
            validation_predictions_relative_path=str(raw.get("validation_predictions_relative_path", cls.validation_predictions_relative_path)),
            expected_run_signature=(str(raw["expected_run_signature"]) if raw.get("expected_run_signature") else None),
            expected_source_run_name=str(raw.get("expected_source_run_name", cls.expected_source_run_name)),
            expected_ragtruth_dataset_signature=str(raw.get("expected_ragtruth_dataset_signature", cls.expected_ragtruth_dataset_signature)),
            expected_ragtruth_schema=str(raw.get("expected_ragtruth_schema", cls.expected_ragtruth_schema)),
            expected_ragtruth_split_signature=str(raw.get("expected_ragtruth_split_signature", cls.expected_ragtruth_split_signature)),
            expected_best_epoch=int(raw.get("expected_best_epoch", cls.expected_best_epoch)),
            expected_model_revision=str(raw.get("expected_model_revision", cls.expected_model_revision)),
            publichearing_path=resolve(raw.get("publichearing_path"), base_dir / "../data/PublicHearingBR_NLI.jsonl"),  # type: ignore[arg-type]
            publichearing_dataset_sha256=(str(raw["publichearing_dataset_sha256"]) if raw.get("publichearing_dataset_sha256") else None),
            publichearing_dataset_revision=(str(raw["publichearing_dataset_revision"]) if raw.get("publichearing_dataset_revision") else None),
            expected_publichearing_examples=int(raw.get("expected_publichearing_examples", cls.expected_publichearing_examples)),
            expected_publichearing_positives=(int(raw["expected_publichearing_positives"]) if raw.get("expected_publichearing_positives") is not None else None),
            batch_size=int(raw.get("batch_size", cls.batch_size)),
            dtype=str(raw.get("dtype", cls.dtype)),
            device=str(raw.get("device", cls.device)),
            output_root=resolve(raw.get("output_root"), base_dir / "../results/ragtruth_to_publichearing_zero_shot/preliminary"),  # type: ignore[arg-type]
            run_name=str(raw.get("run_name", cls.run_name)),
            expected_chunks=int(raw.get("expected_chunks", cls.expected_chunks)),
            indomain_reference_dir=resolve(raw.get("indomain_reference_dir")),  # type: ignore[arg-type]
            seed=int(raw.get("seed", cls.seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        for key in ("ragtruth_run_dir", "publichearing_path", "output_root", "indomain_reference_dir"):
            value[key] = str(value[key]) if value[key] is not None else None
        return value


def load_zero_shot_config(path: Path) -> ZeroShotConfig:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração zero-shot inválida: {path}")
    return ZeroShotConfig.from_mapping(raw, path.parent.resolve())


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _validate_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in dict(manifest.get("artifacts", {})).items():
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != str(expected):
            raise ValueError(f"Artefato de checkpoint ausente/corrompido: {relative}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON inválido: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Esperava objeto JSON: {path}")
    return value


def validate_frozen_source(config: ZeroShotConfig) -> tuple[Path, dict[str, Any], dict[str, Any], ExperimentConfig]:
    run_dir = config.ragtruth_run_dir.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto do run RAGTruth não encontrado: {run_manifest_path}")
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError("O run RAGTruth precisa estar completed.")
    run_config_raw = run_manifest.get("config")
    if not isinstance(run_config_raw, dict):
        raise ValueError("Manifesto não contém config resolvida.")
    run_name = str(run_config_raw.get("run_name", ""))
    if run_name != config.expected_source_run_name or "smoke" in run_name.lower():
        raise ValueError(f"Run de origem não é o screening full esperado: {run_name!r}")
    source_signature = str(run_manifest.get("signature") or run_manifest.get("config_fingerprint") or "")
    if config.expected_run_signature and source_signature != config.expected_run_signature:
        raise ValueError(f"Assinatura do run incompatível: {source_signature!r}")
    if int(run_manifest.get("best_epoch", -1)) != config.expected_best_epoch:
        raise ValueError("best_epoch do run não corresponde ao checkpoint esperado.")
    dataset_metadata = run_manifest.get("dataset", {})
    if dataset_metadata.get("signature") != config.expected_ragtruth_dataset_signature:
        raise ValueError("Assinatura da visão RAGTruth incompatível.")
    if dataset_metadata.get("schema_version") != config.expected_ragtruth_schema:
        raise ValueError("Schema da visão RAGTruth incompatível.")
    split_hashes = run_manifest.get("data_split_sha256", {})
    checkpoint_split_hashes = {}
    if isinstance(split_hashes, dict):
        checkpoint_split_hashes = split_hashes
    if checkpoint_split_hashes.get("split") != config.expected_ragtruth_split_signature:
        raise ValueError("Assinatura da divisão RAGTruth incompatível.")
    parent_manifest_path = dataset_metadata.get("manifest_path")
    if parent_manifest_path:
        parent_manifest_file = Path(str(parent_manifest_path))
        if not parent_manifest_file.is_file():
            raise ValueError(f"Manifesto parent RAGTruth não encontrado: {parent_manifest_file}")
        expected_parent_hash = checkpoint_split_hashes.get("manifest")
        if expected_parent_hash and sha256_file(parent_manifest_file) != str(expected_parent_hash):
            raise ValueError("SHA do manifesto parent RAGTruth não coincide com o run.")
    checkpoint = (run_dir / config.checkpoint_relative_path).resolve()
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    marker_path = checkpoint / "CHECKPOINT_COMPLETE"
    if not checkpoint_manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Checkpoint incompleto: {checkpoint}")
    checkpoint_manifest = _load_json(checkpoint_manifest_path)
    if marker_path.read_text(encoding="ascii").strip() != sha256_file(checkpoint_manifest_path):
        raise ValueError("CHECKPOINT_COMPLETE não corresponde ao manifesto.")
    _validate_artifacts(checkpoint, checkpoint_manifest)
    if int(checkpoint_manifest.get("epoch", -1)) != config.expected_best_epoch:
        raise ValueError("Época do checkpoint incompatível.")
    best_checkpoint = str(run_manifest.get("best_checkpoint", ""))
    if best_checkpoint and Path(best_checkpoint).as_posix() != Path(config.checkpoint_relative_path).as_posix():
        raise ValueError("Checkpoint solicitado não é o checkpoint selecionado no manifesto do run.")
    if str(checkpoint_manifest.get("model_revision")) != config.expected_model_revision:
        raise ValueError("Revisão do mDeBERTa incompatível.")
    checkpoint_data_hashes = checkpoint_manifest.get("data_split_sha256", {})
    if isinstance(checkpoint_data_hashes, dict) and checkpoint_data_hashes.get("split") != config.expected_ragtruth_split_signature:
        raise ValueError("Checkpoint não pertence à divisão RAGTruth esperada.")
    if str(run_config_raw.get("model_revision")) != config.expected_model_revision:
        raise ValueError("Revisão do modelo no run não está fixada conforme configuração.")
    if run_config_raw.get("architecture") not in {"gated_attention", "mean", "max", "attention", "set_transformer"} or run_config_raw.get("encoder_mode") != "lora":
        raise ValueError("Checkpoint não é um modelo LoRA com pooling suportado.")
    model_config = ExperimentConfig.from_mapping(run_config_raw)
    if model_config.max_length != int(run_config_raw.get("max_length", 512)):
        raise ValueError("Configuração de tokenização inconsistente.")
    adapter_config_path = checkpoint / "adapter" / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError("Checkpoint LoRA sem adapter/adapter_config.json.")
    adapter_config = _load_json(adapter_config_path)
    expected_lora = run_config_raw.get("lora", {})
    checks = {
        "r": int(expected_lora.get("r", 8)),
        "lora_alpha": int(expected_lora.get("alpha", 16)),
        "lora_dropout": float(expected_lora.get("dropout", 0.1)),
    }
    for key, expected in checks.items():
        actual = adapter_config.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Parâmetro LoRA incompatível: {key}")
        elif actual != expected:
            raise ValueError(f"Parâmetro LoRA incompatível: {key}")
    expected_targets = set(str(value) for value in expected_lora.get("target_modules", []))
    actual_targets = set(str(value) for value in adapter_config.get("target_modules", []))
    if expected_targets != actual_targets:
        raise ValueError(f"targets LoRA incompatíveis: {actual_targets} != {expected_targets}")
    required_head = checkpoint / "head.pt"
    if not required_head.is_file():
        raise ValueError("Checkpoint sem head.pt.")
    return checkpoint, run_manifest, checkpoint_manifest, model_config


def _thresholds_from_validation(config: ZeroShotConfig, checkpoint: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    validation_path = (config.ragtruth_run_dir / config.validation_predictions_relative_path).resolve()
    if not validation_path.is_file():
        validation_path = checkpoint / "validation_predictions.csv"
    if not validation_path.is_file():
        raise FileNotFoundError(f"Previsões de validation não encontradas: {validation_path}")
    frame = pd.read_csv(validation_path)
    if "label" not in frame or "score" not in frame:
        raise ValueError("Validation precisa conter label e score.")
    labels = frame["label"].astype(bool).to_numpy()
    scores = frame["score"].astype(float).to_numpy()
    if len(labels) == 0 or not np.isfinite(scores).all() or np.unique(labels).size < 2:
        raise ValueError("Validation inválida para thresholds.")
    thresholds: dict[str, Any] = {}
    saved = run_manifest.get("thresholds")
    for criterion in ("f1", "fpr10"):
        value = saved.get(criterion) if isinstance(saved, dict) else None
        if not isinstance(value, dict) or "threshold" not in value:
            threshold, development, feasible = select_threshold(labels, scores, criterion)
            source = "validation_predictions_recomputed"
        else:
            threshold = float(value["threshold"])
            development = value.get("development_metrics") or binary_metrics(labels, scores >= threshold)
            feasible = bool(value.get("constraint_feasible", True))
            source = "run_manifest_validation_thresholds"
        thresholds[criterion] = {
            "threshold": float(threshold),
            "development_metrics": development,
            "constraint_feasible": feasible,
            "source": source,
            "validation_examples": int(len(labels)),
            "validation_sha256": sha256_file(validation_path),
            "validation_path": str(validation_path),
        }
    return thresholds


def _iter_publichearing_inputs(path: Path) -> Iterable[dict[str, Any]]:
    for record in read_jsonl(path):
        hearing_id = str(record["id"])
        metadata = record["metadados_extraidos"]
        for person_index, person in enumerate(metadata.get("envolvidos", [])):
            for opinion_index, opinion_entry in enumerate(person.get("opinioes", [])):
                chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                if len(chunks) != 4:
                    continue
                # Keep the exact modelability rules used by the in-domain
                # PublicHearingBR normalizer, without reading its label.
                if not hearing_id or not str(opinion_entry.get("opiniao", "")).strip() or any(not chunk.strip() for chunk in chunks):
                    continue
                yield {
                    "example_id": f"{hearing_id}:{person_index}:{opinion_index}",
                    "hearing_id": hearing_id,
                    "source_id": hearing_id,
                    "claim": str(opinion_entry.get("opiniao", "")),
                    "evidence": chunks,
                    "evidence_mask": [True, True, True, True],
                    "label": 0,
                    "task_type": "PublicHearingBR",
                }


def load_publichearing_inputs(config: ZeroShotConfig) -> list[dict[str, Any]]:
    path = config.publichearing_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PublicHearingBR não encontrado: {path}")
    if config.publichearing_dataset_sha256 and sha256_file(path) != config.publichearing_dataset_sha256:
        raise ValueError("SHA-256 do PublicHearingBR não coincide com a configuração.")
    rows = list(_iter_publichearing_inputs(path))
    if len(rows) != config.expected_publichearing_examples:
        raise ValueError(f"Quantidade PublicHearingBR inesperada: {len(rows)}")
    ids = [str(row["example_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("example_id duplicado no PublicHearingBR.")
    for row in rows:
        if not str(row["claim"]).strip() or len(row["evidence"]) != config.expected_chunks:
            raise ValueError(f"Exemplo PublicHearingBR inválido: {row['example_id']}")
        if not all(str(chunk).strip() for chunk in row["evidence"]):
            raise ValueError(f"Evidência vazia no PublicHearingBR: {row['example_id']}")
    return rows


def _load_publichearing_labels(path: Path, expected_ids: list[str], expected_positives: int | None) -> np.ndarray:
    labels: dict[str, int] = {}
    for record in read_jsonl(path):
        hearing_id = str(record["id"])
        metadata = record["metadados_extraidos"]
        for person_index, person in enumerate(metadata.get("envolvidos", [])):
            for opinion_index, opinion_entry in enumerate(person.get("opinioes", [])):
                chunks = opinion_entry.get("chunks_proximos") or []
                if len(chunks) != 4:
                    continue
                if not hearing_id or not str(opinion_entry.get("opiniao", "")).strip() or any(not str(chunk).strip() for chunk in chunks):
                    continue
                key = f"{hearing_id}:{person_index}:{opinion_index}"
                labels[key] = int(bool(opinion_entry.get("verificacao_alucinacao", {}).get("verificacao_manual")))
    if set(labels) != set(expected_ids):
        raise ValueError("IDs de labels PublicHearingBR não coincidem com inputs.")
    result = np.asarray([labels[key] for key in expected_ids], dtype=bool)
    if expected_positives is not None and int(result.sum()) != expected_positives:
        raise ValueError(f"Quantidade de positivos inesperada: {int(result.sum())}")
    return result


@torch.inference_mode()
def _infer(model: torch.nn.Module, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int, max_length: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=BagCollator(tokenizer, max_length), pin_memory=device.type == "cuda")
    probabilities: list[np.ndarray] = []
    pooling_weights: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        batch.pop("labels")
        batch.pop("example_ids")
        batch.pop("source_ids")
        batch.pop("task_types")
        tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits, weights = model(**tensors)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        pooling_weights.append(weights.float().cpu().numpy())
    scores = np.concatenate(probabilities)
    weights = np.concatenate(pooling_weights)
    if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
        raise ValueError("Probabilidades zero-shot não finitas ou fora de [0,1].")
    if weights.shape[1] != 4 or not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Pesos de pooling PublicHearingBR inválidos.")
    return scores, weights


def _metric_payload(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold
    payload = binary_metrics(labels, prediction, scores)
    payload["Brier"] = float(brier_score_loss(labels, scores))
    payload["threshold"] = float(threshold)
    return payload


def _comparison(config: ZeroShotConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "interpretation": "descriptive_only",
        "in_domain_confirmatory": None,
        "baseline_max_entailment": {"AUPRC": 0.343, "AUROC": 0.769, "F1": 0.371, "FPR": 0.076, "MCC": 0.291, "source": "descriptive_reference_from_protocol"},
        "formal_superiority_test": False,
    }
    if config.indomain_reference_dir:
        metrics_path = config.indomain_reference_dir / "outputs" / "overall_oof_metrics.csv"
        if metrics_path.is_file():
            frame = pd.read_csv(metrics_path)
            result["in_domain_confirmatory"] = {str(row["criterion"]): {key: (None if pd.isna(row[key]) else float(row[key])) for key in ("AUPRC", "AUROC", "Brier", "F1", "FPR", "MCC") if key in row} for _, row in frame.iterrows()}
            result["in_domain_source"] = str(metrics_path)
    return result


def run_zero_shot(config: ZeroShotConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    checkpoint, run_manifest, checkpoint_manifest, model_config = validate_frozen_source(config)
    thresholds = _thresholds_from_validation(config, checkpoint, run_manifest)
    rows = load_publichearing_inputs(config)
    checkpoint_hash = _json_hash(_file_hashes(checkpoint))
    dataset_hash = sha256_file(config.publichearing_path)
    signature_payload = {
        "schema": ZERO_SHOT_SCHEMA, "code_version": CODE_VERSION, "source_run_signature": str(run_manifest.get("signature") or run_manifest.get("config_fingerprint")),
        "checkpoint_hash": checkpoint_hash, "checkpoint_epoch": checkpoint_manifest["epoch"], "model": model_config.to_dict(),
        "pooling_type": model_config.canonical_pooling_type, "thresholds": thresholds, "publichearing_sha256": dataset_hash, "preprocessing": "existing_publichearing_chunks_proximos_exactly_four", "batch_size": config.batch_size, "dtype": config.dtype,
    }
    signature = _json_hash(signature_payload)[:16]
    output_dir = (config.output_root / signature).resolve()
    if output_dir.exists():
        if resume and (output_dir / "manifest.json").is_file():
            existing = _load_json(output_dir / "manifest.json")
            if existing.get("signature") == signature:
                _validate_artifacts(output_dir, existing)
                return existing
        raise FileExistsError(f"Output já existe ou é parcial: {output_dir}")
    if validate_only:
        return {"status": "valid", "signature": signature, "pooling_type": model_config.canonical_pooling_type, "model_loaded": False, "cuda_initialized": False, "inference_executed": False, "backward_executed": False, "checkpoint": str(checkpoint), "checkpoint_epoch": checkpoint_manifest["epoch"], "thresholds": thresholds, "publichearing_examples": len(rows), "publichearing_sha256": dataset_hash}
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    if config.dtype not in {"float32", "float16"}:
        raise ValueError("dtype deve ser float32 ou float16")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, revision=model_config.model_revision, use_fast=True)
    model, model_metadata = build_model(model_config, adapter_path=checkpoint / "adapter", adapter_trainable=False)
    state = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_encoder = [key for key in missing if not key.startswith("encoder.")]
    if missing_non_encoder or unexpected:
        raise ValueError(f"Head incompatível: missing={missing_non_encoder}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    before = _json_hash({key: value.detach().cpu().numpy().tobytes().hex() for key, value in model.state_dict().items()})
    scores, pooling_weights = _infer(model, tokenizer, rows, config.batch_size, model_config.max_length, device)
    after = _json_hash({key: value.detach().cpu().numpy().tobytes().hex() for key, value in model.state_dict().items()})
    if before != after:
        raise RuntimeError("Pesos do modelo mudaram durante inferência.")
    # Labels are deliberately loaded only after probabilities are complete.
    ids = [str(row["example_id"]) for row in rows]
    labels = _load_publichearing_labels(config.publichearing_path, ids, config.expected_publichearing_positives)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        predictions = pd.DataFrame({"example_id": ids, "hearing_id": [str(row["hearing_id"]) for row in rows], "label": labels.astype(int), "probability": scores, "seed": int(config.seed), "prediction_ragtruth_best_f1": scores >= thresholds["f1"]["threshold"], "prediction_ragtruth_fpr10": scores >= thresholds["fpr10"]["threshold"], "ragtruth_best_f1_threshold": thresholds["f1"]["threshold"], "ragtruth_fpr10_threshold": thresholds["fpr10"]["threshold"], "model_run_signature": str(run_manifest.get("signature") or run_manifest.get("config_fingerprint")), "checkpoint_hash": checkpoint_hash, "publichearing_dataset_signature": config.publichearing_dataset_sha256 or dataset_hash})
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        metrics = {"protocol": "preliminary exploratory zero-shot diagnostic", "N": int(len(labels)), "prevalence": float(labels.mean()), "average_predicted_probability": float(scores.mean()), "probability_by_label": {"0": {"N": int((~labels).sum()), "mean": float(scores[~labels].mean())}, "1": {"N": int(labels.sum()), "mean": float(scores[labels].mean())}}, "threshold_free": {"AUPRC": float(average_precision_score(labels, scores)), "AUROC": float(roc_auc_score(labels, scores)), "Brier": float(brier_score_loss(labels, scores))}, "ragtruth_validation_best_f1_threshold": _metric_payload(labels, scores, thresholds["f1"]["threshold"]), "ragtruth_validation_fpr10_threshold": _metric_payload(labels, scores, thresholds["fpr10"]["threshold"])}
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "thresholds.json", thresholds)
        write_json(stage / "comparison.json", _comparison(config))
        write_json(stage / "resolved_config.json", config.to_dict())
        write_json(stage / "integrity_audit.json", {"labels_loaded_after_inference": True, "optimizer_created": False, "backward_executed": False, "model_eval": not model.training, "all_parameters_requires_grad_false": not any(parameter.requires_grad for parameter in model.parameters()), "weights_hash_before": before, "weights_hash_after": after, "four_slots_all_valid": bool(np.all(np.asarray([row["evidence_mask"] for row in rows], dtype=bool))), "unique_example_ids": len(set(ids)) == len(ids), "pooling_type": model_config.canonical_pooling_type, "pooling_weights_shape": list(pooling_weights.shape), "pooling_weights_sum_to_one": bool(np.allclose(pooling_weights.sum(axis=1), 1.0, atol=1e-5))})
        run_log = {"event": "completed", "seconds": time.time() - started, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
        (stage / "run_log.jsonl").write_text(json.dumps(run_log, ensure_ascii=False) + "\n", encoding="utf-8")
        artifact_hashes = _file_hashes(stage)
        manifest = {"schema_version": ZERO_SHOT_SCHEMA, "status": "completed", "diagnostic_classification": "preliminary exploratory zero-shot diagnostic", "signature": signature, "pooling_type": model_config.canonical_pooling_type, "signature_payload": signature_payload, "source_run": {"run_dir": str(config.ragtruth_run_dir.resolve()), "run_manifest_sha256": sha256_file(config.ragtruth_run_dir / "run_manifest.json"), "signature": str(run_manifest.get("signature") or run_manifest.get("config_fingerprint")), "best_epoch": checkpoint_manifest["epoch"], "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json")}, "publichearing": {"path": str(config.publichearing_path.resolve()), "sha256": dataset_hash, "dataset_revision": config.publichearing_dataset_revision, "examples": len(labels), "positives": int(labels.sum())}, "threshold_origin": "RAGTruth validation only", "model": model_metadata, "artifacts": artifact_hashes}
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
