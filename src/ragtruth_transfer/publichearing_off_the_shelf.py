from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io_utils import read_jsonl, sha256_file, write_json


SCHEMA_VERSION = "publichearing-off-the-shelf-nli-v1"
CODE_VERSION = "off-the-shelf-nli-v1"
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MODEL_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"


@dataclass(frozen=True)
class OffTheShelfConfig:
    run_name: str
    output_root: Path
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_path: Path
    dataset_sha256: str
    dataset_revision: str
    expected_examples: int
    expected_positives: int
    batch_size: int
    max_length: int
    truncation: str
    padding: str
    device: str
    confirmatory_prediction_paths: tuple[Path, ...]
    expected_label_mapping: tuple[tuple[str, int], ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "OffTheShelfConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")

        def resolve(value: Any) -> Path:
            value_path = Path(str(value)).expanduser()
            if not value_path.is_absolute():
                value_path = path.parent / value_path
            return value_path.resolve()

        protocol = raw.get("protocol", {})
        model = raw.get("model", {})
        dataset = raw.get("dataset", {})
        evaluation = raw.get("evaluation", {})
        output = raw.get("output", {})
        reference = raw.get("confirmatory_reference", {})
        declared_mapping = model.get("label_mapping", {"entailment": 0, "neutral": 1, "contradiction": 2})
        if protocol.get("evaluation_type") != "deterministic_zero_shot_nli_baseline":
            raise ValueError("protocol.evaluation_type deve declarar o baseline NLI determinístico")
        if model.get("aggregation") != "max_entailment" or model.get("positive_score") != "one_minus_max_entailment":
            raise ValueError("A configuração deve declarar max_entailment e one_minus_max_entailment")
        if bool(evaluation.get("use_threshold", True)) or bool(evaluation.get("compare_old_hardcoded_values", True)):
            raise ValueError("O baseline não pode usar threshold nem números hardcoded históricos")
        paths = reference.get("prediction_paths", [])
        if not isinstance(paths, list) or not paths:
            raise ValueError("confirmatory_reference.prediction_paths deve conter ao menos uma previsão")
        config = cls(
            run_name=str(protocol.get("name", "publichearing_off_the_shelf_max_entailment")),
            output_root=resolve(output.get("root", "../runs/publichearing_off_the_shelf_max_entailment")),
            model_id=str(model.get("model_id")),
            model_revision=str(model.get("revision")),
            tokenizer_id=str(model.get("tokenizer_id", model.get("model_id"))),
            tokenizer_revision=str(model.get("tokenizer_revision", model.get("revision"))),
            dataset_path=resolve(dataset.get("path")),
            dataset_sha256=str(dataset.get("sha256")),
            dataset_revision=str(dataset.get("revision")),
            expected_examples=int(dataset.get("expected_examples", 4235)),
            expected_positives=int(dataset.get("expected_positives", 501)),
            batch_size=int(model.get("batch_size", 16)),
            max_length=int(model.get("max_length", 512)),
            truncation=str(model.get("truncation", "only_first")),
            padding=str(model.get("padding", "longest")),
            device=str(model.get("device", "auto")),
            confirmatory_prediction_paths=tuple(resolve(value) for value in paths),
            expected_label_mapping=tuple(sorted((str(key), int(value)) for key, value in declared_mapping.items())),
        )
        config.validate_declarations()
        return config

    def validate_declarations(self) -> None:
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("Modelo/revisão não correspondem ao checkpoint-base congelado")
        if self.tokenizer_id != self.model_id or self.tokenizer_revision != self.model_revision:
            raise ValueError("Tokenizer deve usar o mesmo modelo e revisão congelados")
        if self.max_length != 512 or self.truncation != "only_first" or self.padding not in {"longest", "max_length"}:
            raise ValueError("Contrato de tokenização esperado: max_length=512, truncation=only_first")
        if self.expected_examples != 4235 or self.expected_positives != 501:
            raise ValueError("O baseline deve avaliar exatamente 4235 exemplos e 501 positivos")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        if dict(self.expected_label_mapping) != {"entailment": 0, "neutral": 1, "contradiction": 2}:
            raise ValueError("label_mapping declarado deve refletir o mapeamento NLI congelado do checkpoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {"name": self.run_name, "evaluation_type": "deterministic_zero_shot_nli_baseline"},
            "model": {
                "model_id": self.model_id, "revision": self.model_revision,
                "tokenizer_id": self.tokenizer_id, "tokenizer_revision": self.tokenizer_revision,
                "max_length": self.max_length, "premise_field": "evidence", "hypothesis_field": "claim",
                "truncation": self.truncation, "padding": self.padding,
                "aggregation": "max_entailment", "positive_score": "one_minus_max_entailment",
                "label_mapping": dict(self.expected_label_mapping), "batch_size": self.batch_size, "device": self.device,
            },
            "dataset": {
                "path": str(self.dataset_path), "sha256": self.dataset_sha256, "revision": self.dataset_revision,
                "expected_examples": self.expected_examples, "expected_positives": self.expected_positives,
                "chunk_order": ["chunk_1", "chunk_2", "chunk_3", "chunk_4"],
            },
            "evaluation": {"use_threshold": False, "calculate_threshold_free_metrics": True, "compare_old_hardcoded_values": False},
            "output": {"root": str(self.output_root)},
            "confirmatory_reference": {"prediction_paths": [str(path) for path in self.confirmatory_prediction_paths]},
        }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    write_json(temporary, value)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _iter_modelable(path: Path) -> Iterable[dict[str, Any]]:
    for record in read_jsonl(path):
        hearing_id = str(record["id"])
        metadata = record["metadados_extraidos"]
        for person_index, person in enumerate(metadata.get("envolvidos", [])):
            for opinion_index, opinion_entry in enumerate(person.get("opinioes", [])):
                chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                claim = str(opinion_entry.get("opiniao", ""))
                if len(chunks) != 4 or not hearing_id or not claim.strip() or any(not chunk.strip() for chunk in chunks):
                    continue
                yield {
                    "example_id": f"{hearing_id}:{person_index}:{opinion_index}",
                    "hearing_id": hearing_id,
                    "claim": claim,
                    "evidence": chunks,
                    "label": int(bool(opinion_entry.get("verificacao_alucinacao", {}).get("verificacao_manual"))),
                }


def _example_set_hash(rows: list[dict[str, Any]]) -> str:
    return _canonical_hash([
        {"example_id": row["example_id"], "hearing_id": row["hearing_id"], "label": int(row["label"]),
         "claim": row["claim"], "evidence": list(row["evidence"])} for row in rows
    ])


def _load_rows(config: OffTheShelfConfig) -> tuple[list[dict[str, Any]], str]:
    if not config.dataset_path.is_file():
        raise FileNotFoundError(f"PublicHearingBR não encontrado: {config.dataset_path}")
    actual_sha = sha256_file(config.dataset_path)
    if actual_sha != config.dataset_sha256:
        raise ValueError(f"SHA-256 do PublicHearingBR diverge: {actual_sha} != {config.dataset_sha256}")
    rows = list(_iter_modelable(config.dataset_path))
    if len(rows) != config.expected_examples:
        raise ValueError(f"Quantidade de exemplos inesperada: {len(rows)}")
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("example_id duplicado")
    if sum(int(row["label"]) for row in rows) != config.expected_positives:
        raise ValueError("Quantidade de positivos inesperada")
    if len({row["hearing_id"] for row in rows}) != 206:
        raise ValueError("Quantidade de audiências inesperada")
    return rows, _example_set_hash(rows)


def _reference_paths(config: OffTheShelfConfig) -> list[Path]:
    paths: list[Path] = []
    for path in config.confirmatory_prediction_paths:
        if path.is_dir():
            paths.extend(sorted(path.glob("seed_*/publichearing_zero_shot/*/predictions.parquet")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Previsão confirmatória não encontrada: {path}")
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    if not unique:
        raise FileNotFoundError("Nenhuma previsão confirmatória encontrada")
    return unique


def _check_references(config: OffTheShelfConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = pd.DataFrame({"example_id": [r["example_id"] for r in rows], "hearing_id": [r["hearing_id"] for r in rows], "label": [int(r["label"]) for r in rows]})
    expected = expected.set_index("example_id")
    results: list[dict[str, Any]] = []
    for path in _reference_paths(config):
        frame = pd.read_parquet(path)
        required = {"example_id", "hearing_id", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Previsão confirmatória sem colunas obrigatórias: {path}")
        actual = frame[["example_id", "hearing_id", "label"]].copy()
        actual["example_id"] = actual["example_id"].astype(str)
        actual = actual.set_index("example_id")
        only_baseline = sorted(set(expected.index) - set(actual.index))
        only_reference = sorted(set(actual.index) - set(expected.index))
        common = expected.index.intersection(actual.index)
        label_diff = int((expected.loc[common, "label"].astype(int) != actual.loc[common, "label"].astype(int)).sum())
        hearing_diff = int((expected.loc[common, "hearing_id"].astype(str) != actual.loc[common, "hearing_id"].astype(str)).sum())
        scores_finite = True
        if "probability" in frame.columns:
            scores_finite = bool(np.isfinite(frame["probability"].astype(float).to_numpy()).all())
        result = {
            "path": str(path), "rows": int(len(frame)), "ids_unique": bool(frame["example_id"].is_unique),
            "ids_only_baseline": len(only_baseline), "ids_only_reference": len(only_reference),
            "label_divergences": label_diff, "hearing_id_divergences": hearing_diff,
            "pairable_examples": int(len(common)), "score_orientation": "hallucination_score higher means hallucination", "reference_scores_finite": scores_finite,
            "readiness": bool(bool(frame["example_id"].is_unique) and scores_finite and len(only_baseline) == 0 and len(only_reference) == 0 and label_diff == 0 and hearing_diff == 0 and len(common) == len(expected)),
        }
        results.append(result)
        if not result["readiness"]:
            raise ValueError(f"Correspondência confirmatória inválida: {path}: {result}")
    return {"references": results, "all_ready": all(item["readiness"] for item in results)}


def _normalize_labels(model: Any) -> dict[str, int]:
    id2label = getattr(model.config, "id2label", None) or {}
    label2id = getattr(model.config, "label2id", None) or {}
    normalized: dict[str, int] = {}
    canonical_id2label = {int(index): str(label) for index, label in id2label.items()}
    for index, label in canonical_id2label.items():
        index = int(index)
        text = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if text.startswith("label ") or text.startswith("class ") or not text:
            continue
        for name in ("entailment", "neutral", "contradiction"):
            if text == name:
                if name in normalized:
                    raise ValueError(f"Mapeamento {name} ambíguo")
                normalized[name] = index
    if set(normalized) != {"entailment", "neutral", "contradiction"}:
        raise ValueError(f"id2label não fornece mapeamento NLI inequívoco: {id2label}")
    for name, index in normalized.items():
        label = canonical_id2label[index]
        reverse = label2id.get(label, label2id.get(label.upper()))
        if reverse is None:
            raise ValueError("id2label e label2id inconsistentes: label ausente")
        if int(reverse) != index:
            raise ValueError("id2label e label2id inconsistentes")
    return normalized


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _signature(
    config: OffTheShelfConfig,
    example_hash: str,
    label_mapping: dict[str, int] | None = None,
    *,
    premise: str = "chunk",
    hypothesis: str = "claim",
    aggregation: str = "max_entailment",
    positive_score: str = "1 - max_entailment",
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": SCHEMA_VERSION, "code_version": CODE_VERSION,
        "model_id": config.model_id, "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer_id, "tokenizer_revision": config.tokenizer_revision,
        "dataset_sha256": config.dataset_sha256, "dataset_revision": config.dataset_revision,
        "example_set_hash": example_hash, "chunk_order": [1, 2, 3, 4],
        "premise": premise, "hypothesis": hypothesis, "max_length": config.max_length,
        "truncation": config.truncation, "padding": config.padding,
        "label_mapping": label_mapping or {"entailment": 0, "neutral": 1, "contradiction": 2},
        "softmax": "torch.softmax(logits, dim=-1)", "aggregation": aggregation,
        "positive_score": positive_score, "threshold": False,
    }
    return _canonical_hash(payload)[:16], payload


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(values, q / 100.0)) for q in (5, 25, 50, 75, 95)}


def _metrics(rows: list[dict[str, Any]], scores: np.ndarray, supports: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    hallucination = np.asarray(scores, dtype=float)
    result: dict[str, Any] = {
        "protocol": "deterministic_zero_shot_nli_baseline", "N": int(len(labels)),
        "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "prevalence": float(labels.mean()),
        "AUPRC": float(average_precision_score(labels, hallucination)), "AUROC": float(roc_auc_score(labels, hallucination)),
        "Brier": float(brier_score_loss(labels, hallucination)),
        "support_score": {"mean": float(supports.mean()), "std": float(supports.std()), "min": float(supports.min()), "max": float(supports.max())},
        "hallucination_score": {"mean": float(hallucination.mean()), "std": float(hallucination.std()), "min": float(hallucination.min()), "max": float(hallucination.max())},
        "descriptive_by_label": {},
        "sanity_checks": {
            "support_hallucination_correlation": float(np.corrcoef(supports, hallucination)[0, 1]),
            "correlation_expected_minus_one": bool(np.isclose(np.corrcoef(supports, hallucination)[0, 1], -1.0, atol=1e-12)),
            "hallucination_equals_one_minus_support": bool(np.array_equal(hallucination, 1.0 - supports)),
        },
        "thresholds": None,
    }
    for label in (0, 1):
        values = hallucination[labels == label]
        result["descriptive_by_label"][str(label)] = {"N": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "quantiles": _quantiles(values)}
    return result


def _report(config: OffTheShelfConfig, manifest: dict[str, Any], metrics: dict[str, Any], pairability: dict[str, Any]) -> str:
    lines = [
        "# PublicHearingBR off-the-shelf NLI baseline", "",
        "Execução determinística, somente por inferência, usando a cabeça NLI original do mDeBERTa congelado.", "",
        f"- Run signature: `{manifest['signature']}`", f"- Modelo: `{config.model_id}` @ `{config.model_revision}`",
        f"- Dataset: `{config.dataset_revision}` / SHA-256 `{config.dataset_sha256}`", f"- Exemplos: {metrics['N']} (positivos={metrics['positives']}, negativos={metrics['negatives']})",
        "- Pares: premise=`chunk_j`, hypothesis=`claim`; truncation=`only_first`, max_length=512; padding por lote.",
        "- Agregação pré-especificada: `support_score=max(P(entailment))`; `hallucination_score=1-support_score`.",
        "- Não houve threshold, calibração, treinamento, LoRA, adapter, optimizer, scheduler ou bootstrap.", "",
        "## Métricas threshold-free", "",
        f"- AUPRC: {metrics['AUPRC']:.6f}", f"- AUROC: {metrics['AUROC']:.6f}", f"- Brier: {metrics['Brier']:.6f}",
        f"- support mean/std: {metrics['support_score']['mean']:.6f} / {metrics['support_score']['std']:.6f}",
        f"- hallucination mean/std: {metrics['hallucination_score']['mean']:.6f} / {metrics['hallucination_score']['std']:.6f}", "",
        "## Pareabilidade com a campanha confirmatória", "",
    ]
    for reference in pairability["references"]:
        lines.append(f"- `{reference['path']}`: {reference['pairable_examples']} pareáveis; IDs faltantes/extras={reference['ids_only_baseline']}/{reference['ids_only_reference']}; labels divergentes={reference['label_divergences']}; hearing_id divergentes={reference['hearing_id_divergences']}; readiness=`{reference['readiness']}`.")
    lines += ["", "A fórmula `1 - max(P(entailment))` é um score heurístico de ausência de suporte, não necessariamente uma probabilidade calibrada de alucinação."]
    return "\n".join(lines) + "\n"


def _resume_from_complete_predictions(
    config: OffTheShelfConfig,
    output_dir: Path,
    rows: list[dict[str, Any]],
    example_hash: str,
    signature: str,
    signature_payload: dict[str, Any],
    pairability: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = output_dir / "predictions.parquet"
    frame = pd.read_parquet(prediction_path)
    required = {"example_id", "hearing_id", "label", "max_entailment", "max_entailment_chunk_index", "hallucination_score", "run_signature", "model_id", "model_revision"}
    if not required.issubset(frame.columns):
        raise ValueError("predictions.parquet incompleto para resume derivado")
    expected_ids = [str(row["example_id"]) for row in rows]
    if len(frame) != len(rows) or not frame["example_id"].astype(str).is_unique or frame["example_id"].astype(str).tolist() != expected_ids:
        raise ValueError("predictions.parquet não corresponde à ordem/quantidade esperada")
    if set(frame["run_signature"].astype(str)) != {signature}:
        raise ValueError("Assinatura divergente em predictions.parquet")
    if frame["model_id"].astype(str).nunique() != 1 or frame["model_id"].iloc[0] != config.model_id:
        raise ValueError("Modelo divergente em predictions.parquet")
    if frame["model_revision"].astype(str).nunique() != 1 or frame["model_revision"].iloc[0] != config.model_revision:
        raise ValueError("Revisão divergente em predictions.parquet")
    expected_labels = np.asarray([int(row["label"]) for row in rows])
    if not np.array_equal(frame["label"].to_numpy(dtype=int), expected_labels):
        raise ValueError("Labels divergentes em predictions.parquet")
    expected_hearings = np.asarray([str(row["hearing_id"]) for row in rows])
    if not np.array_equal(frame["hearing_id"].astype(str).to_numpy(), expected_hearings):
        raise ValueError("hearing_id divergente em predictions.parquet")
    probability_columns = {name: [f"{name}_chunk_{index}" for index in range(1, 5)] for name in ("entailment", "neutral", "contradiction")}
    if not all(set(columns).issubset(frame.columns) for columns in probability_columns.values()):
        raise ValueError("Probabilidades entailment ausentes para validar resume")
    entailment_columns = probability_columns["entailment"]
    entailment = frame[entailment_columns].to_numpy(dtype=float)
    neutral = frame[probability_columns["neutral"]].to_numpy(dtype=float)
    contradiction = frame[probability_columns["contradiction"]].to_numpy(dtype=float)
    support = frame["max_entailment"].to_numpy(dtype=float)
    hallucination = frame["hallucination_score"].to_numpy(dtype=float)
    chunk_index = frame["max_entailment_chunk_index"].to_numpy(dtype=int)
    if not np.isfinite(np.concatenate([entailment.ravel(), neutral.ravel(), contradiction.ravel(), support, hallucination])).all():
        raise ValueError("NaN/inf em predictions.parquet")
    if not np.allclose(entailment + neutral + contradiction, 1.0, atol=1e-6):
        raise ValueError("Probabilidades NLI não somam 1 em predictions.parquet")
    if not np.allclose(support, entailment.max(axis=1), atol=1e-7) or not np.array_equal(chunk_index, entailment.argmax(axis=1) + 1):
        raise ValueError("Agregação max-entailment inconsistente em predictions.parquet")
    if not np.allclose(hallucination, 1.0 - support, atol=1e-7) or not ((hallucination >= 0).all() and (hallucination <= 1).all()):
        raise ValueError("hallucination_score inconsistente em predictions.parquet")
    metrics = _metrics(rows, hallucination, support)
    integrity = {
        "resumed_from_complete_predictions": True, "inference_reexecuted": False,
        "model_loaded": False, "weights_hash_available": False, "predictions_rows": len(frame),
        "ids_unique": True, "example_set_hash": example_hash, "pairability": pairability,
        "nli_probabilities_finite": True, "final_score_finite": True,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(output_dir / "resolved_config.json", config.to_dict())
    _atomic_json(output_dir / "integrity_audit.json", integrity)
    _atomic_text(output_dir / "run_log.jsonl", json.dumps({"event": "derived_artifacts_resumed", "inference_executed": False}, ensure_ascii=False) + "\n")
    provisional = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "model": {"id": config.model_id, "revision": config.model_revision}, "dataset": {"path": str(config.dataset_path), "sha256": config.dataset_sha256, "revision": config.dataset_revision, "examples": len(rows), "hearings": len({row['hearing_id'] for row in rows}), "positives": int(sum(row['label'] for row in rows))}}
    _atomic_text(output_dir / "report.md", _report(config, provisional, metrics, pairability))
    provisional["artifacts"] = _file_hashes(output_dir)
    _atomic_json(output_dir / "manifest.json", provisional)
    return provisional


def run_baseline(config: OffTheShelfConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    rows, example_hash = _load_rows(config)
    pairability = _check_references(config, rows)
    signature, signature_payload = _signature(config, example_hash)
    output_dir = (config.output_root / signature).resolve()
    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if resume and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") != signature or manifest.get("status") != "completed":
                raise ValueError("Manifesto existente tem assinatura/status incompatível")
            for name, expected in manifest.get("artifacts", {}).items():
                path = output_dir / name
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError(f"Artefato corrompido ou ausente: {path}")
            return manifest
        if resume and (output_dir / "predictions.parquet").is_file() and not manifest_path.exists():
            return _resume_from_complete_predictions(config, output_dir, rows, example_hash, signature, signature_payload, pairability)
        raise FileExistsError(f"Output já existe; use --resume para reutilizar: {output_dir}")
    if validate_only:
        return {
            "status": "valid", "model_loaded": False, "cuda_initialized": False, "inference_executed": False, "training_executed": False,
            "n_examples": len(rows), "n_hearings": len({row["hearing_id"] for row in rows}), "positives": sum(int(row["label"]) for row in rows), "negatives": sum(int(not row["label"]) for row in rows),
            "model_id": config.model_id, "model_revision": config.model_revision, "aggregation": "max_entailment", "positive_score": "one_minus_max_entailment",
            "lora_used": False, "adapter_loaded": False, "trained_checkpoint_loaded": False,
            "run_signature": signature, "output_dir": str(output_dir), "example_set_hash": example_hash, "pairability": pairability,
        }

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id, revision=config.tokenizer_revision, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_id, revision=config.model_revision)
    if hasattr(model, "peft_config") or any("lora" in name.lower() or "adapter" in name.lower() for name, _ in model.named_modules()):
        raise RuntimeError("Modelo off-the-shelf não pode conter LoRA/adapter/PEFT")
    mapping = _normalize_labels(model)
    if mapping != dict(config.expected_label_mapping):
        raise ValueError(f"Mapeamento NLI resolvido diverge do declarado: {mapping}")
    if int(getattr(model.config, "num_labels", 0)) != 3:
        raise ValueError("Modelo NLI deve ter exatamente três classes")
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit is not None and str(resolved_commit) != config.model_revision:
        raise ValueError(f"Revisão carregada diverge da revisão congelada: {resolved_commit}")
    if not hasattr(model, "classifier") and not hasattr(model, "score"):
        raise ValueError("Cabeça de classificação NLI não identificável")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    before = _state_hash(model)
    probabilities: dict[str, list[np.ndarray]] = {"entailment": [], "neutral": [], "contradiction": []}
    with torch.inference_mode():
        for chunk_index in range(4):
            for start in range(0, len(rows), config.batch_size):
                batch = rows[start:start + config.batch_size]
                premises = [row["evidence"][chunk_index] for row in batch]
                claims = [row["claim"] for row in batch]
                padding = True if config.padding == "longest" else "max_length"
                encoded = tokenizer(premises, claims, truncation=config.truncation, max_length=config.max_length, padding=padding, return_tensors="pt")
                tensors = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
                logits = model(**tensors).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
                    raise RuntimeError("Probabilidades NLI inválidas")
                for name in probabilities:
                    probabilities[name].append(probs[:, mapping[name]])
    after = _state_hash(model)
    if before != after:
        raise RuntimeError("Pesos do modelo mudaram durante inferência")
    entailment = np.concatenate(probabilities["entailment"]).reshape(4, len(rows)).T
    neutral = np.concatenate(probabilities["neutral"]).reshape(4, len(rows)).T
    contradiction = np.concatenate(probabilities["contradiction"]).reshape(4, len(rows)).T
    support = entailment.max(axis=1)
    best_index = entailment.argmax(axis=1) + 1
    hallucination = 1.0 - support
    if not np.isfinite(hallucination).all() or not ((hallucination >= 0).all() & (hallucination <= 1).all()):
        raise RuntimeError("hallucination_score inválido")
    signature, signature_payload = _signature(config, example_hash, mapping)
    output_dir = (config.output_root / signature).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output já existe para a assinatura final: {output_dir}")
    metrics = _metrics(rows, hallucination, support)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        predictions = pd.DataFrame({
            "example_id": [row["example_id"] for row in rows], "hearing_id": [row["hearing_id"] for row in rows], "label": [int(row["label"]) for row in rows],
            **{f"entailment_chunk_{i}": entailment[:, i - 1] for i in range(1, 5)}, **{f"neutral_chunk_{i}": neutral[:, i - 1] for i in range(1, 5)}, **{f"contradiction_chunk_{i}": contradiction[:, i - 1] for i in range(1, 5)},
            "max_entailment": support, "max_entailment_chunk_index": best_index, "support_score": support, "hallucination_score": hallucination,
            "model_id": config.model_id, "model_revision": config.model_revision, "dataset_signature": config.dataset_sha256, "dataset_revision": config.dataset_revision, "run_signature": signature,
        })
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        pairability = _check_references(config, rows)
        integrity = {"model_eval": not model.training, "inference_mode": True, "optimizer_created": False, "scheduler_created": False, "backward_executed": False, "all_parameters_requires_grad_false": not any(p.requires_grad for p in model.parameters()), "weights_hash_before": before, "weights_hash_after": after, "model_class": model.__class__.__name__, "model_architectures": list(getattr(model.config, "architectures", []) or []), "hidden_size": int(getattr(model.config, "hidden_size", 0)), "resolved_model_revision": resolved_commit or config.model_revision, "tokenizer_class": tokenizer.__class__.__name__, "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", config.tokenizer_id)), "tokenizer_has_token_type_ids": "token_type_ids" in tokenizer(["x"], ["y"], truncation=config.truncation, max_length=config.max_length, padding=True), "num_labels": int(model.config.num_labels), "id2label": {str(k): str(v) for k, v in model.config.id2label.items()}, "label2id": {str(k): int(v) for k, v in model.config.label2id.items()}, "label_mapping": mapping, "num_parameters": int(sum(p.numel() for p in model.parameters())), "predictions_rows": len(predictions), "ids_unique": bool(predictions["example_id"].is_unique), "nli_probabilities_finite": bool(np.isfinite(np.concatenate([entailment, neutral, contradiction])).all()), "nli_probability_sums_one": bool(np.allclose(entailment + neutral + contradiction, 1.0, atol=1e-6)), "final_score_finite": bool(np.isfinite(hallucination).all()), "final_score_in_unit_interval": bool(((hallucination >= 0) & (hallucination <= 1)).all()), "example_set_hash": example_hash, "pairability": pairability}
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "resolved_config.json", config.to_dict())
        write_json(stage / "integrity_audit.json", integrity)
        (stage / "run_log.jsonl").write_text(json.dumps({"event": "completed", "seconds": time.time() - started, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "model": {"id": config.model_id, "revision": config.model_revision, "label_mapping": mapping}, "dataset": {"path": str(config.dataset_path), "sha256": config.dataset_sha256, "revision": config.dataset_revision, "examples": len(rows), "hearings": len({row['hearing_id'] for row in rows}), "positives": int(sum(row['label'] for row in rows))}, "artifacts": _file_hashes(stage)}
        (stage / "report.md").write_text(_report(config, manifest, metrics, pairability), encoding="utf-8")
        manifest["artifacts"] = _file_hashes(stage)
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
