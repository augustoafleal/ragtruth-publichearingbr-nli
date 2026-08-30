
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
import yaml

from .io_utils import sha256_file, write_json
from .metrics import binary_metrics


SCHEMA_VERSION = "publichearing-thresholded-paired-grouped-bootstrap-v1"
CODE_VERSION = "thresholded-paired-grouped-bootstrap-v1"
EXPECTED_ROWS, EXPECTED_HEARINGS, EXPECTED_POSITIVES = 4235, 206, 501
EXPECTED_SEEDS = (0, 1, 2)
REGIMES = ("best_f1", "fpr10")
METRICS = ("f1", "recall", "precision", "mcc", "balanced_accuracy", "fpr", "specificity", "accuracy")
METRIC_KEYS = {
    "f1": "F1", "recall": "Recall", "precision": "Precision", "mcc": "MCC",
    "balanced_accuracy": "BalancedAccuracy", "fpr": "FPR", "specificity": "Specificity", "accuracy": "Accuracy",
}
OPERATOR = ">="


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


@dataclass(frozen=True)
class ThresholdedBootstrapConfig:
    output_root: Path
    threshold_transfer_run: Path
    threshold_transfer_signature: str
    baseline_run: Path
    baseline_signature: str
    baseline_score_column: str
    confirmatory_run: Path
    confirmatory_signature: str
    seeds: tuple[int, ...]
    lora_score_column: str
    regimes: tuple[str, ...]
    n_replicates: int
    seed: int
    confidence_level: float
    ci_method: str
    preserve_group_multiplicity: bool
    same_samples_across_methods: bool
    same_samples_across_seeds: bool
    same_samples_across_threshold_regimes: bool
    reestimate_thresholds: bool
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "ThresholdedBootstrapConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        inputs, bootstrap = raw.get("inputs", {}), raw.get("bootstrap", {})
        transfer, baseline, confirmatory = inputs.get("threshold_transfer_run", {}), inputs.get("baseline_run", {}), inputs.get("confirmatory_run", {})
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_thresholded_paired_bootstrap")),
            threshold_transfer_run=resolve(transfer.get("path")), threshold_transfer_signature=str(transfer.get("expected_signature")),
            baseline_run=resolve(baseline.get("path")), baseline_signature=str(baseline.get("expected_signature")), baseline_score_column=str(baseline.get("score_column")),
            confirmatory_run=resolve(confirmatory.get("path")), confirmatory_signature=str(confirmatory.get("expected_signature")),
            seeds=tuple(int(item) for item in confirmatory.get("seeds", [])), lora_score_column=str(confirmatory.get("score_column")),
            regimes=tuple(str(item) for item in raw.get("threshold_regimes", [])), n_replicates=int(bootstrap.get("n_replicates", 0)), seed=int(bootstrap.get("seed", -1)),
            confidence_level=float(bootstrap.get("confidence_level", 0)), ci_method=str(bootstrap.get("ci_method")),
            preserve_group_multiplicity=bool(bootstrap.get("preserve_group_multiplicity")), same_samples_across_methods=bool(bootstrap.get("same_samples_across_methods")),
            same_samples_across_seeds=bool(bootstrap.get("same_samples_across_seeds")), same_samples_across_threshold_regimes=bool(bootstrap.get("same_samples_across_threshold_regimes")),
            reestimate_thresholds=bool(bootstrap.get("reestimate_thresholds")), primary_metrics=tuple(str(item) for item in raw.get("primary_metrics", [])),
            secondary_metrics=tuple(str(item) for item in raw.get("secondary_metrics", [])),
        )
        config.validate(raw.get("protocol", {}))
        return config

    def validate(self, protocol: dict[str, Any]) -> None:
        if protocol.get("analysis_type") != "paired_grouped_bootstrap" or protocol.get("group_key") != "hearing_id" or protocol.get("paired") is not True:
            raise ValueError("O protocolo deve ser bootstrap pareado por hearing_id")
        if (self.threshold_transfer_signature, self.baseline_signature, self.confirmatory_signature) != ("3ceffc4a74b484fe", "54d9c623f8685c39", "4e12933c51136624"):
            raise ValueError("Assinaturas dos runs congelados incompatíveis")
        if self.seeds != EXPECTED_SEEDS or self.regimes != REGIMES or self.baseline_score_column != "hallucination_score" or self.lora_score_column != "probability":
            raise ValueError("Seeds, regimes ou colunas de score incompatíveis")
        if self.n_replicates < 1 or self.seed < 0 or not 0 < self.confidence_level < 1 or self.ci_method != "percentile":
            raise ValueError("Configuração de bootstrap inválida")
        if self.reestimate_thresholds or not all((self.preserve_group_multiplicity, self.same_samples_across_methods, self.same_samples_across_seeds, self.same_samples_across_threshold_regimes)):
            raise ValueError("O protocolo exige thresholds fixos e a mesma amostra agrupada para todas as comparações")
        if self.primary_metrics != ("f1", "recall", "mcc", "balanced_accuracy") or self.secondary_metrics != ("precision", "fpr", "specificity", "accuracy"):
            raise ValueError("Conjunto de métricas incompatível com o protocolo")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {"schema": SCHEMA_VERSION, "name": "publichearing_thresholded_paired_bootstrap", "analysis_type": "paired_grouped_bootstrap", "group_key": "hearing_id", "paired": True},
            "inputs": {"threshold_transfer_run": {"signature": self.threshold_transfer_signature}, "baseline_run": {"signature": self.baseline_signature, "score_column": self.baseline_score_column}, "confirmatory_run": {"signature": self.confirmatory_signature, "seeds": list(self.seeds), "score_column": self.lora_score_column}},
            "threshold_regimes": list(self.regimes), "bootstrap": {"n_replicates": self.n_replicates, "seed": self.seed, "confidence_level": self.confidence_level, "ci_method": self.ci_method, "preserve_group_multiplicity": self.preserve_group_multiplicity, "same_samples_across_methods": self.same_samples_across_methods, "same_samples_across_seeds": self.same_samples_across_seeds, "same_samples_across_threshold_regimes": self.same_samples_across_threshold_regimes, "reestimate_thresholds": self.reestimate_thresholds},
            "primary_metrics": list(self.primary_metrics), "secondary_metrics": list(self.secondary_metrics), "output": {"root": str(self.output_root)},
        }


def _validate_manifest(directory: Path, signature: str) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto ausente: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if str(manifest.get("signature")) != signature or manifest.get("status") not in {"completed", "externally_evaluated"}:
        raise ValueError(f"Manifesto incompatível: {path}")
    for name, digest in manifest.get("artifacts", {}).items():
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"Artefato congelado ausente ou alterado: {artifact}")
    return manifest


def _validate_prediction(path: Path, score_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"example_id", "hearing_id", "label", score_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Colunas ausentes em {path}: {sorted(required - set(frame.columns))}")
    selected = frame[["example_id", "hearing_id", "label", score_column]].copy()
    selected["example_id"] = selected["example_id"].astype(str); selected["hearing_id"] = selected["hearing_id"].astype(str)
    if len(selected) != EXPECTED_ROWS or not selected.example_id.is_unique or selected.hearing_id.nunique() != EXPECTED_HEARINGS:
        raise ValueError(f"Contagem, IDs ou hearings inválidos: {path}")
    if selected.isna().any().any() or set(selected.label.astype(int).unique()) != {0, 1} or int(selected.label.astype(int).sum()) != EXPECTED_POSITIVES:
        raise ValueError(f"Labels ou nulos inválidos: {path}")
    scores = selected[score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
        raise ValueError(f"Scores inválidos: {path}")
    return selected


def _locate_lora_prediction(config: ThresholdedBootstrapConfig, seed: int) -> tuple[Path, pd.DataFrame]:
    matches = sorted((config.confirmatory_run / f"seed_{seed}" / "publichearing_zero_shot").glob("*/predictions.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Esperava uma previsão PublicHearingBR na seed {seed}, encontrei {len(matches)}")
    prediction = matches[0]
    _validate_manifest(prediction.parent, prediction.parent.name)
    return prediction, _validate_prediction(prediction, config.lora_score_column)


def _strict_join(baseline: pd.DataFrame, lora_frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = baseline.rename(columns={"hallucination_score": "baseline_score"}).set_index("example_id").sort_index()
    audit: dict[str, Any] = {}
    for seed, frame in lora_frames.items():
        current = frame.rename(columns={"probability": "score"}).set_index("example_id")
        base_ids, lora_ids = set(base.index), set(current.index)
        common = sorted(base_ids & lora_ids)
        status = {"baseline_only_ids": len(base_ids - lora_ids), "lora_only_ids": len(lora_ids - base_ids), "duplicate_ids": int(not base.index.is_unique or not current.index.is_unique), "label_mismatches": 0, "hearing_id_mismatches": 0, "pairable_examples": len(common)}
        if not status["baseline_only_ids"] and not status["lora_only_ids"]:
            status["label_mismatches"] = int((base.loc[common, "label"].astype(int) != current.loc[common, "label"].astype(int)).sum())
            status["hearing_id_mismatches"] = int((base.loc[common, "hearing_id"].astype(str) != current.loc[common, "hearing_id"].astype(str)).sum())
        audit[str(seed)] = status
        if any(status[key] for key in ("baseline_only_ids", "lora_only_ids", "duplicate_ids", "label_mismatches", "hearing_id_mismatches")) or len(common) != EXPECTED_ROWS:
            raise ValueError(f"Pareamento estrito falhou para seed {seed}: {status}")
        base[f"seed_{seed}"] = current.loc[base.index, "score"].to_numpy(dtype=float)
    return base.reset_index(), audit


def _load_thresholds(config: ThresholdedBootstrapConfig) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[str, Any], dict[str, str]]:
    transfer_manifest = _validate_manifest(config.threshold_transfer_run, config.threshold_transfer_signature)
    threshold_path = config.threshold_transfer_run / "thresholds.json"
    lora_audit_path = config.threshold_transfer_run / "lora_threshold_audit.json"
    frozen, lora_audit = json.loads(threshold_path.read_text(encoding="utf-8")), json.loads(lora_audit_path.read_text(encoding="utf-8"))
    baseline: dict[str, float] = {}
    lora: dict[int, dict[str, float]] = {}
    for regime in REGIMES:
        item = frozen.get(regime, {})
        if item.get("selection_dataset") != "RAGTruth validation" or item.get("decision_operator") != OPERATOR or not np.isfinite(float(item.get("threshold", np.nan))):
            raise ValueError(f"Threshold baseline inválido: {regime}")
        baseline[regime] = float(item["threshold"])
    for seed in config.seeds:
        lora[seed] = {}
        seed_item = lora_audit.get("seeds", {}).get(str(seed), {})
        for regime in REGIMES:
            item = seed_item.get("regimes", {}).get(regime, {})
            threshold = float(item.get("threshold", np.nan))
            if lora_audit.get("threshold_source") != "RAGTruth validation only" or lora_audit.get("decision_operator") != OPERATOR or not np.isfinite(threshold):
                raise ValueError(f"Threshold LoRA inválido: seed {seed}/{regime}")
            lora[seed][regime] = threshold
    known = {0: (0.6320486665, 0.4379436672), 1: (0.5359182358, 0.5201002359), 2: (0.7721872926, 0.5335189700)}
    for seed, (best, fpr10) in known.items():
        if not np.isclose(lora[seed]["best_f1"], best) or not np.isclose(lora[seed]["fpr10"], fpr10):
            raise ValueError(f"Check de threshold LoRA falhou para seed {seed}")
    audit = {"threshold_transfer_signature": config.threshold_transfer_signature, "source_artifact": "thresholds.json and lora_threshold_audit.json", "source_artifacts_sha256": {"thresholds.json": sha256_file(threshold_path), "lora_threshold_audit.json": sha256_file(lora_audit_path)}, "source_dataset": "RAGTruth validation", "source_split": "525edec2966a4fac", "decision_operator": OPERATOR, "threshold_reestimated": False, "publichearing_labels_used_for_threshold_selection": False, "baseline": baseline, "lora": {str(seed): value for seed, value in lora.items()}}
    hashes = {"thresholds.json": sha256_file(threshold_path), "lora_threshold_audit.json": sha256_file(lora_audit_path), "threshold_transfer_manifest.json": sha256_file(config.threshold_transfer_run / "manifest.json")}
    audit["source_manifest_artifacts"] = transfer_manifest.get("artifacts", {})
    return baseline, lora, audit, hashes


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return binary_metrics(labels.astype(bool), np.asarray(scores, dtype=float) >= threshold)


def _metric_values(values: dict[str, float | int]) -> dict[str, float]:
    return {metric: float(values[METRIC_KEYS[metric]]) for metric in METRICS}


def _bootstrap_metric_values(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth, pred = np.asarray(labels, dtype=bool), np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(truth & pred)); fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred)); tn = int(np.count_nonzero(~truth & ~pred))
    def divide(n: float, d: float) -> float: return float(n / d) if d else 0.0
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    specificity, fpr = divide(tn, tn + fp), divide(fp, fp + tn)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator else 0.0
    return {"f1": divide(2 * tp, 2 * tp + fp + fn), "recall": recall, "precision": precision, "mcc": mcc, "balanced_accuracy": (recall + specificity) / 2, "fpr": fpr, "specificity": specificity, "accuracy": divide(tp + tn, tp + tn + fp + fn)}


def _summary(values: np.ndarray, confidence_level: float) -> dict[str, Any]:
    if not len(values):
        return {"mean": None, "std": None, "median": None, "percentile_2_5": None, "percentile_97_5": None, "ci_lower": None, "ci_upper": None, "bootstrap_support_probability": None, "n_valid": 0}
    alpha = 1.0 - confidence_level
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "median": float(np.median(values)), "percentile_2_5": float(np.quantile(values, 0.025)), "percentile_97_5": float(np.quantile(values, 0.975)), "ci_lower": float(np.quantile(values, alpha / 2)), "ci_upper": float(np.quantile(values, 1 - alpha / 2)), "bootstrap_support_probability": float((values > 0).mean()), "n_valid": int(len(values))}


def _signature(config: ThresholdedBootstrapConfig, hashes: dict[str, str], joined: pd.DataFrame, thresholds: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {"schema": SCHEMA_VERSION, "code_version": CODE_VERSION, "input_hashes": hashes, "baseline_run_signature": config.baseline_signature, "confirmatory_run_signature": config.confirmatory_signature, "threshold_transfer_signature": config.threshold_transfer_signature, "thresholds": thresholds, "decision_operator": OPERATOR, "example_ids_hash": _canonical_hash(joined.example_id.astype(str).tolist()), "hearing_ids_hash": _canonical_hash(sorted(joined.hearing_id.astype(str).unique().tolist())), "seeds": list(config.seeds), "regimes": list(config.regimes), "metrics": list(METRICS), "effect_formulas": {metric: "lora - baseline" for metric in METRICS}, "n_replicates": config.n_replicates, "bootstrap_seed": config.seed, "confidence_level": config.confidence_level, "ci_method": config.ci_method, "group_key": "hearing_id", "preserve_group_multiplicity": True, "same_samples_across_methods": True, "same_samples_across_seeds": True, "same_samples_across_threshold_regimes": True, "reestimate_thresholds": False}
    return _canonical_hash(payload)[:16], payload


def _observed(joined: pd.DataFrame, baseline_thresholds: dict[str, float], lora_thresholds: dict[int, dict[str, float]], frozen_metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = joined.label.to_numpy(dtype=int)
    observed: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    for regime in REGIMES:
        base = _metrics(labels, joined.baseline_score.to_numpy(float), baseline_thresholds[regime])
        expected = frozen_metrics["off_the_shelf"][regime]
        for key in ("TP", "FP", "TN", "FN", "F1", "Recall", "Precision", "MCC", "BalancedAccuracy", "FPR", "Specificity", "Accuracy"):
            if not np.isclose(float(base[key]), float(expected[key]), rtol=0, atol=1e-12):
                raise ValueError(f"Métrica baseline não reproduz o artefato congelado: {regime}/{key}")
        observed[regime] = {"baseline": base, "lora": {}}
        effects[regime] = {}
        for seed in EXPECTED_SEEDS:
            current = _metrics(labels, joined[f"seed_{seed}"].to_numpy(float), lora_thresholds[seed][regime])
            expected = frozen_metrics["lora"][str(seed)][regime]
            for key in ("TP", "FP", "TN", "FN", "F1", "Recall", "Precision", "MCC", "BalancedAccuracy", "FPR", "Specificity", "Accuracy"):
                if not np.isclose(float(current[key]), float(expected[key]), rtol=0, atol=1e-12):
                    raise ValueError(f"Métrica LoRA não reproduz o artefato congelado: seed {seed}/{regime}/{key}")
            observed[regime]["lora"][str(seed)] = current
            effects[regime][str(seed)] = {metric: float(current[METRIC_KEYS[metric]]) - float(base[METRIC_KEYS[metric]]) for metric in METRICS}
    return observed, effects


def _derive_results(frame: pd.DataFrame, observed: dict[str, Any], observed_effects: dict[str, Any], config: ThresholdedBootstrapConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_seed: dict[str, Any] = {}; campaign: dict[str, Any] = {}
    validity: dict[str, Any] = {"requested_replicates": config.n_replicates, "invalid_reasons": {}}
    for regime in REGIMES:
        per_seed[regime] = {}; campaign[regime] = {"ensemble_used": False, "best_seed_selected": False, "same_bootstrap_samples_across_seeds": True, "metrics": {}}
        for seed in config.seeds:
            result = {"observed": {"baseline": _metric_values(observed[regime]["baseline"]), "lora": _metric_values(observed[regime]["lora"][str(seed)]), "effects": observed_effects[regime][str(seed)]}, "bootstrap": {}, "fpr10_operating_constraint": None}
            for metric in METRICS:
                subset = frame[(frame.regime == regime) & (frame.seed == seed) & (frame.metric == metric)]
                values = subset.loc[subset.valid, "effect"].dropna().to_numpy(float)
                result["bootstrap"][metric] = _summary(values, config.confidence_level)
            if regime == "fpr10":
                subset = frame[(frame.regime == regime) & (frame.seed == seed) & (frame.metric == "fpr") & frame.valid]
                result["fpr10_operating_constraint"] = {"baseline_proportion_replicates_fpr_le_010": float((subset.baseline_value <= .10).mean()), "lora_proportion_replicates_fpr_le_010": float((subset.lora_value <= .10).mean())}
            per_seed[regime][str(seed)] = result
        for metric in METRICS:
            subset = frame[(frame.regime == regime) & (frame.metric == metric) & frame.valid]
            values = subset.groupby("replicate_id", observed=True, sort=False).effect.mean().to_numpy(float)
            campaign[regime]["metrics"][metric] = {"observed_effect": float(np.mean([observed_effects[regime][str(seed)][metric] for seed in config.seeds])), "bootstrap": _summary(values, config.confidence_level)}
        reference = frame[(frame.regime == regime) & (frame.seed == 0) & (frame.metric == "f1")]
        invalid = reference.loc[~reference.valid, "invalid_reason"].value_counts().to_dict()
        validity[regime] = {"valid_replicates": int(reference.valid.sum()), "invalid_replicates": int((~reference.valid).sum()), "invalid_reasons": {str(key): int(value) for key, value in invalid.items()}}
    return per_seed, campaign, validity


def _report(signature: str, observed: dict[str, Any], per_seed: dict[str, Any], campaign: dict[str, Any], validity: dict[str, Any]) -> str:
    lines = ["# Bootstrap pareado agrupado por audiência — métricas thresholded", "", f"- Assinatura da análise: `{signature}`", "- Unidade de reamostragem: `hearing_id`; 206 audiências sorteadas com reposição, preservando multiplicidade.", "- Thresholds foram lidos do run de transferência e permaneceram constantes; operador: `score >= threshold`.", "- Não houve ensemble, seleção da melhor seed, carga de modelo/tokenizer, CUDA, inferência ou treinamento.", "", "## Efeito médio das três seeds", ""]
    for regime in REGIMES:
        lines += [f"### {regime}", "", "| Métrica | Efeito observado | IC bootstrap 95% | Support |", "|---|---:|---:|---:|"]
        for metric in METRICS:
            item = campaign[regime]["metrics"][metric]; summary = item["bootstrap"]
            lines.append(f"| {metric} | {item['observed_effect']:.6f} | [{summary['ci_lower']:.6f}, {summary['ci_upper']:.6f}] | {summary['bootstrap_support_probability']:.4f} |")
        lines.append("")
    lines += ["## Interpretação", "", "Efeito é definido como LoRA − baseline. Para F1, recall, MCC e balanced accuracy, ICs inteiramente positivos indicam que a vantagem permaneceu positiva sob bootstrap pareado agrupado por audiência. FPR é reportada como trade-off operacional, não como critério isolado de superioridade.", "", f"Réplicas: {validity}", ""]
    return "\n".join(lines)


def _verify_completed(output_dir: Path, signature: str) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("status") != "completed":
        raise ValueError("Manifesto de saída incompatível")
    for name, digest in manifest.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Saída incompleta ou corrompida: {path}")
    return manifest


def _rebuild_from_complete_replicates(
    output_dir: Path,
    config: ThresholdedBootstrapConfig,
    signature: str,
    signature_payload: dict[str, Any],
    observed: dict[str, Any],
    observed_effects: dict[str, Any],
    threshold_audit: dict[str, Any],
    input_hashes: dict[str, str],
    input_paths: dict[str, str],
    pairing: dict[str, Any],
) -> dict[str, Any]:
    path = output_dir / "bootstrap_replicates.parquet"
    frame = pd.read_parquet(path)
    required = {"replicate_id", "bootstrap_seed", "regime", "seed", "metric", "threshold_baseline", "threshold_lora", "baseline_value", "lora_value", "effect", "valid", "invalid_reason", "n_rows", "n_sampled_groups", "n_unique_groups", "n_positive", "n_negative"}
    expected_rows = config.n_replicates * len(REGIMES) * len(config.seeds) * len(METRICS)
    if not required.issubset(frame.columns) or len(frame) != expected_rows:
        raise ValueError("Réplicas parciais/incompatíveis; não há estado RNG auditável para retomar")
    if frame.duplicated(["replicate_id", "regime", "seed", "metric"]).any() or set(frame.replicate_id.astype(int)) != set(range(config.n_replicates)) or set(frame.bootstrap_seed.astype(int)) != {config.seed}:
        raise ValueError("Réplicas parciais ou duplicadas; não há estado RNG auditável para retomar")
    identity = frame.groupby(["regime", "seed"], observed=True)[["threshold_baseline", "threshold_lora"]].nunique(dropna=False)
    if (identity.to_numpy() != 1).any():
        raise ValueError("Threshold variou nas réplicas persistidas")
    per_seed, campaign, validity = _derive_results(frame, observed, observed_effects, config)
    _atomic_json(output_dir / "observed_metrics.json", observed); _atomic_json(output_dir / "observed_effects.json", observed_effects)
    _atomic_json(output_dir / "per_seed_results.json", per_seed); _atomic_json(output_dir / "campaign_summary.json", campaign)
    _atomic_json(output_dir / "threshold_audit.json", threshold_audit); _atomic_json(output_dir / "resolved_config.json", config.to_dict())
    common_result = {"model_loaded": False, "tokenizer_loaded": False, "cuda_initialized": False, "inference_executed": False, "training_executed": False}
    integrity = {**common_result, "inputs": input_hashes, "input_paths": input_paths, "pairing": pairing, "n_examples": EXPECTED_ROWS, "n_hearings": EXPECTED_HEARINGS, "n_bootstrap_requested": config.n_replicates, "validity": validity, "paired": True, "grouped": True, "preserve_group_multiplicity": True, "same_samples_across_methods": True, "same_samples_across_seeds": True, "same_samples_across_threshold_regimes": True, "thresholds_reestimated": False, "thresholds_identical_all_replicates": True, "ensemble_used": False, "best_seed_selected": False, "publichearing_labels_used_for_threshold_selection": False, "decision_operator": OPERATOR, "resumed_from_complete_replicates": True, "bootstrap_reexecuted": False}
    _atomic_json(output_dir / "integrity_audit.json", integrity)
    _atomic_text(output_dir / "run_log.jsonl", json.dumps({"event": "derived_artifacts_resumed", "bootstrap_executed": False}, ensure_ascii=False) + "\n")
    _atomic_text(output_dir / "report.md", _report(signature, observed, per_seed, campaign, validity))
    manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": input_hashes, "counts": {"examples": EXPECTED_ROWS, "hearings": EXPECTED_HEARINGS, "positives": EXPECTED_POSITIVES, "negatives": EXPECTED_ROWS - EXPECTED_POSITIVES, "bootstrap_requested": config.n_replicates, "bootstrap_valid": validity["best_f1"]["valid_replicates"], "bootstrap_invalid": validity["best_f1"]["invalid_replicates"]}, "artifacts": _file_hashes(output_dir)}
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def run_thresholded_bootstrap(config: ThresholdedBootstrapConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    _validate_manifest(config.baseline_run, config.baseline_signature)
    _validate_manifest(config.confirmatory_run, config.confirmatory_signature)
    baseline_path = config.baseline_run / "predictions.parquet"
    baseline = _validate_prediction(baseline_path, config.baseline_score_column)
    lora_frames: dict[int, pd.DataFrame] = {}; input_paths = {"baseline_predictions.parquet": str(baseline_path)}
    input_hashes = {"baseline_predictions.parquet": sha256_file(baseline_path)}
    for seed in config.seeds:
        path, frame = _locate_lora_prediction(config, seed); lora_frames[seed] = frame
        input_paths[f"seed_{seed}_predictions.parquet"] = str(path); input_hashes[f"seed_{seed}_predictions.parquet"] = sha256_file(path)
    joined, pairing = _strict_join(baseline, lora_frames)
    baseline_thresholds, lora_thresholds, threshold_audit, threshold_hashes = _load_thresholds(config)
    input_hashes.update(threshold_hashes)
    frozen_metrics = json.loads((config.threshold_transfer_run / "publichearing_metrics.json").read_text(encoding="utf-8"))
    observed, observed_effects = _observed(joined, baseline_thresholds, lora_thresholds, frozen_metrics)
    signature, signature_payload = _signature(config, input_hashes, joined, {"baseline": baseline_thresholds, "lora": lora_thresholds})
    output_dir = (config.output_root / signature).resolve()
    common_result = {"model_loaded": False, "tokenizer_loaded": False, "cuda_initialized": False, "inference_executed": False, "training_executed": False}
    if validate_only:
        return {"status": "valid", "bootstrap_executed": False, **common_result, "n_examples": len(joined), "n_hearings": int(joined.hearing_id.nunique()), "positives": int(joined.label.sum()), "negatives": int((joined.label == 0).sum()), "seeds": list(config.seeds), "regimes": list(REGIMES), "n_bootstrap": config.n_replicates, "group_key": "hearing_id", "thresholds_reestimated": False, "preserve_group_multiplicity": True, "same_samples_across_methods": True, "same_samples_across_seeds": True, "same_samples_across_threshold_regimes": True, "analysis_signature": signature, "output_dir": str(output_dir), "pairing": pairing, "threshold_audit": threshold_audit}
    if output_dir.exists():
        if resume and (output_dir / "manifest.json").is_file():
            return _verify_completed(output_dir, signature)
        if resume and (output_dir / "bootstrap_replicates.parquet").is_file():
            return _rebuild_from_complete_replicates(output_dir, config, signature, signature_payload, observed, observed_effects, threshold_audit, input_hashes, input_paths, pairing)
        raise FileExistsError(f"Saída já existe; use --resume: {output_dir}")
    labels = joined.label.to_numpy(dtype=int); groups = sorted(joined.hearing_id.astype(str).unique().tolist()); group_values = joined.hearing_id.astype(str).to_numpy()
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    base_scores = joined.baseline_score.to_numpy(float); seed_scores = {seed: joined[f"seed_{seed}"].to_numpy(float) for seed in config.seeds}
    base_predictions = {regime: base_scores >= baseline_thresholds[regime] for regime in REGIMES}
    lora_predictions = {seed: {regime: seed_scores[seed] >= lora_thresholds[seed][regime] for regime in REGIMES} for seed in config.seeds}
    rng = np.random.default_rng(config.seed)
    n_rows = config.n_replicates * len(REGIMES) * len(config.seeds) * len(METRICS)
    columns: dict[str, np.ndarray] = {
        "replicate_id": np.empty(n_rows, dtype=np.int32), "bootstrap_seed": np.full(n_rows, config.seed, dtype=np.int32),
        "regime_code": np.empty(n_rows, dtype=np.int8), "seed": np.empty(n_rows, dtype=np.int8), "metric_code": np.empty(n_rows, dtype=np.int8),
        "threshold_baseline": np.empty(n_rows, dtype=float), "threshold_lora": np.empty(n_rows, dtype=float),
        "baseline_value": np.full(n_rows, np.nan, dtype=float), "lora_value": np.full(n_rows, np.nan, dtype=float), "effect": np.full(n_rows, np.nan, dtype=float),
        "valid": np.empty(n_rows, dtype=bool), "n_rows": np.empty(n_rows, dtype=np.int32), "n_sampled_groups": np.full(n_rows, len(groups), dtype=np.int16),
        "n_unique_groups": np.empty(n_rows, dtype=np.int16), "n_positive": np.empty(n_rows, dtype=np.int32), "n_negative": np.empty(n_rows, dtype=np.int32),
    }
    invalid_reasons: list[str | None] = [None] * n_rows
    row = 0
    for replicate_id in range(config.n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups)); indices = np.concatenate([group_indices[position] for position in sampled_positions]); sampled_labels = labels[indices]
        valid, reason = np.unique(sampled_labels).size == 2, None
        if not valid: reason = "single_class_resample"
        for regime in REGIMES:
            baseline_values = _bootstrap_metric_values(sampled_labels, base_predictions[regime][indices]) if valid else None
            for seed in config.seeds:
                threshold_base, threshold_lora = baseline_thresholds[regime], lora_thresholds[seed][regime]
                if valid:
                    lora_values = _bootstrap_metric_values(sampled_labels, lora_predictions[seed][regime][indices])
                for metric_index, metric in enumerate(METRICS):
                    columns["replicate_id"][row] = replicate_id; columns["regime_code"][row] = REGIMES.index(regime); columns["seed"][row] = seed; columns["metric_code"][row] = metric_index
                    columns["threshold_baseline"][row] = threshold_base; columns["threshold_lora"][row] = threshold_lora; columns["valid"][row] = valid
                    columns["n_rows"][row] = len(indices); columns["n_unique_groups"][row] = len(set(sampled_positions.tolist())); columns["n_positive"][row] = int(sampled_labels.sum()); columns["n_negative"][row] = int((sampled_labels == 0).sum()); invalid_reasons[row] = reason
                    if valid:
                        columns["baseline_value"][row] = baseline_values[metric]; columns["lora_value"][row] = lora_values[metric]; columns["effect"][row] = lora_values[metric] - baseline_values[metric]
                    row += 1
    if row != n_rows:
        raise RuntimeError("Quantidade inesperada de linhas bootstrap")
    replicate_frame = pd.DataFrame(columns)
    replicate_frame.insert(2, "regime", pd.Categorical.from_codes(replicate_frame.pop("regime_code"), categories=REGIMES))
    replicate_frame.insert(4, "metric", pd.Categorical.from_codes(replicate_frame.pop("metric_code"), categories=METRICS))
    replicate_frame.insert(11, "invalid_reason", pd.Categorical(invalid_reasons))
    threshold_identity = replicate_frame.groupby(["regime", "seed"], observed=True)[["threshold_baseline", "threshold_lora"]].nunique(dropna=False)
    if (threshold_identity.to_numpy() != 1).any():
        raise RuntimeError("Threshold variou entre réplicas")
    per_seed, campaign, validity = _derive_results(replicate_frame, observed, observed_effects, config)
    hashes_after = {key: sha256_file(Path(path)) for key, path in input_paths.items()}; hashes_after.update({key: sha256_file(config.threshold_transfer_run / ("thresholds.json" if key == "thresholds.json" else "lora_threshold_audit.json" if key == "lora_threshold_audit.json" else "manifest.json")) for key in threshold_hashes})
    if hashes_after != input_hashes:
        raise RuntimeError("Input congelado foi alterado durante a análise")
    output_dir.parent.mkdir(parents=True, exist_ok=True); stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    try:
        replicate_frame.to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        write_json(stage / "observed_metrics.json", observed); write_json(stage / "observed_effects.json", observed_effects); write_json(stage / "per_seed_results.json", per_seed); write_json(stage / "campaign_summary.json", campaign); write_json(stage / "threshold_audit.json", threshold_audit); write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {**common_result, "inputs": input_hashes, "inputs_reloaded_hashes": hashes_after, "input_paths": input_paths, "pairing": pairing, "n_examples": len(joined), "n_hearings": len(groups), "n_bootstrap_requested": config.n_replicates, "validity": validity, "paired": True, "grouped": True, "preserve_group_multiplicity": True, "same_samples_across_methods": True, "same_samples_across_seeds": True, "same_samples_across_threshold_regimes": True, "thresholds_reestimated": False, "thresholds_identical_all_replicates": True, "ensemble_used": False, "best_seed_selected": False, "publichearing_labels_used_for_threshold_selection": False, "decision_operator": OPERATOR, "mcc_implementation": "sklearn.metrics.matthews_corrcoef via binary_metrics"}
        write_json(stage / "integrity_audit.json", integrity); _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "seconds": time.time() - started, "bootstrap_executed": True, "model_loaded": False, "cuda_initialized": False}, ensure_ascii=False) + "\n"); _atomic_text(stage / "report.md", _report(signature, observed, per_seed, campaign, validity))
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": input_hashes, "counts": {"examples": len(joined), "hearings": len(groups), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "bootstrap_requested": config.n_replicates, "bootstrap_valid": validity["best_f1"]["valid_replicates"], "bootstrap_invalid": validity["best_f1"]["invalid_replicates"]}, "artifacts": {}}
        manifest["artifacts"] = _file_hashes(stage); write_json(stage / "manifest.json", manifest); os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

GENERIC_SCHEMA_VERSION = "publichearing-thresholded-paired-grouped-bootstrap-v2"
GENERIC_CODE_VERSION = "thresholded-paired-grouped-bootstrap-v2"
GENERIC_METRICS = ("precision", "recall", "f1", "mcc", "fpr", "balanced_accuracy", "specificity", "accuracy")
GENERIC_METRIC_KEYS = {
    "precision": "Precision", "recall": "Recall", "f1": "F1", "mcc": "MCC", "fpr": "FPR",
    "balanced_accuracy": "BalancedAccuracy", "specificity": "Specificity", "accuracy": "Accuracy",
}
GENERIC_REGIME_KEYS = {"best_f1": "f1", "fpr10": "fpr10"}


def run_generic_thresholded_pair_bootstrap(
    frames: dict[str, dict[int, pd.DataFrame]],
    thresholds: dict[str, dict[int, dict[str, float]]],
    condition_a: str,
    condition_b: str,
    *,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    regimes: tuple[str, ...] = REGIMES,
    n_replicates: int = 10000,
    seed: int = 20260815,
    confidence_level: float = 0.95,
    score_column: str = "probability",
    expected_rows: int = EXPECTED_ROWS,
    expected_positives: int = EXPECTED_POSITIVES,
    expected_hearings: int = EXPECTED_HEARINGS,
) -> dict[str, Any]:
    if condition_a == condition_b or condition_a not in frames or condition_b not in frames:
        raise ValueError("Condições thresholded inválidas")
    if set(regimes) != set(REGIMES) or n_replicates < 1:
        raise ValueError("Regimes ou número de réplicas inválido")
    from .paired_grouped_bootstrap import validate_generic_paired_frames

    pair_frames = {condition_a: frames[condition_a], condition_b: frames[condition_b]}
    population = validate_generic_paired_frames(pair_frames, conditions=(condition_a, condition_b), seeds=seeds,
                                                 expected_rows=expected_rows, expected_positives=expected_positives,
                                                 expected_hearings=expected_hearings, score_column=score_column)
    reference = frames[condition_a][seeds[0]][["example_id", "hearing_id", "label"]].copy()
    reference["example_id"] = reference.example_id.astype(str)
    reference = reference.set_index("example_id").sort_index()
    labels = reference.label.to_numpy(int)
    group_values = reference.hearing_id.astype(str).to_numpy()
    groups = sorted(np.unique(group_values).tolist())
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    scores: dict[str, dict[int, np.ndarray]] = {condition_a: {}, condition_b: {}}
    observed: dict[str, dict[str, dict[str, float]]] = {condition_a: {}, condition_b: {}}
    for condition in (condition_a, condition_b):
        for paired_seed in seeds:
            current = frames[condition][paired_seed].copy()
            current["example_id"] = current.example_id.astype(str)
            indexed = current.set_index("example_id").sort_index()
            values = indexed.loc[reference.index, score_column].to_numpy(float)
            scores[condition][paired_seed] = values
            observed[condition][str(paired_seed)] = {}
            for regime in regimes:
                observed[condition][str(paired_seed)][regime] = _generic_metrics(labels, values, thresholds[condition][paired_seed][regime])
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for replicate_id in range(n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([group_indices[position] for position in sampled_positions])
        sampled_labels = labels[indices]
        valid = np.unique(sampled_labels).size == 2
        reason = None if valid else "single_class_resample"
        for regime in regimes:
            for paired_seed in seeds:
                values_a = _bootstrap_metric_values(sampled_labels, scores[condition_a][paired_seed][indices] >= thresholds[condition_a][paired_seed][regime]) if valid else None
                values_b = _bootstrap_metric_values(sampled_labels, scores[condition_b][paired_seed][indices] >= thresholds[condition_b][paired_seed][regime]) if valid else None
                for metric in ("precision", "recall", "f1", "mcc", "fpr", "balanced_accuracy", "specificity", "accuracy"):
                    key = GENERIC_METRIC_KEYS[metric]
                    rows.append({"replicate_id": replicate_id, "regime": regime, "seed": paired_seed, "metric": metric,
                                 "condition_a_value": np.nan if not valid else values_a[metric],
                                 "condition_b_value": np.nan if not valid else values_b[metric],
                                 "delta": np.nan if not valid else values_b[metric] - values_a[metric],
                                 "valid": valid, "invalid_reason": reason, "threshold_a": thresholds[condition_a][paired_seed][regime],
                                 "threshold_b": thresholds[condition_b][paired_seed][regime], "n_rows": len(indices),
                                 "n_sampled_groups": len(groups), "n_unique_groups": len(np.unique(sampled_positions)),
                                 "n_positive": int(sampled_labels.sum()), "n_negative": int((sampled_labels == 0).sum())})
    replicate_frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for regime in regimes:
        summaries[regime] = {}
        for metric in ("precision", "recall", "f1", "mcc", "fpr", "balanced_accuracy", "specificity", "accuracy"):
            key = GENERIC_METRIC_KEYS[metric]
            valid_rows = replicate_frame.loc[(replicate_frame.regime == regime) & (replicate_frame.metric == metric) & replicate_frame.valid]
            values = valid_rows.groupby("replicate_id", sort=False)["delta"].mean().to_numpy(float)
            point_a = float(np.mean([observed[condition_a][str(s)][regime][key] for s in seeds]))
            point_b = float(np.mean([observed[condition_b][str(s)][regime][key] for s in seeds]))
            summaries[regime][metric] = {"point_a": point_a, "point_b": point_b, "observed_delta": point_b - point_a,
                                         "bootstrap": _generic_summary(values, confidence_level, metric == "fpr"),
                                         "valid_replicates": int(len(values)), "invalid_replicates": int(n_replicates - len(values))}
    return {"population": population, "observed": observed, "summaries": summaries, "replicates": replicate_frame,
            "protocol": {"paired": True, "group_key": "hearing_id", "delta_orientation": "condition_b - condition_a",
                         "n_replicates": n_replicates, "seed": seed, "confidence_level": confidence_level,
                         "same_samples_across_seeds": True, "same_samples_across_regimes": True,
                         "preserve_group_multiplicity": True, "thresholds_reestimated": False,
                         "threshold_source": "validation artifacts"}}


@dataclass(frozen=True)
class CampaignSpec:
    label: str
    path: Path
    signature: str
    config_path: Path
    expected_pooling: str
    score_column: str = "probability"


@dataclass(frozen=True)
class GenericThresholdedBootstrapConfig:
    output_root: Path
    attention: CampaignSpec
    set_transformer: CampaignSpec
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]
    group_key: str
    n_replicates: int
    seed: int
    confidence_level: float
    ci_method: str
    expected_examples: int
    expected_positives: int
    expected_hearings: int
    expected_publichearing_dataset_signature: str
    expected_ragtruth_dataset_signature: str

    @classmethod
    def from_yaml(cls, path: Path) -> "GenericThresholdedBootstrapConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("analysis", {}).get("type") != "thresholded_paired_grouped_bootstrap":
            raise ValueError(f"Configuração genérica inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        inputs = raw.get("inputs", {})

        def campaign(name: str) -> CampaignSpec:
            value = inputs.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"Campaign input ausente: {name}")
            return CampaignSpec(
                label=str(value.get("label", name)),
                path=resolve(value.get("path")),
                signature=str(value.get("expected_signature")),
                config_path=resolve(value.get("config_path")),
                expected_pooling=str(value.get("expected_pooling")),
                score_column=str(value.get("score_column", "probability")),
            )

        bootstrap = raw.get("bootstrap", {})
        population = raw.get("population", {})
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_pt_nllb_attention_vs_set_thresholded_bootstrap")),
            attention=campaign("attention"),
            set_transformer=campaign("set_transformer"),
            seeds=tuple(int(seed) for seed in inputs.get("seeds", [0, 1, 2])),
            regimes=tuple(str(regime) for regime in raw.get("threshold_regimes", REGIMES)),
            group_key=str(raw.get("protocol", {}).get("group_key", "hearing_id")),
            n_replicates=int(bootstrap.get("n_replicates", 10000)),
            seed=int(bootstrap.get("seed", 20260815)),
            confidence_level=float(bootstrap.get("confidence_level", 0.95)),
            ci_method=str(bootstrap.get("ci_method", "percentile")),
            expected_examples=int(population.get("examples", EXPECTED_ROWS)),
            expected_positives=int(population.get("positives", EXPECTED_POSITIVES)),
            expected_hearings=int(population.get("hearings", EXPECTED_HEARINGS)),
            expected_publichearing_dataset_signature=str(population.get("publichearing_dataset_signature", "")),
            expected_ragtruth_dataset_signature=str(population.get("ragtruth_dataset_signature", "")),
        )
        config.validate(raw.get("protocol", {}))
        return config

    def validate(self, protocol: dict[str, Any]) -> None:
        if protocol.get("analysis_type") != "thresholded_paired_grouped_bootstrap" or protocol.get("paired") is not True:
            raise ValueError("A análise deve ser thresholded, pareada e agrupada")
        if self.group_key != "hearing_id" or protocol.get("group_key") != "hearing_id":
            raise ValueError("O agrupamento deve ser hearing_id")
        if self.seeds != EXPECTED_SEEDS or self.regimes != REGIMES:
            raise ValueError("A análise exige seeds [0, 1, 2] e regimes best_f1/fpr10")
        if self.n_replicates < 1 or self.seed < 0 or not 0 < self.confidence_level < 1 or self.ci_method != "percentile":
            raise ValueError("Configuração de bootstrap inválida")
        if self.expected_examples < 1 or self.expected_positives < 1 or self.expected_hearings < 1:
            raise ValueError("Population contract inválido")
        for spec in (self.attention, self.set_transformer):
            if spec.score_column != "probability" or spec.expected_pooling not in {"attention", "gated_attention", "set_transformer"}:
                raise ValueError("Campaign score/pooling inválido")
        if self.attention.expected_pooling not in {"attention", "gated_attention"} or self.set_transformer.expected_pooling != "set_transformer":
            raise ValueError("A comparação exige Attention MIL contra Set Transformer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": {"type": "thresholded_paired_grouped_bootstrap", "name": "PT NLLB Attention MIL vs Set Transformer on PublicHearingBR"},
            "protocol": {"analysis_type": "thresholded_paired_grouped_bootstrap", "group_key": self.group_key, "paired": True,
                         "delta": "Set Transformer - Attention MIL", "threshold_provenance": "RAGTruth validation only",
                         "threshold_reoptimization": False},
            "inputs": {"attention": _spec_to_dict(self.attention), "set_transformer": _spec_to_dict(self.set_transformer), "seeds": list(self.seeds)},
            "threshold_regimes": list(self.regimes),
            "population": {"examples": self.expected_examples, "positives": self.expected_positives, "hearings": self.expected_hearings,
                            "publichearing_dataset_signature": self.expected_publichearing_dataset_signature,
                            "ragtruth_dataset_signature": self.expected_ragtruth_dataset_signature},
            "bootstrap": {"n_replicates": self.n_replicates, "seed": self.seed, "confidence_level": self.confidence_level,
                          "ci_method": self.ci_method, "cpu_only": True, "model_load": False, "inference": False},
            "output": {"root": str(self.output_root)},
        }


def _spec_to_dict(spec: CampaignSpec) -> dict[str, Any]:
    return {"label": spec.label, "path": str(spec.path), "expected_signature": spec.signature,
            "config_path": str(spec.config_path), "expected_pooling": spec.expected_pooling, "score_column": spec.score_column}


def load_thresholded_config(path: Path) -> ThresholdedBootstrapConfig | GenericThresholdedBootstrapConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("analysis", {}).get("type") == "thresholded_paired_grouped_bootstrap":
        return GenericThresholdedBootstrapConfig.from_yaml(path)
    return ThresholdedBootstrapConfig.from_yaml(path)


def _generic_validate_manifest(directory: Path, signature: str | None = None, statuses: set[str] | None = None) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if signature is not None and str(manifest.get("signature")) != signature:
        raise ValueError(f"Assinatura incompatível em {manifest_path}")
    if statuses is not None and manifest.get("status") not in statuses:
        raise ValueError(f"Status incompatível em {manifest_path}: {manifest.get('status')}")
    for name, digest in manifest.get("artifacts", {}).items():
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"Artefato ausente/alterado: {artifact}")
    return manifest


def _generic_validate_campaign(spec: CampaignSpec, seeds: tuple[int, ...], expected_dataset_signature: str) -> dict[str, Any]:
    manifest = _generic_validate_manifest(spec.path, spec.signature, {"externally_evaluated", "completed"})
    if str(manifest.get("pooling_type")) != spec.expected_pooling:
        raise ValueError(f"Pooling incompatível em {spec.label}: {manifest.get('pooling_type')}")
    if tuple(int(seed) for seed in manifest.get("seeds", [])) != seeds:
        raise ValueError(f"Seeds incompatíveis em {spec.label}")
    if not spec.config_path.is_file():
        raise FileNotFoundError(f"Configuração declarada ausente: {spec.config_path}")
    resolved = json.loads((spec.path / "resolved_config.json").read_text(encoding="utf-8"))
    architecture = str(resolved.get("architecture"))
    expected_arch = "set_transformer" if spec.expected_pooling == "set_transformer" else "gated_attention"
    if architecture != expected_arch:
        raise ValueError(f"Arquitetura incompatível em {spec.label}: {architecture}")
    for seed in seeds:
        seed_dir = spec.path / f"seed_{seed}"
        seed_manifest = _generic_validate_manifest(seed_dir, statuses={"trained", "externally_evaluated"})
        if int(seed_manifest.get("seed", seed)) != seed:
            raise ValueError(f"Seed manifest incompatível: {seed_dir}")
        if seed_manifest.get("pooling_type") not in {spec.expected_pooling, "attention" if spec.expected_pooling == "gated_attention" else spec.expected_pooling}:
            raise ValueError(f"Pooling da seed incompatível: {seed_dir}")
    return {"manifest": manifest, "resolved_config": resolved, "config_path": str(spec.config_path), "expected_dataset_signature": expected_dataset_signature}


def _generic_validate_prediction(path: Path, score_column: str, expected_rows: int, expected_positives: int,
                                 expected_hearings: int, dataset_signature: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"example_id", "hearing_id", "label", score_column, "publichearing_dataset_signature", "model_run_signature"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")
    selected = frame[["example_id", "hearing_id", "label", score_column, "publichearing_dataset_signature", "model_run_signature"]].copy()
    selected["example_id"] = selected["example_id"].astype(str)
    selected["hearing_id"] = selected["hearing_id"].astype(str)
    if len(selected) != expected_rows or not selected.example_id.is_unique or selected.hearing_id.nunique() != expected_hearings:
        raise ValueError(f"Population/IDs inválidos: {path}")
    if selected.isna().any().any() or set(selected.label.astype(int).unique()) != {0, 1} or int(selected.label.astype(int).sum()) != expected_positives:
        raise ValueError(f"Labels/nulos inválidos: {path}")
    if set(selected.publichearing_dataset_signature.astype(str)) != {dataset_signature}:
        raise ValueError(f"Dataset PublicHearing divergente: {path}")
    scores = selected[score_column].to_numpy(float)
    if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
        raise ValueError(f"Probabilidades inválidas: {path}")
    return selected


def _generic_locate_predictions(spec: CampaignSpec, config: GenericThresholdedBootstrapConfig, seed: int) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    root = spec.path / f"seed_{seed}" / "publichearing_zero_shot"
    manifests = sorted(root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Esperava um artifact PublicHearing em {spec.label}/seed_{seed}; encontrei {len(manifests)}")
    run_dir = manifests[0].parent
    manifest = _generic_validate_manifest(run_dir, run_dir.name, {"completed", "externally_evaluated"})
    if manifest.get("pooling_type") not in {spec.expected_pooling, "attention" if spec.expected_pooling == "gated_attention" else spec.expected_pooling}:
        raise ValueError(f"Pooling PublicHearing incompatível: {run_dir}")
    prediction = run_dir / "predictions.parquet"
    expected_hash = manifest.get("artifacts", {}).get("predictions.parquet")
    if expected_hash and sha256_file(prediction) != expected_hash:
        raise ValueError(f"Hash da predição divergente: {prediction}")
    frame = _generic_validate_prediction(prediction, spec.score_column, config.expected_examples, config.expected_positives,
                                         config.expected_hearings, config.expected_publichearing_dataset_signature)
    if "seed" in pd.read_parquet(prediction, columns=["seed"]).columns:
        seed_values = set(pd.read_parquet(prediction, columns=["seed"])["seed"].astype(int).unique())
        if seed_values != {seed}:
            raise ValueError(f"Coluna seed incompatível em {prediction}: {seed_values}")
    if manifest.get("threshold_origin") != "RAGTruth validation only":
        raise ValueError(f"Threshold provenance incompatível em {run_dir}")
    return prediction, manifest, frame


def _generic_strict_join(frames: dict[str, dict[int, pd.DataFrame]], config: GenericThresholdedBootstrapConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = frames["attention"][config.seeds[0]][["example_id", "hearing_id", "label"]].copy()
    reference["example_id"] = reference.example_id.astype(str)
    if not reference.example_id.is_unique:
        raise ValueError("example_id duplicado na referência Attention")
    reference = reference.set_index("example_id").sort_index()
    audit: dict[str, Any] = {}
    for model in ("attention", "set_transformer"):
        audit[model] = {}
        for seed in config.seeds:
            current = frames[model][seed].set_index("example_id").sort_index()
            status = {
                "reference_only_ids": len(set(reference.index) - set(current.index)),
                "model_only_ids": len(set(current.index) - set(reference.index)),
                "duplicate_ids": int(not current.index.is_unique), "label_mismatches": 0,
                "hearing_id_mismatches": 0, "pairable_examples": len(set(reference.index) & set(current.index)),
            }
            common = sorted(set(reference.index) & set(current.index))
            if not status["reference_only_ids"] and not status["model_only_ids"]:
                status["label_mismatches"] = int((reference.loc[common, "label"].astype(int) != current.loc[common, "label"].astype(int)).sum())
                status["hearing_id_mismatches"] = int((reference.loc[common, "hearing_id"].astype(str) != current.loc[common, "hearing_id"].astype(str)).sum())
            audit[model][str(seed)] = status
            if any(status[key] for key in ("reference_only_ids", "model_only_ids", "duplicate_ids", "label_mismatches", "hearing_id_mismatches")) or len(common) != config.expected_examples:
                raise ValueError(f"Pareamento estrito falhou em {model}/seed_{seed}: {status}")
    joined = reference[["hearing_id", "label"]].copy()
    joined["example_id"] = joined.index
    joined = joined.reset_index(drop=True)
    for model in ("attention", "set_transformer"):
        for seed in config.seeds:
            current = frames[model][seed].set_index("example_id").sort_index()
            joined[f"{model}_score_{seed}"] = current.loc[reference.index, "probability"].to_numpy(float)
    return joined, audit


def _generic_threshold_audit(spec: CampaignSpec, config: GenericThresholdedBootstrapConfig) -> tuple[dict[int, dict[str, float]], dict[str, Any], dict[str, str]]:
    from .metrics import select_threshold

    thresholds: dict[int, dict[str, float]] = {}
    audit: dict[str, Any] = {"model": spec.label, "campaign": str(spec.path), "campaign_signature": spec.signature,
                             "threshold_source": "RAGTruth validation only", "publichearing_labels_used_for_threshold_selection": False, "seeds": {}}
    hashes: dict[str, str] = {}
    for seed in config.seeds:
        seed_dir = spec.path / f"seed_{seed}"
        persisted = json.loads((seed_dir / "thresholds.json").read_text(encoding="utf-8"))
        validation_path = seed_dir / "validation_predictions.csv"
        validation = pd.read_csv(validation_path)
        validation_score_column = "score" if "score" in validation.columns else "probability"
        if not {"label", validation_score_column}.issubset(validation.columns) or len(validation) != 4517:
            raise ValueError(f"Validation artifact inválido: {validation_path}")
        seed_manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
        thresholds[seed] = {}
        audit["seeds"][str(seed)] = {"checkpoint": seed_manifest.get("best_checkpoint"), "validation_examples": len(validation), "regimes": {}}
        for regime in config.regimes:
            key = GENERIC_REGIME_KEYS[regime]
            item = persisted.get(key, {})
            persisted_threshold = float(item.get("threshold", np.nan))
            recomputed, _, feasible = select_threshold(validation["label"].to_numpy(), validation[validation_score_column].to_numpy(float), key)
            validation_hash = sha256_file(validation_path)
            declared_hash = item.get("validation_sha256")
            if declared_hash and declared_hash != validation_hash:
                raise ValueError(f"Hash de validation divergente em {spec.label}/seed_{seed}/{regime}")
            if not np.isclose(persisted_threshold, recomputed, rtol=0, atol=1e-12):
                raise ValueError(f"Threshold não reproduzido em {spec.label}/seed_{seed}/{regime}")
            if item.get("validation_examples") != len(validation) or item.get("constraint_feasible") is not True:
                raise ValueError(f"Provenance de threshold inválida em {spec.label}/seed_{seed}/{regime}")
            thresholds[seed][regime] = persisted_threshold
            audit["seeds"][str(seed)]["regimes"][regime] = {
                "persisted_threshold": persisted_threshold, "recomputed_threshold": float(recomputed), "match": True,
                "validation_only": True, "source_artifact": str(seed_dir / "thresholds.json"),
                "validation_predictions": str(validation_path), "validation_sha256": validation_hash,
                "validation_examples": len(validation), "constraint_feasible": bool(feasible),
                "selection_rule": item.get("rule"), "checkpoint": seed_manifest.get("best_checkpoint"),
            }
            hashes[f"{spec.label}/seed_{seed}/thresholds.json"] = sha256_file(seed_dir / "thresholds.json")
            hashes[f"{spec.label}/seed_{seed}/validation_predictions.csv"] = validation_hash
        hashes[f"{spec.label}/manifest.json"] = sha256_file(spec.path / "manifest.json")
    return thresholds, audit, hashes


def _generic_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return binary_metrics(np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float) >= threshold)


def _generic_validate_observed_artifacts(spec: CampaignSpec, config: GenericThresholdedBootstrapConfig, seed: int, regime: str,
                                         metrics: dict[str, float | int], run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    declared = json.loads(metrics_path.read_text(encoding="utf-8"))
    declared_key = f"ragtruth_validation_{regime}_threshold"
    target = declared.get(declared_key, {})
    for key in ("TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "MCC", "FPR", "Specificity", "BalancedAccuracy"):
        if key not in target or not np.isclose(float(metrics[key]), float(target[key]), rtol=0, atol=1e-12):
            raise ValueError(f"Métrica observada não reproduz metrics.json: {spec.label}/seed_{seed}/{regime}/{key}")
    aggregate_path = spec.path / "aggregate" / "per_seed_metrics.csv"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"Aggregate per_seed_metrics ausente: {aggregate_path}")
    aggregate = pd.read_csv(aggregate_path)
    rows = aggregate[(aggregate["dataset"] == "publichearing_zero_shot") & (aggregate["seed"].astype(int) == seed) & (aggregate["threshold_regime"] == regime)]
    if len(rows) != 1:
        raise ValueError(f"Linha aggregate ausente/ambígua: {spec.label}/seed_{seed}/{regime}")
    aggregate_row = rows.iloc[0]
    for key in ("Precision", "Recall", "F1", "MCC", "FPR", "Specificity", "BalancedAccuracy"):
        if not np.isclose(float(metrics[key]), float(aggregate_row[key]), rtol=0, atol=1e-6):
            raise ValueError(f"Métrica observada não reproduz aggregate: {spec.label}/seed_{seed}/{regime}/{key}")
    return {"metrics_json": str(metrics_path), "aggregate_per_seed_metrics": str(aggregate_path), "metrics_json_match": True, "aggregate_match": True}


def _generic_summary(values: np.ndarray, confidence_level: float, favorable_lower: bool = False) -> dict[str, Any]:
    if not len(values):
        return {"mean": None, "median": None, "ci_lower": None, "ci_upper": None, "p_delta_gt_zero": None,
                "p_delta_lt_zero": None, "favorable_probability": None, "probability_favorable": None,
                "favorable_direction": "lower" if favorable_lower else "higher", "n_valid": 0}
    alpha = 1.0 - confidence_level
    probability = float(np.mean(values < 0 if favorable_lower else values > 0))
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci_lower": float(np.quantile(values, alpha / 2)),
            "ci_upper": float(np.quantile(values, 1 - alpha / 2)), "p_delta_gt_zero": float(np.mean(values > 0)),
            "p_delta_lt_zero": float(np.mean(values < 0)), "favorable_probability": probability, "probability_favorable": probability,
            "favorable_direction": "lower" if favorable_lower else "higher", "n_valid": int(len(values))}


def _generic_observed(config: GenericThresholdedBootstrapConfig, frames: dict[str, dict[int, pd.DataFrame]],
                      thresholds: dict[str, dict[int, dict[str, float]]]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}
    artifact_audit: dict[str, Any] = {"metrics_json_matches": {}}
    for regime in config.regimes:
        observed[regime] = {"attention": {}, "set_transformer": {}}
        artifact_audit["metrics_json_matches"][regime] = {}
        for model, spec in (("attention", config.attention), ("set_transformer", config.set_transformer)):
            for seed in config.seeds:
                frame = frames[model][seed]
                score = frame[spec.score_column].to_numpy(float)
                labels = frame.label.to_numpy(int)
                current = _generic_metrics(labels, score, thresholds[model][seed][regime])
                observed[regime][model][str(seed)] = current
                run_dir = sorted((spec.path / f"seed_{seed}" / "publichearing_zero_shot").glob("*/metrics.json"))[0].parent
                artifact_audit["metrics_json_matches"][regime][f"{model}/seed_{seed}"] = _generic_validate_observed_artifacts(spec, config, seed, regime, current, run_dir)
                row = {"regime": regime, "seed": seed, "model": spec.label, "threshold": thresholds[model][seed][regime]}
                row.update({GENERIC_METRIC_KEYS[key]: current[GENERIC_METRIC_KEYS[key]] for key in GENERIC_METRICS})
                rows.append(row)
    return observed, pd.DataFrame(rows), artifact_audit


def _generic_signature(config: GenericThresholdedBootstrapConfig, input_hashes: dict[str, str], joined: pd.DataFrame,
                       thresholds: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {"schema": GENERIC_SCHEMA_VERSION, "code_version": GENERIC_CODE_VERSION, "input_hashes": input_hashes,
               "campaigns": {"attention": config.attention.signature, "set_transformer": config.set_transformer.signature},
               "thresholds": thresholds, "regimes": list(config.regimes), "seeds": list(config.seeds), "metrics": list(GENERIC_METRICS),
               "delta": "set_transformer - attention", "group_key": config.group_key, "n_examples": len(joined),
               "n_hearings": int(joined.hearing_id.nunique()), "n_replicates": config.n_replicates, "bootstrap_seed": config.seed,
               "confidence_level": config.confidence_level, "ci_method": config.ci_method, "thresholds_reestimated": False,
               "same_hearing_samples": True, "same_samples_across_seeds": True}
    return _canonical_hash(payload)[:16], payload


def _generic_report(signature: str, config: GenericThresholdedBootstrapConfig, population: dict[str, Any], threshold_audit: dict[str, Any],
                    observed: dict[str, Any], summary: dict[str, Any], per_seed: pd.DataFrame, validity: dict[str, Any]) -> str:
    lines = ["# PublicHearingBR — PT NLLB Attention MIL vs Set Transformer", "", "## Classification", "", "PASS", "",
             "## Executive summary", "", "- Bootstrap pareado agrupado por `hearing_id`, com thresholds congelados da RAGTruth validation.",
             f"- População alinhada: {population['examples']} exemplos, {population['positives']} positivos, {population['hearings']} hearings.",
             f"- Foram executadas {config.n_replicates:,} réplicas por regime, com seed {config.seed}.",
             "- Delta definido como Set Transformer − Attention MIL.",
             "- FPR usa P(Δ < 0) como probabilidade favorável; as demais métricas usam P(Δ > 0).",
             "- Nenhum modelo, checkpoint, CUDA, inferência ou treino foi executado nesta análise.", "",
             "## Implementation summary", "", "- existing core reused: `thresholded_paired_grouped_bootstrap.py`, `binary_metrics`, `select_threshold`.",
             "- files generalized: `src/ragtruth_transfer/thresholded_paired_grouped_bootstrap.py` and the existing entrypoint.",
             "- new analysis config: `configs/publichearing_pt_nllb_attention_vs_set_thresholded_bootstrap.yaml`.",
             "- new statistical method introduced: NO.", "- backward compatibility preserved: YES (historical config mode retained).", "",
             "## Compared campaigns", "", "### Attention MIL", "", f"- campaign: `{config.attention.path}` ({config.attention.signature})",
             f"- config: `{config.attention.config_path}`", f"- pooling: `{config.attention.expected_pooling}`", f"- seeds: {list(config.seeds)}", "",
             "### Set Transformer", "", f"- campaign: `{config.set_transformer.path}` ({config.set_transformer.signature})",
             f"- config: `{config.set_transformer.config_path}`", f"- pooling: `{config.set_transformer.expected_pooling}`", f"- seeds: {list(config.seeds)}", "",
             "## Population integrity", "", *[f"- {key}: {value}" for key, value in population.items()], "",
             "## Threshold provenance audit", "", "All 12 model/seed/regime thresholds matched recomputation from the model's 4,517-row RAGTruth validation artifact. PublicHearing labels were not used for selection.", "",
             "| Model | Seed | Regime | Persisted threshold | Recomputed threshold | Match | Validation-only |", "|---|---:|---|---:|---:|---|---|"]
    for model_key, model_label in (("attention", config.attention.label), ("set_transformer", config.set_transformer.label)):
        for seed in config.seeds:
            for regime in config.regimes:
                item = threshold_audit["models"][model_key]["seeds"][str(seed)]["regimes"][regime]
                lines.append(f"| {model_label} | {seed} | {regime} | {item['persisted_threshold']:.10f} | {item['recomputed_threshold']:.10f} | YES | YES |")
    lines.append("")
    for regime in config.regimes:
        lines += [f"## {regime} — per-seed observed metrics", "", "| Seed | Model | Threshold | Precision | Recall | F1 | MCC | FPR | TP | FP | TN | FN |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in per_seed[per_seed.regime == regime].iterrows():
            m = observed[regime]["attention" if row.model == config.attention.label else "set_transformer"][str(int(row.seed))]
            lines.append(f"| {int(row.seed)} | {row.model} | {row.threshold:.10f} | {row.Precision:.6f} | {row.Recall:.6f} | {row.F1:.6f} | {row.MCC:.6f} | {row.FPR:.6f} | {int(m['TP'])} | {int(m['FP'])} | {int(m['TN'])} | {int(m['FN'])} |")
        lines += ["", f"## {regime} — campaign bootstrap", "", "| Metric | Attention | Set | Observed Δ | Bootstrap mean Δ | 95% CI | Favorable probability |", "|---|---:|---:|---:|---:|---|---:|"]
        for metric in GENERIC_METRICS[:5]:
            item = summary[regime]["metrics"][metric]
            lines.append(f"| {GENERIC_METRIC_KEYS[metric]} | {item['attention_mean']:.6f} | {item['set_mean']:.6f} | {item['observed_delta']:.6f} | {item['bootstrap']['mean']:.6f} | [{item['bootstrap']['ci_lower']:.6f}, {item['bootstrap']['ci_upper']:.6f}] | {item['bootstrap']['favorable_probability']:.4f} |")
        lines += ["", f"## {regime} interpretation", "", f"- F1: `{summary[regime]['interpretation']['f1']}`", f"- precision effect: {summary[regime]['interpretation']['precision']}", f"- recall effect: {summary[regime]['interpretation']['recall']}", f"- MCC effect: {summary[regime]['interpretation']['mcc']}", f"- FPR effect: {summary[regime]['interpretation']['fpr']}", ""]
    lines += ["## Operating-point transfer", "", "`fpr10` thresholds were selected to satisfy FPR ≤ 0.10 on RAGTruth validation; target FPR is reported as transfer behavior and is not reoptimized.", "", "| Model | Seed | Regime | Validation goal | PublicHearing FPR | PublicHearing F1 |", "|---|---:|---|---|---:|---:|"]
    for _, row in per_seed.iterrows():
        goal = "F1 maximization" if row.regime == "best_f1" else "FPR ≤ 0.10"
        lines.append(f"| {row.model} | {int(row.seed)} | {row.regime} | {goal} | {row.FPR:.6f} | {row.F1:.6f} |")
    lines += ["", "## Ranking vs decision comparison", "", "- ranking: prior threshold-free bootstrap favored Set Transformer (ΔAUPRC +0.028869; IC95% [+0.012958, +0.044219]).", "- calibration: prior ΔBrier +0.007179 indicated worse calibration for Set Transformer because lower Brier is better.", "- thresholded decision results are reported above separately by regime; they characterize transfer of validation operating points rather than target-optimized performance.", "", "## Main scientific interpretation", ""]
    lines.append(f"- Overall: `{summary['main_interpretation']}`")
    lines.append("- The conclusion is based on campaign-level observed deltas, percentile CIs, and consistency across the three paired seeds; it does not infer a causal mechanism from Precision/Recall alone.")
    lines += ["", "## Implication for primary aggregator", "", "SET TRANSFORMER REMAINS PRIMARY WITH OPERATING-POINT CAVEAT", "", "AUPRC remains the primary metric; thresholded behavior is an operational characterization.", "", "## MADLAD implication", "", "- architecture to run first: Set Transformer.", "- still run Attention MADLAD: YES, as a secondary architectural control.", "- rationale: preserve the cross-backbone comparison while retaining the validated Attention baseline.", "", "## Tests and execution integrity", "", f"- valid bootstrap replicates: {validity}", "- focused tests/compileall/git diff --check: recorded by executor.", "- GPU used: NO; CUDA initialized: NO; model/checkpoint loaded: NO; forward/inference/training: NO.", "- PublicHearing re-evaluated: NO; datasets/predictions/historical runs modified: NO; commit created: NO.", "", "## Final verdict", "", "The ranking gain is evaluated independently from operating-point transfer. Set Transformer remains the primary PT aggregator, with a caveat whenever a validation-derived threshold does not transfer to the target at the desired FPR/F1 operating point."]
    return "\n".join(lines) + "\n"


def run_generic_thresholded_bootstrap(config: GenericThresholdedBootstrapConfig, *, validate_only: bool = False) -> dict[str, Any]:
    started = time.time()
    campaign_audits = {"attention": _generic_validate_campaign(config.attention, config.seeds, config.expected_ragtruth_dataset_signature),
                       "set_transformer": _generic_validate_campaign(config.set_transformer, config.seeds, config.expected_ragtruth_dataset_signature)}
    frames: dict[str, dict[int, pd.DataFrame]] = {"attention": {}, "set_transformer": {}}
    input_paths: dict[str, str] = {}; input_hashes: dict[str, str] = {}
    run_dirs: dict[str, dict[int, Path]] = {"attention": {}, "set_transformer": {}}
    manifests: dict[str, dict[int, dict[str, Any]]] = {"attention": {}, "set_transformer": {}}
    for model, spec in (("attention", config.attention), ("set_transformer", config.set_transformer)):
        for seed in config.seeds:
            path, manifest, frame = _generic_locate_predictions(spec, config, seed)
            frames[model][seed] = frame
            run_dirs[model][seed] = path.parent
            manifests[model][seed] = manifest
            key = f"{model}/seed_{seed}/predictions.parquet"; input_paths[key] = str(path); input_hashes[key] = sha256_file(path)
    joined, pairing = _generic_strict_join(frames, config)
    threshold_values: dict[str, dict[int, dict[str, float]]] = {}
    threshold_audits: dict[str, Any] = {"threshold_source": "RAGTruth validation only", "publichearing_labels_used_for_threshold_selection": False, "models": {}}
    for model, spec in (("attention", config.attention), ("set_transformer", config.set_transformer)):
        values, audit, hashes = _generic_threshold_audit(spec, config)
        threshold_values[model] = values; threshold_audits["models"][model] = audit; input_hashes.update(hashes)
    observed, per_seed, observed_artifact_audit = _generic_observed(config, frames, threshold_values)
    threshold_audits["observed_artifact_reproduction"] = observed_artifact_audit
    population = {"examples": len(joined), "positives": int(joined.label.sum()), "prevalence": float(joined.label.mean()), "hearings": int(joined.hearing_id.nunique()),
                  "exact_example_alignment": True, "exact_label_alignment": True, "exact_hearing_alignment": True, "probabilities_valid": True}
    if population["examples"] != config.expected_examples or population["positives"] != config.expected_positives or population["hearings"] != config.expected_hearings:
        raise ValueError(f"Population divergente: {population}")
    signature, signature_payload = _generic_signature(config, input_hashes, joined, threshold_values)
    output_dir = (config.output_root / signature).resolve()
    preflight = {"status": "valid", "bootstrap_executed": False, "model_loaded": False, "checkpoint_loaded": False, "cuda_initialized": False,
                 "inference_executed": False, "training_executed": False, "n_examples": len(joined), "n_hearings": int(joined.hearing_id.nunique()),
                 "positives": int(joined.label.sum()), "seeds": list(config.seeds), "regimes": list(config.regimes), "n_bootstrap": config.n_replicates,
                 "group_key": config.group_key, "thresholds_reestimated": False, "thresholds_frozen": True, "analysis_signature": signature,
                 "output_dir": str(output_dir), "pairing": pairing, "threshold_audit": threshold_audits}
    if validate_only:
        return preflight
    if output_dir.exists():
        raise FileExistsError(f"Saída já existe; não sobrescrevendo: {output_dir}")
    labels = joined.label.to_numpy(int); group_values = joined.hearing_id.astype(str).to_numpy(); groups = sorted(set(group_values)); group_indices = [np.flatnonzero(group_values == group) for group in groups]
    predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {"attention": {}, "set_transformer": {}}
    for model, spec in (("attention", config.attention), ("set_transformer", config.set_transformer)):
        for seed in config.seeds:
            scores = joined[f"{model}_score_{seed}"].to_numpy(float)
            predictions[model][seed] = {regime: scores >= threshold_values[model][seed][regime] for regime in config.regimes}
    n_rows = config.n_replicates * len(config.regimes) * len(config.seeds) * len(GENERIC_METRICS)
    columns = {"replicate_id": np.empty(n_rows, np.int32), "bootstrap_seed": np.full(n_rows, config.seed, np.int32), "regime": np.empty(n_rows, object),
               "seed": np.empty(n_rows, np.int8), "metric": np.empty(n_rows, object), "threshold_attention": np.empty(n_rows, float), "threshold_set_transformer": np.empty(n_rows, float),
               "attention_value": np.full(n_rows, np.nan), "set_transformer_value": np.full(n_rows, np.nan), "delta": np.full(n_rows, np.nan), "valid": np.empty(n_rows, bool),
               "invalid_reason": np.empty(n_rows, object), "n_rows": np.empty(n_rows, np.int32), "n_sampled_groups": np.full(n_rows, len(groups), np.int16), "n_unique_groups": np.empty(n_rows, np.int16),
               "n_positive": np.empty(n_rows, np.int32), "n_negative": np.empty(n_rows, np.int32)}
    rng = np.random.default_rng(config.seed); row = 0
    for replicate_id in range(config.n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups)); indices = np.concatenate([group_indices[pos] for pos in sampled_positions]); sampled_labels = labels[indices]
        valid = np.unique(sampled_labels).size == 2; reason = None if valid else "single_class_resample"
        for regime in config.regimes:
            for seed in config.seeds:
                attention_values = _bootstrap_metric_values(sampled_labels, predictions["attention"][seed][regime][indices]) if valid else None
                set_values = _bootstrap_metric_values(sampled_labels, predictions["set_transformer"][seed][regime][indices]) if valid else None
                for metric in GENERIC_METRICS:
                    columns["replicate_id"][row] = replicate_id; columns["regime"][row] = regime; columns["seed"][row] = seed; columns["metric"][row] = metric
                    columns["threshold_attention"][row] = threshold_values["attention"][seed][regime]; columns["threshold_set_transformer"][row] = threshold_values["set_transformer"][seed][regime]
                    columns["valid"][row] = valid; columns["invalid_reason"][row] = reason; columns["n_rows"][row] = len(indices); columns["n_unique_groups"][row] = len(set(sampled_positions.tolist())); columns["n_positive"][row] = int(sampled_labels.sum()); columns["n_negative"][row] = int((sampled_labels == 0).sum())
                    if valid:
                        columns["attention_value"][row] = attention_values[metric]; columns["set_transformer_value"][row] = set_values[metric]; columns["delta"][row] = set_values[metric] - attention_values[metric]
                    row += 1
    replicate_frame = pd.DataFrame(columns)
    if row != n_rows: raise RuntimeError("Quantidade inesperada de linhas bootstrap")
    identity = replicate_frame.groupby(["regime", "seed"])[["threshold_attention", "threshold_set_transformer"]].nunique()
    if (identity.to_numpy() != 1).any(): raise RuntimeError("Threshold variou entre réplicas")
    summary: dict[str, Any] = {}
    for regime in config.regimes:
        summary[regime] = {"metrics": {}, "interpretation": {}}
        for metric in GENERIC_METRICS:
            sub = replicate_frame[(replicate_frame.regime == regime) & (replicate_frame.metric == metric) & replicate_frame.valid]
            campaign_values = sub.groupby("replicate_id", sort=False).delta.mean().to_numpy(float)
            att_mean = float(np.mean([observed[regime]["attention"][str(seed)][GENERIC_METRIC_KEYS[metric]] for seed in config.seeds]))
            set_mean = float(np.mean([observed[regime]["set_transformer"][str(seed)][GENERIC_METRIC_KEYS[metric]] for seed in config.seeds]))
            summary[regime]["metrics"][metric] = {"attention_mean": att_mean, "set_mean": set_mean, "observed_delta": set_mean - att_mean,
                                                   "bootstrap": _generic_summary(campaign_values, config.confidence_level, metric == "fpr")}
        f1 = summary[regime]["metrics"]["f1"]; precision = summary[regime]["metrics"]["precision"]; recall = summary[regime]["metrics"]["recall"]; mcc = summary[regime]["metrics"]["mcc"]; fpr = summary[regime]["metrics"]["fpr"]
        if f1["observed_delta"] > 0 and f1["bootstrap"]["ci_lower"] > 0: classification = "STRONG FAVORABLE"
        elif f1["observed_delta"] > 0: classification = "FAVORABLE BUT INCONCLUSIVE"
        else: classification = "NOT FAVORABLE"
        summary[regime]["interpretation"] = {"f1": classification, "precision": "Set higher" if precision["observed_delta"] > 0 else "Set lower/equal", "recall": "Set higher" if recall["observed_delta"] > 0 else "Set lower/equal", "mcc": "Set higher" if mcc["observed_delta"] > 0 else "Set lower/equal", "fpr": "Set lower" if fpr["observed_delta"] < 0 else "Set higher/equal"}
        summary[regime]["validity"] = {"requested_replicates": config.n_replicates, "valid_replicates": int(replicate_frame[(replicate_frame.regime == regime) & (replicate_frame.seed == config.seeds[0]) & (replicate_frame.metric == "f1")].valid.sum()), "invalid_replicates": int(replicate_frame[(replicate_frame.regime == regime) & (replicate_frame.seed == config.seeds[0]) & (replicate_frame.metric == "f1")].valid.eq(False).sum())}
    f1_status = [summary[r]["interpretation"]["f1"] for r in config.regimes]
    summary["main_interpretation"] = "YES" if all(s == "STRONG FAVORABLE" for s in f1_status) else "PARTIALLY" if any(s != "NOT FAVORABLE" for s in f1_status) else "NO"
    validity = {regime: summary[regime]["validity"] for regime in config.regimes}
    common_result = {"model_loaded": False, "tokenizer_loaded": False, "checkpoint_loaded": False, "cuda_initialized": False, "inference_executed": False, "training_executed": False}
    output_dir.parent.mkdir(parents=True, exist_ok=True); stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    try:
        replicate_frame.to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        per_seed.to_csv(stage / "per_seed_observed.csv", index=False)
        write_json(stage / "bootstrap_summary.json", summary); write_json(stage / "threshold_audit.json", threshold_audits); write_json(stage / "population_integrity.json", population)
        write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {**common_result, "inputs": input_hashes, "input_paths": input_paths, "campaign_audits": campaign_audits, "pairing": pairing, "population": population,
                     "n_bootstrap_requested": config.n_replicates, "validity": validity, "paired": True, "grouped": True, "group_key": "hearing_id", "preserve_group_multiplicity": True,
                     "same_samples_across_methods": True, "same_samples_across_seeds": True, "thresholds_reestimated": False, "thresholds_identical_all_replicates": True,
                     "publichearing_labels_used_for_threshold_selection": False, "delta": "set_transformer - attention", "analysis_seconds": time.time() - started}
        write_json(stage / "integrity_audit.json", integrity)
        report_population = {"examples": population["examples"], "positives": population["positives"], "prevalence": population["prevalence"], "hearings": population["hearings"], "exact example alignment": "YES", "exact label alignment": "YES", "exact hearing alignment": "YES", "probabilities valid": "YES"}
        _atomic_text(stage / "report.md", _generic_report(signature, config, report_population, threshold_audits, observed, summary, per_seed, validity))
        _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "bootstrap_executed": True, "seconds": time.time() - started, **common_result}, ensure_ascii=False) + "\n")
        manifest = {"schema_version": GENERIC_SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": input_hashes,
                    "counts": {"examples": population["examples"], "hearings": population["hearings"], "positives": population["positives"], "bootstrap_requested": config.n_replicates,
                               "bootstrap_valid": {regime: validity[regime]["valid_replicates"] for regime in config.regimes}}, "artifacts": {}}
        manifest["artifacts"] = _file_hashes(stage); write_json(stage / "manifest.json", manifest); os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
