
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
    # Include the frozen run's declared artifact map in provenance without paths/timestamps.
    audit["source_manifest_artifacts"] = transfer_manifest.get("artifacts", {})
    return baseline, lora, audit, hashes


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    # The exact shared decision operator.  Equality must be positive.
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
            # This is a mean of the three *paired effects* in each replica, not
            # an ensemble score and not an average of probabilities.
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
    # All decisions are fixed before resampling.  Nothing below can use a
    # PublicHearing label to adjust a threshold.
    base_predictions = {regime: base_scores >= baseline_thresholds[regime] for regime in REGIMES}
    lora_predictions = {seed: {regime: seed_scores[seed] >= lora_thresholds[seed][regime] for regime in REGIMES} for seed in config.seeds}
    rng = np.random.default_rng(config.seed)
    # A list of Python dictionaries for 480,000 rows can take several hundred
    # MiB.  Allocate typed columns once so this local, CPU-only analysis stays
    # lightweight while retaining one Parquet row per requested combination.
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
