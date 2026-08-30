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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .io_utils import sha256_file, write_json


SCHEMA_VERSION = "publichearing-paired-grouped-bootstrap-v1"
CODE_VERSION = "paired-grouped-bootstrap-v2"
EXPECTED_ROWS = 4235
EXPECTED_HEARINGS = 206
EXPECTED_POSITIVES = 501
EXPECTED_SEEDS = (0, 1, 2)
METRICS = ("auprc", "auroc", "brier_improvement")


def validate_generic_paired_frames(
    frames: dict[str, dict[int, pd.DataFrame]],
    *,
    conditions: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    expected_rows: int = EXPECTED_ROWS,
    expected_positives: int = EXPECTED_POSITIVES,
    expected_hearings: int = EXPECTED_HEARINGS,
    score_column: str = "probability",
) -> dict[str, Any]:
    names = tuple(conditions or frames)
    if not names or set(frames) != set(names):
        raise ValueError(f"Condições incompatíveis: esperado {names}, recebido {tuple(frames)}")
    reference: pd.DataFrame | None = None
    audit: dict[str, Any] = {}
    for condition in names:
        if set(frames[condition]) != set(seeds):
            raise ValueError(f"Seeds incompatíveis em {condition}")
        audit[condition] = {}
        for paired_seed in seeds:
            frame = frames[condition][paired_seed]
            required = {"example_id", "hearing_id", "label", score_column}
            if not required.issubset(frame.columns):
                raise ValueError(f"Colunas ausentes em {condition}/seed_{paired_seed}")
            current = frame[["example_id", "hearing_id", "label", score_column]].copy()
            current["example_id"] = current["example_id"].astype(str)
            current["hearing_id"] = current["hearing_id"].astype(str)
            if len(current) != expected_rows or not current.example_id.is_unique:
                raise ValueError(f"População/IDs inválidos em {condition}/seed_{paired_seed}")
            if current.hearing_id.nunique() != expected_hearings or current.isna().any().any():
                raise ValueError(f"Hearings/nulos inválidos em {condition}/seed_{paired_seed}")
            if not set(current.label.unique()).issubset({0, 1}) or set(current.label.unique()) != {0, 1} or int(current.label.sum()) != expected_positives:
                raise ValueError(f"Labels inválidos em {condition}/seed_{paired_seed}")
            scores = current[score_column].to_numpy(float)
            if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
                raise ValueError(f"Scores inválidos em {condition}/seed_{paired_seed}")
            indexed = current.set_index("example_id").sort_index()
            if reference is None:
                reference = indexed[["hearing_id", "label"]]
            else:
                if not reference.index.equals(indexed.index):
                    raise ValueError(f"example_id mismatch em {condition}/seed_{paired_seed}")
                if not reference["hearing_id"].equals(indexed["hearing_id"]):
                    raise ValueError(f"hearing_id mismatch em {condition}/seed_{paired_seed}")
                if not reference["label"].astype(int).equals(indexed["label"].astype(int)):
                    raise ValueError(f"label mismatch em {condition}/seed_{paired_seed}")
            audit[condition][str(paired_seed)] = {"examples": len(current), "hearings": int(current.hearing_id.nunique()), "positives": int(current.label.sum())}
    assert reference is not None
    return {"examples": expected_rows, "positives": expected_positives, "hearings": expected_hearings,
            "prevalence": expected_positives / expected_rows, "exact_example_alignment": True,
            "exact_hearing_alignment": True, "exact_label_alignment": True, "probabilities_valid": True,
            "per_condition": audit}


def _generic_summary(values: np.ndarray, confidence_level: float, favorable_positive: bool = True) -> dict[str, Any]:
    if len(values) == 0:
        return {"mean": None, "median": None, "ci_lower": None, "ci_upper": None, "p_delta_gt_zero": None,
                "p_delta_lt_zero": None, "probability_favorable": None, "favorable_direction": "higher" if favorable_positive else "lower", "n_valid": 0}
    alpha = 1.0 - confidence_level
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_lower": float(np.quantile(values, alpha / 2)), "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
            "p_delta_gt_zero": float(np.mean(values > 0)), "p_delta_lt_zero": float(np.mean(values < 0)),
            "probability_favorable": float(np.mean(values > 0 if favorable_positive else values < 0)),
            "favorable_direction": "higher" if favorable_positive else "lower", "n_valid": int(len(values))}


def _fast_metric_values(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_scores)]
    positives_by_score = np.add.reduceat(sorted_labels, starts)
    rows_by_score = ends - starts
    cumulative_positives = np.cumsum(positives_by_score)
    cumulative_rows = np.cumsum(rows_by_score)
    total_positive = int(labels.sum())
    total_negative = int(len(labels) - total_positive)
    average_precision = float(np.sum((cumulative_positives / cumulative_rows) * positives_by_score) / total_positive)
    average_ranks = (starts + 1 + ends) / 2.0
    u_statistic = float(np.sum(average_ranks * positives_by_score) - total_positive * (total_positive + 1) / 2)
    return {"auprc": average_precision, "auroc": 1.0 - u_statistic / (total_positive * total_negative),
            "brier": float(np.mean((scores - labels) ** 2))}


def run_generic_pairwise_bootstrap(
    frames: dict[str, dict[int, pd.DataFrame]],
    contrasts: dict[str, tuple[str, str]],
    *,
    interactions: dict[str, tuple[str, str]] | None = None,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    n_replicates: int = 10000,
    seed: int = 20260815,
    confidence_level: float = 0.95,
    metrics: tuple[str, ...] = ("auprc", "auroc", "brier"),
    expected_rows: int = EXPECTED_ROWS,
    expected_positives: int = EXPECTED_POSITIVES,
    expected_hearings: int = EXPECTED_HEARINGS,
    score_column: str = "probability",
) -> dict[str, Any]:
    allowed_metrics = {"auprc", "auroc", "brier"}
    if not metrics or not set(metrics).issubset(allowed_metrics) or n_replicates < 1:
        raise ValueError("Métricas ou número de réplicas inválido")
    interactions = interactions or {}
    for name, (left, right) in contrasts.items():
        if left not in frames or right not in frames or left == right:
            raise ValueError(f"Contraste inválido {name}: {left}, {right}")
    for name, (first, second) in interactions.items():
        if first not in contrasts or second not in contrasts:
            raise ValueError(f"Interaction inválida {name}")
    population = validate_generic_paired_frames(frames, conditions=tuple(frames), seeds=seeds,
                                                 expected_rows=expected_rows, expected_positives=expected_positives,
                                                 expected_hearings=expected_hearings, score_column=score_column)
    names = tuple(frames)
    reference = frames[names[0]][seeds[0]][["example_id", "hearing_id", "label"]].copy()
    reference["example_id"] = reference.example_id.astype(str)
    reference = reference.set_index("example_id").sort_index()
    labels = reference.label.to_numpy(int)
    group_values = reference.hearing_id.astype(str).to_numpy()
    groups = sorted(np.unique(group_values).tolist())
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    scores: dict[str, dict[int, np.ndarray]] = {}
    observed: dict[str, dict[str, dict[str, float]]] = {}
    for condition in names:
        scores[condition] = {}
        observed[condition] = {}
        for paired_seed in seeds:
            current = frames[condition][paired_seed].copy()
            current["example_id"] = current.example_id.astype(str)
            indexed = current.set_index("example_id").sort_index()
            values = indexed.loc[reference.index, score_column].to_numpy(float)
            scores[condition][paired_seed] = values
            observed[condition][str(paired_seed)] = {metric: _metric_value(metric, labels, values) for metric in metrics}
    observed_campaign = {name: {metric: float(np.mean([observed[right][str(s)][metric] - observed[left][str(s)][metric] for s in seeds]))
                                for metric in metrics} for name, (left, right) in contrasts.items()}
    observed_campaign.update({name: {metric: observed_campaign[first][metric] - observed_campaign[second][metric] for metric in metrics}
                              for name, (first, second) in interactions.items()})
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    all_contrasts = tuple(contrasts) + tuple(interactions)
    for replicate_id in range(n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([group_indices[position] for position in sampled_positions])
        sampled_labels = labels[indices]
        valid = np.unique(sampled_labels).size == 2
        reason = None if valid else "single_class_resample"
        replicate_values: dict[str, dict[str, float]] = {condition: {} for condition in names}
        if valid:
            for condition in names:
                seed_metrics = [_fast_metric_values(sampled_labels, scores[condition][s][indices]) for s in seeds]
                for metric in metrics:
                    replicate_values[condition][metric] = float(np.mean([item[metric] for item in seed_metrics]))
        for contrast in all_contrasts:
            for metric in metrics:
                if contrast in contrasts:
                    left, right = contrasts[contrast]
                    delta = replicate_values[right][metric] - replicate_values[left][metric] if valid else np.nan
                else:
                    first, second = interactions[contrast]
                    delta = ((replicate_values[contrasts[first][1]][metric] - replicate_values[contrasts[first][0]][metric]) -
                             (replicate_values[contrasts[second][1]][metric] - replicate_values[contrasts[second][0]][metric])) if valid else np.nan
                rows.append({"replicate_id": replicate_id, "contrast": contrast, "metric": metric, "delta": delta,
                             "valid": valid, "invalid_reason": reason, "n_rows": len(indices),
                             "n_sampled_groups": len(groups), "n_unique_groups": len(np.unique(sampled_positions)),
                             "n_positive": int(sampled_labels.sum()), "n_negative": int((sampled_labels == 0).sum())})
    replicate_frame = pd.DataFrame(rows)
    summaries: dict[str, dict[str, Any]] = {}
    for contrast in all_contrasts:
        summaries[contrast] = {}
        for metric in metrics:
            values = replicate_frame.loc[(replicate_frame.contrast == contrast) & (replicate_frame.metric == metric) & replicate_frame.valid, "delta"].to_numpy(float)
            favorable_positive = metric != "brier"
            summaries[contrast][metric] = _generic_summary(values, confidence_level, favorable_positive)
    return {"population": population, "observed": observed, "observed_campaign": observed_campaign, "summaries": summaries,
            "replicates": replicate_frame, "protocol": {"paired": True, "group_key": "hearing_id", "n_replicates": n_replicates,
                                                          "seed": seed, "confidence_level": confidence_level,
                                                          "same_samples_all_conditions": True, "same_samples_all_seeds": True,
                                                          "preserve_group_multiplicity": True, "campaign_statistic": "mean paired seed effects",
                                                          "metrics": list(metrics), "delta_orientation": "condition_b - condition_a"}}


@dataclass(frozen=True)
class BootstrapConfig:
    output_root: Path
    baseline_run: Path
    baseline_signature: str
    baseline_score_column: str
    confirmatory_run: Path
    confirmatory_signature: str
    seeds: tuple[int, ...]
    zero_shot_score_column: str
    n_replicates: int
    seed: int
    confidence_level: float
    ci_method: str
    preserve_group_multiplicity: bool
    same_samples_across_methods: bool
    same_samples_across_seeds: bool
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "BootstrapConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            return candidate.resolve()

        protocol = raw.get("protocol", {})
        inputs = raw.get("inputs", {})
        baseline = inputs.get("baseline_run", {})
        confirmatory = inputs.get("ragtruth_confirmatory_run", {})
        bootstrap = raw.get("bootstrap", {})
        metrics = tuple(str(metric) for metric in raw.get("metrics", ["auprc", "auroc", "brier_improvement"]))
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_paired_grouped_bootstrap")),
            baseline_run=resolve(baseline.get("path")),
            baseline_signature=str(baseline.get("expected_signature")),
            baseline_score_column=str(baseline.get("score_column", "hallucination_score")),
            confirmatory_run=resolve(confirmatory.get("path")),
            confirmatory_signature=str(confirmatory.get("expected_signature")),
            seeds=tuple(int(seed) for seed in confirmatory.get("seeds", [0, 1, 2])),
            zero_shot_score_column=str(confirmatory.get("score_column", "probability")),
            n_replicates=int(bootstrap.get("n_replicates", 10000)),
            seed=int(bootstrap.get("seed", 20260806)),
            confidence_level=float(bootstrap.get("confidence_level", 0.95)),
            ci_method=str(bootstrap.get("ci_method", "percentile")),
            preserve_group_multiplicity=bool(bootstrap.get("preserve_group_multiplicity", True)),
            same_samples_across_methods=bool(bootstrap.get("same_samples_across_methods", True)),
            same_samples_across_seeds=bool(bootstrap.get("same_samples_across_seeds", True)),
            metrics=metrics,
        )
        config.validate(protocol)
        return config

    def validate(self, protocol: dict[str, Any] | None = None) -> None:
        protocol = protocol or {}
        if protocol.get("analysis_type") not in {None, "paired_grouped_bootstrap"}:
            raise ValueError("analysis_type deve ser paired_grouped_bootstrap")
        if protocol.get("group_key", "hearing_id") != "hearing_id" or protocol.get("paired", True) is not True:
            raise ValueError("O protocolo deve ser pareado e agrupado por hearing_id")
        if self.baseline_signature != "54d9c623f8685c39" or self.confirmatory_signature != "4e12933c51136624":
            raise ValueError("Assinaturas científicas de entrada não correspondem aos runs congelados")
        if self.seeds != EXPECTED_SEEDS:
            raise ValueError("A análise exige exatamente as seeds [0, 1, 2]")
        if self.baseline_score_column != "hallucination_score" or self.zero_shot_score_column != "probability":
            raise ValueError("Colunas de score incompatíveis com o protocolo")
        if self.n_replicates < 1 or self.seed < 0 or not 0 < self.confidence_level < 1 or self.ci_method != "percentile":
            raise ValueError("Configuração de bootstrap inválida")
        if tuple(self.metrics) != METRICS:
            raise ValueError(f"Métricas esperadas: {METRICS}")
        if not self.preserve_group_multiplicity or not self.same_samples_across_methods or not self.same_samples_across_seeds:
            raise ValueError("O protocolo exige multiplicidade e amostras compartilhadas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {"name": "publichearing_ragtruth_vs_off_the_shelf_paired_bootstrap", "analysis_type": "paired_grouped_bootstrap", "group_key": "hearing_id", "paired": True},
            "inputs": {
                "baseline_run": {"path": str(self.baseline_run), "score_column": self.baseline_score_column, "expected_signature": self.baseline_signature},
                "ragtruth_confirmatory_run": {"path": str(self.confirmatory_run), "seeds": list(self.seeds), "score_column": self.zero_shot_score_column, "expected_signature": self.confirmatory_signature},
            },
            "bootstrap": {"n_replicates": self.n_replicates, "seed": self.seed, "confidence_level": self.confidence_level, "ci_method": self.ci_method, "preserve_group_multiplicity": self.preserve_group_multiplicity, "same_samples_across_methods": self.same_samples_across_methods, "same_samples_across_seeds": self.same_samples_across_seeds},
            "metrics": list(self.metrics),
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
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}


def _validate_manifest(path: Path, expected_signature: str) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("signature")) != expected_signature:
        raise ValueError(f"Assinatura divergente em {manifest_path}")
    if manifest.get("status") not in {"completed", "externally_evaluated"}:
        raise ValueError(f"Status inválido em {manifest_path}")
    for name, expected_hash in manifest.get("artifacts", {}).items():
        artifact = path / name
        if artifact.is_file() and sha256_file(artifact) != expected_hash:
            raise ValueError(f"Hash divergente no input {artifact}")
    return manifest


def _locate_seed_prediction(config: BootstrapConfig, seed: int) -> tuple[Path, dict[str, Any], str]:
    seed_root = config.confirmatory_run / f"seed_{seed}" / "publichearing_zero_shot"
    manifests = sorted(seed_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Esperava um manifesto zero-shot para seed {seed}, encontrei {len(manifests)}")
    manifest_path = manifests[0]
    run_dir = manifest_path.parent
    manifest = _validate_manifest(run_dir, str(manifest_path.parent.name))
    prediction = run_dir / "predictions.parquet"
    if not prediction.is_file():
        raise FileNotFoundError(f"Predição ausente: {prediction}")
    expected_hash = manifest.get("artifacts", {}).get("predictions.parquet")
    actual_hash = sha256_file(prediction)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Hash da predição divergente: {prediction}")
    return prediction, manifest, actual_hash


def _validate_prediction(path: Path, score_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"example_id", "hearing_id", "label", score_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} não contém {sorted(required)}")
    if len(frame) != EXPECTED_ROWS or not frame["example_id"].is_unique:
        raise ValueError(f"Contagem/IDs inválidos em {path}")
    if frame["hearing_id"].nunique() != EXPECTED_HEARINGS or frame["label"].isna().any() or not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Grupos/labels inválidos em {path}")
    if int(frame["label"].sum()) != EXPECTED_POSITIVES or frame[["example_id", "hearing_id", "label", score_column]].isna().any().any():
        raise ValueError(f"Contagem de positivos/valores nulos inválida em {path}")
    score = frame[score_column].to_numpy(dtype=float)
    if not np.isfinite(score).all() or not ((score >= 0).all() and (score <= 1).all()):
        raise ValueError(f"Score inválido em {path}")
    return frame


def _strict_join(baseline: pd.DataFrame, seed_frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = baseline[["example_id", "hearing_id", "label", "hallucination_score"]].copy()
    base["example_id"] = base["example_id"].astype(str)
    if not base["example_id"].is_unique:
        raise ValueError("example_id duplicado no baseline")
    audit: dict[str, Any] = {}
    for seed, frame in seed_frames.items():
        current = frame[["example_id", "hearing_id", "label", "probability"]].copy()
        current["example_id"] = current["example_id"].astype(str)
        if not current["example_id"].is_unique:
            raise ValueError(f"example_id duplicado na seed {seed}")
        base_ids, seed_ids = set(base.example_id), set(current.example_id)
        common = base_ids & seed_ids
        if base_ids - seed_ids or seed_ids - base_ids:
            raise ValueError(f"IDs não pareáveis na seed {seed}")
        left = base.set_index("example_id").loc[sorted(common)]
        right = current.set_index("example_id").loc[sorted(common)]
        label_mismatches = int((left["label"].astype(int) != right["label"].astype(int)).sum())
        hearing_mismatches = int((left["hearing_id"].astype(str) != right["hearing_id"].astype(str)).sum())
        audit[str(seed)] = {"baseline_only_ids": len(base_ids - seed_ids), "zero_shot_only_ids": len(seed_ids - base_ids), "label_mismatches": label_mismatches, "hearing_id_mismatches": hearing_mismatches, "duplicate_ids": int((not base.example_id.is_unique) or (not current.example_id.is_unique)), "pairable_examples": len(common)}
        if label_mismatches or hearing_mismatches:
            raise ValueError(f"Labels/hearing_id divergentes na seed {seed}")
    joined = base.sort_values("example_id").reset_index(drop=True)
    for seed, frame in seed_frames.items():
        current = frame[["example_id", "probability"]].copy()
        current["example_id"] = current["example_id"].astype(str)
        joined = joined.merge(current, on="example_id", how="left", validate="one_to_one").rename(columns={"probability": f"seed_{seed}"})
    if joined[[f"seed_{seed}" for seed in EXPECTED_SEEDS]].isna().any().any():
        raise ValueError("Join deixou valores ausentes")
    return joined, audit


def _observed_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {"auprc": float(average_precision_score(labels, scores)), "auroc": float(roc_auc_score(labels, scores)), "brier": float(brier_score_loss(labels, scores))}


def _metric_value(metric: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "brier":
        return float(brier_score_loss(labels, scores))
    raise ValueError(metric)


def _effect(metric: str, baseline: float, model: float) -> float:
    return float(model - baseline) if metric in {"auprc", "auroc"} else float(baseline - model)


def _input_signature(config: BootstrapConfig, input_hashes: dict[str, str], joined: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": SCHEMA_VERSION, "code_version": CODE_VERSION,
        "baseline_signature": config.baseline_signature, "confirmatory_signature": config.confirmatory_signature,
        "input_hashes": input_hashes, "seeds": list(config.seeds),
        "example_ids_hash": _canonical_hash(joined.example_id.astype(str).tolist()),
        "hearing_ids_hash": _canonical_hash(sorted(joined.hearing_id.astype(str).unique().tolist())),
        "score_columns": {"baseline": config.baseline_score_column, "zero_shot": config.zero_shot_score_column},
        "metrics": list(config.metrics), "effect_formulas": {"auprc": "zero_shot - baseline", "auroc": "zero_shot - baseline", "brier_improvement": "baseline - zero_shot"},
        "n_replicates": config.n_replicates, "bootstrap_seed": config.seed, "confidence_level": config.confidence_level, "ci_method": config.ci_method,
        "group_key": "hearing_id", "preserve_group_multiplicity": config.preserve_group_multiplicity, "same_samples_across_methods": config.same_samples_across_methods, "same_samples_across_seeds": config.same_samples_across_seeds,
    }
    return _canonical_hash(payload)[:16], payload


def _summary(values: np.ndarray, confidence_level: float, favorable: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"mean": None, "std": None, "median": None, "ci_lower": None, "ci_upper": None, "percentile_2_5": None, "percentile_97_5": None, "bootstrap_support_probability": None, "n_valid": 0}
    alpha = 1.0 - confidence_level
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "median": float(np.median(values)), "ci_lower": float(np.quantile(values, alpha / 2)), "ci_upper": float(np.quantile(values, 1 - alpha / 2)), "percentile_2_5": float(np.quantile(values, 0.025)), "percentile_97_5": float(np.quantile(values, 0.975)), "bootstrap_support_probability": float(favorable.mean()), "n_valid": int(len(values))}


def _build_summaries(
    replicate_frame: pd.DataFrame,
    config: BootstrapConfig,
    observed: dict[str, dict[str, float]],
    observed_effects: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any], int, int, dict[str, int]]:
    reference_rows = replicate_frame[(replicate_frame["seed"] == config.seeds[0]) & (replicate_frame["metric"] == config.metrics[0])]
    valid_count = int(reference_rows["valid"].sum())
    invalid_count = config.n_replicates - valid_count
    invalid_reasons = {str(key): int(value) for key, value in reference_rows.loc[~reference_rows["valid"], "invalid_reason"].value_counts().items()}
    per_seed: dict[str, Any] = {}
    for seed in config.seeds:
        bootstrap: dict[str, Any] = {}
        for metric in config.metrics:
            values = replicate_frame.loc[(replicate_frame["seed"] == seed) & (replicate_frame["metric"] == metric) & replicate_frame["valid"], "effect"].dropna().to_numpy(dtype=float)
            bootstrap[metric] = _summary(values, config.confidence_level, values > 0)
        per_seed[str(seed)] = {
            "observed_metrics": {"baseline": observed["baseline"], "zero_shot": observed[f"seed_{seed}"]},
            "observed_effects": observed_effects[str(seed)],
            "bootstrap": bootstrap,
            "n_bootstrap_invalid": invalid_count,
        }
    campaign_bootstrap: dict[str, Any] = {}
    for metric in config.metrics:
        values = replicate_frame.loc[(replicate_frame["seed"] == -1) & (replicate_frame["metric"] == metric) & replicate_frame["valid"], "effect"].dropna().to_numpy(dtype=float)
        campaign_bootstrap[metric] = _summary(values, config.confidence_level, values > 0)
    campaign = {
        "observed_effects": {metric: float(np.mean([observed_effects[str(seed)][metric] for seed in config.seeds])) for metric in config.metrics},
        "observed_effect_min": {metric: float(min(observed_effects[str(seed)][metric] for seed in config.seeds)) for metric in config.metrics},
        "observed_effect_max": {metric: float(max(observed_effects[str(seed)][metric] for seed in config.seeds)) for metric in config.metrics},
        "bootstrap": campaign_bootstrap,
        "ensemble_used": False,
        "best_seed_selected": False,
        "same_bootstrap_samples_across_seeds": True,
    }
    return per_seed, campaign, valid_count, invalid_count, invalid_reasons


def _report(config: BootstrapConfig, signature: str, observed: dict[str, Any], per_seed: dict[str, Any], campaign: dict[str, Any], valid: int, invalid: int, seconds: float) -> str:
    lines = ["# Bootstrap pareado agrupado por hearing_id", "", f"- Analysis signature: `{signature}`", f"- Réplicas solicitadas: {config.n_replicates}; válidas: {valid}; inválidas: {invalid}", "- Unidade de reamostragem: `hearing_id`, com reposição e preservação da multiplicidade.", "- Mesmas amostras para baseline e as três seeds; nenhum ensemble e nenhuma seleção de seed.", "", "## Métricas observadas", ""]
    for name, values in observed.items():
        lines.append(f"- {name}: " + ", ".join(f"{key}={value:.6f}" for key, value in values.items()))
    lines += ["", "## Efeitos bootstrap", ""]
    for seed, result in per_seed.items():
        lines.append(f"### Seed {seed}")
        for metric, summary in result["bootstrap"].items():
            lines.append(f"- {metric}: efeito observado={result['observed_effects'][metric]:.6f}; IC95%=[{summary['ci_lower']:.6f}, {summary['ci_upper']:.6f}]; suporte={summary['bootstrap_support_probability']:.6f}")
    lines += ["", "## Efeito médio entre seeds", ""]
    for metric, summary in campaign["bootstrap"].items():
        lines.append(f"- {metric}: efeito observado={campaign['observed_effects'][metric]:.6f}; IC95%=[{summary['ci_lower']:.6f}, {summary['ci_upper']:.6f}]; suporte={summary['bootstrap_support_probability']:.6f}")
    lines += ["", f"Tempo de execução: {seconds:.3f} s.", "A análise não estabelece causalidade nem generalização fora deste dataset/protocolo. Não foram usados thresholds, ensemble ou os valores históricos não auditáveis."]
    return "\n".join(lines) + "\n"


def _verify_completed(output_dir: Path, signature: str) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("status") != "completed":
        raise ValueError("Manifesto de análise incompatível")
    for name, expected in manifest.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artefato ausente/corrompido: {path}")
    return manifest


def _resume_from_complete_replicates(
    output_dir: Path,
    config: BootstrapConfig,
    signature: str,
    signature_payload: dict[str, Any],
    observed: dict[str, dict[str, float]],
    observed_effects: dict[str, dict[str, float]],
    input_hashes: dict[str, str],
    pair_audit: dict[str, Any],
    n_examples: int,
    n_hearings: int,
    positives: int,
) -> dict[str, Any]:
    path = output_dir / "bootstrap_replicates.parquet"
    frame = pd.read_parquet(path)
    required = {"replicate_id", "bootstrap_seed", "seed", "metric", "baseline_value", "zero_shot_value", "effect", "valid", "invalid_reason", "n_rows", "n_sampled_groups", "n_unique_groups", "n_positive", "n_negative"}
    if not required.issubset(frame.columns):
        raise ValueError("bootstrap_replicates.parquet incompleto para resume")
    expected_rows = config.n_replicates * (len(config.seeds) + 1) * len(config.metrics)
    if len(frame) != expected_rows or frame.duplicated(["replicate_id", "seed", "metric"]).any():
        raise ValueError("Réplica parcial ou duplicada; não há estado RNG auditável para retomar")
    if set(frame["replicate_id"].astype(int)) != set(range(config.n_replicates)) or set(frame["bootstrap_seed"].astype(int)) != {config.seed}:
        raise ValueError("IDs/seed das réplicas incompatíveis")
    per_seed, campaign, valid_count, invalid_count, invalid_reasons = _build_summaries(frame, config, observed, observed_effects)
    _atomic_json(output_dir / "observed_metrics.json", {**observed, "effects": observed_effects})
    _atomic_json(output_dir / "per_seed_results.json", per_seed)
    _atomic_json(output_dir / "campaign_summary.json", campaign)
    _atomic_json(output_dir / "resolved_config.json", config.to_dict())
    integrity = {"inputs": input_hashes, "pairing": pair_audit, "n_examples": n_examples, "n_hearings": n_hearings, "n_bootstrap_requested": config.n_replicates, "n_bootstrap_valid": valid_count, "n_bootstrap_invalid": invalid_count, "invalid_reasons": invalid_reasons, "group_multiplicity_preserved": True, "same_indices_across_methods": True, "same_indices_across_seeds": True, "ensemble_used": False, "best_seed_selected": False, "models_loaded": False, "tokenizers_loaded": False, "cuda_initialized": False, "resumed_from_complete_replicates": True, "bootstrap_reexecuted": False}
    _atomic_json(output_dir / "integrity_audit.json", integrity)
    _atomic_text(output_dir / "run_log.jsonl", json.dumps({"event": "derived_artifacts_resumed", "bootstrap_executed": False}, ensure_ascii=False) + "\n")
    manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": {"baseline_signature": config.baseline_signature, "confirmatory_signature": config.confirmatory_signature, "hashes": input_hashes}, "counts": {"examples": n_examples, "hearings": n_hearings, "positives": positives, "negatives": n_examples - positives, "bootstrap_requested": config.n_replicates, "bootstrap_valid": valid_count, "bootstrap_invalid": invalid_count}, "artifacts": {}}
    _atomic_text(output_dir / "report.md", _report(config, signature, observed, per_seed, campaign, valid_count, invalid_count, 0.0))
    manifest["artifacts"] = _file_hashes(output_dir)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def run_bootstrap(config: BootstrapConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    baseline_dir = config.baseline_run
    baseline_manifest = _validate_manifest(baseline_dir, config.baseline_signature)
    baseline_path = baseline_dir / "predictions.parquet"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    input_hashes = {"baseline_predictions.parquet": sha256_file(baseline_path)}
    baseline = _validate_prediction(baseline_path, config.baseline_score_column)
    _validate_manifest(config.confirmatory_run, config.confirmatory_signature)
    seed_frames: dict[int, pd.DataFrame] = {}
    seed_paths: dict[str, str] = {}
    for seed in config.seeds:
        path, _, digest = _locate_seed_prediction(config, seed)
        seed_frames[seed] = _validate_prediction(path, config.zero_shot_score_column)
        seed_paths[str(seed)] = str(path)
        input_hashes[f"seed_{seed}_predictions.parquet"] = digest
    joined, pair_audit = _strict_join(baseline, seed_frames)
    signature, signature_payload = _input_signature(config, input_hashes, joined)
    labels = joined.label.to_numpy(dtype=int)
    observed: dict[str, dict[str, float]] = {"baseline": _observed_metrics(labels, joined["hallucination_score"].to_numpy(dtype=float))}
    for seed in config.seeds:
        observed[f"seed_{seed}"] = _observed_metrics(labels, joined[f"seed_{seed}"].to_numpy(dtype=float))
    observed_effects: dict[str, dict[str, float]] = {}
    for seed in config.seeds:
        observed_effects[str(seed)] = {metric: _effect(metric, observed["baseline"]["brier" if metric == "brier_improvement" else metric], observed[f"seed_{seed}"]["brier" if metric == "brier_improvement" else metric]) for metric in config.metrics}
    output_dir = (config.output_root / signature).resolve()
    if validate_only:
        return {"status": "valid", "analysis_executed": False, "bootstrap_executed": False, "model_loaded": False, "cuda_initialized": False, "n_examples": len(joined), "n_hearings": int(joined.hearing_id.nunique()), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "seeds": list(config.seeds), "paired_examples_per_seed": {str(seed): pair_audit[str(seed)]["pairable_examples"] for seed in config.seeds}, "n_bootstrap": config.n_replicates, "group_key": "hearing_id", "preserve_group_multiplicity": config.preserve_group_multiplicity, "same_samples_across_methods": config.same_samples_across_methods, "same_samples_across_seeds": config.same_samples_across_seeds, "analysis_signature": signature, "output_dir": str(output_dir), "input_hashes": input_hashes, "pairing": pair_audit}
    if output_dir.exists():
        if resume and (output_dir / "manifest.json").is_file():
            return _verify_completed(output_dir, signature)
        if resume and (output_dir / "bootstrap_replicates.parquet").is_file():
            return _resume_from_complete_replicates(output_dir, config, signature, signature_payload, observed, observed_effects, input_hashes, pair_audit, len(joined), int(joined.hearing_id.nunique()), int(labels.sum()))
        raise FileExistsError(f"Saída já existe; use --resume: {output_dir}")
    group_values = joined["hearing_id"].astype(str).to_numpy()
    groups = sorted(np.unique(group_values).tolist())
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    for replicate_id in range(config.n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        sampled_indices = np.concatenate([group_indices[position] for position in sampled_positions])
        sampled_labels = labels[sampled_indices]
        valid = np.unique(sampled_labels).size == 2
        reason = None if valid else "single_class_resample"
        common = {"replicate_id": replicate_id, "bootstrap_seed": config.seed, "valid": bool(valid), "invalid_reason": reason, "n_rows": int(len(sampled_indices)), "n_sampled_groups": len(groups), "n_unique_groups": int(len(set(sampled_positions.tolist()))), "n_positive": int(sampled_labels.sum()), "n_negative": int((sampled_labels == 0).sum())}
        values: dict[str, float] = {}
        if valid:
            base_values = {"auprc": _metric_value("auprc", sampled_labels, joined["hallucination_score"].to_numpy(dtype=float)[sampled_indices]), "auroc": _metric_value("auroc", sampled_labels, joined["hallucination_score"].to_numpy(dtype=float)[sampled_indices]), "brier": _metric_value("brier", sampled_labels, joined["hallucination_score"].to_numpy(dtype=float)[sampled_indices])}
            for seed in config.seeds:
                model_scores = joined[f"seed_{seed}"].to_numpy(dtype=float)[sampled_indices]
                model_values = {"auprc": _metric_value("auprc", sampled_labels, model_scores), "auroc": _metric_value("auroc", sampled_labels, model_scores), "brier": _metric_value("brier", sampled_labels, model_scores)}
                for metric in config.metrics:
                    effect = _effect(metric, base_values["brier" if metric == "brier_improvement" else metric], model_values["brier" if metric == "brier_improvement" else metric])
                    records.append({**common, "seed": seed, "seed_label": f"seed_{seed}", "metric": metric, "baseline_value": base_values["brier" if metric == "brier_improvement" else metric], "zero_shot_value": model_values["brier" if metric == "brier_improvement" else metric], "effect": effect})
                    values.setdefault(metric, 0.0); values[metric] += effect / len(config.seeds)
                
            for metric in config.metrics:
                records.append({**common, "seed": -1, "seed_label": "mean_seed_effect", "metric": metric, "baseline_value": base_values["brier" if metric == "brier_improvement" else metric], "zero_shot_value": None, "effect": values[metric]})
        else:
            for seed in config.seeds:
                for metric in config.metrics:
                    records.append({**common, "seed": seed, "seed_label": f"seed_{seed}", "metric": metric, "baseline_value": None, "zero_shot_value": None, "effect": None})
            for metric in config.metrics:
                records.append({**common, "seed": -1, "seed_label": "mean_seed_effect", "metric": metric, "baseline_value": None, "zero_shot_value": None, "effect": None})
    replicate_frame = pd.DataFrame.from_records(records)
    per_seed, campaign, valid_count, invalid_count, invalid_reasons = _build_summaries(replicate_frame, config, observed, observed_effects)
    input_hashes_after = {"baseline_predictions.parquet": sha256_file(baseline_path), **{f"seed_{seed}_predictions.parquet": sha256_file(Path(seed_paths[str(seed)])) for seed in config.seeds}}
    if input_hashes_after != input_hashes:
        raise RuntimeError("Um dos Parquets de entrada foi alterado durante a análise")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        replicate_frame.to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        write_json(stage / "observed_metrics.json", {**observed, "effects": observed_effects})
        write_json(stage / "per_seed_results.json", per_seed)
        write_json(stage / "campaign_summary.json", campaign)
        write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {"inputs": input_hashes, "input_paths": {"baseline": str(baseline_path), "seeds": seed_paths}, "pairing": pair_audit, "n_examples": len(joined), "n_hearings": len(groups), "n_bootstrap_requested": config.n_replicates, "n_bootstrap_valid": valid_count, "n_bootstrap_invalid": invalid_count, "invalid_reasons": invalid_reasons, "group_multiplicity_preserved": True, "same_indices_across_methods": True, "same_indices_across_seeds": True, "ensemble_used": False, "best_seed_selected": False, "models_loaded": False, "tokenizers_loaded": False, "cuda_initialized": False, "inputs_reloaded_hashes": input_hashes_after}
        write_json(stage / "integrity_audit.json", integrity)
        _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "seconds": time.time() - started, "models_loaded": False, "bootstrap_seed": config.seed}, ensure_ascii=False) + "\n")
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": {"baseline_signature": config.baseline_signature, "confirmatory_signature": config.confirmatory_signature, "hashes": input_hashes}, "counts": {"examples": len(joined), "hearings": len(groups), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "bootstrap_requested": config.n_replicates, "bootstrap_valid": valid_count, "bootstrap_invalid": invalid_count}, "artifacts": {}}
        _atomic_text(stage / "report.md", _report(config, signature, observed, per_seed, campaign, valid_count, invalid_count, time.time() - started))
        manifest["artifacts"] = _file_hashes(stage)
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
