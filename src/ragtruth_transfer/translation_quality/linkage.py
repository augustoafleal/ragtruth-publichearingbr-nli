from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..io_utils import sha256_file
from ..metrics import binary_metrics
from .config import LinkConfig
from .storage import write_csv_atomic, write_json_atomic, write_parquet_atomic

MANIFEST_SCHEMA = "ragtruth-translation-quality-link-manifest-v1"
_EPS = 1e-7

CORRELATION_SIGNALS = (
    "cometkiwi_claim",
    "cometkiwi_chunk_mean",
    "cometkiwi_chunk_min",
    "nli_abs_delta_mean",
    "nli_abs_delta_max",
    "any_truncation_introduced",
    "any_exclusion_candidate",
)


class LinkValidationError(RuntimeError):
    pass


def load_protocol_threshold(config: LinkConfig) -> tuple[float, dict[str, Any]]:
    path = config.thresholds_json
    if not path.is_file():
        raise FileNotFoundError(f"threshold do protocolo não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if config.threshold_criterion not in payload:
        raise ValueError(
            f"critério {config.threshold_criterion!r} ausente em {path}; "
            f"disponíveis: {sorted(payload)}"
        )
    record = payload[config.threshold_criterion]
    if "threshold" not in record:
        raise ValueError(f"registro de threshold sem campo 'threshold' em {path}")
    return float(record["threshold"]), record


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise LinkValidationError(f"{label} não contém as colunas obrigatórias: {missing}")


def _assert_unique(series: pd.Series, label: str) -> None:
    duplicated = series[series.duplicated()].unique()
    if len(duplicated):
        raise LinkValidationError(
            f"{label} possui example_id duplicado (exemplos: {list(duplicated[:5])})."
        )


def _read_confirmatory_metadata(predictions_parquet: Path) -> dict[str, Any] | None:
    try:
        protocol_dir = predictions_parquet.parents[2]
    except IndexError:
        return None
    for name in ("frozen_protocol.json", "manifest.json"):
        candidate = protocol_dir / name
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def _validate_backend(config: LinkConfig, scores_manifest: dict[str, Any] | None) -> None:
    predictions_path = config.predictions_parquet
    metadata = _read_confirmatory_metadata(predictions_path)
    if metadata is not None:
        protocol_signature = metadata.get("signature")
        payload = metadata.get("payload") if isinstance(metadata.get("payload"), dict) else {}
        protocol_name = payload.get("protocol_name") or metadata.get("protocol_name")
        if config.expected_protocol_signature:
            if protocol_signature is None:
                raise LinkValidationError(
                    "metadados confirmatórios sem assinatura; não é possível validar o protocolo."
                )
            if protocol_signature != config.expected_protocol_signature:
                raise LinkValidationError(
                    f"assinatura do protocolo ({protocol_signature!r}) difere da esperada "
                    f"({config.expected_protocol_signature!r})."
                )
        if config.backend and protocol_name is not None:
            if config.backend.lower() not in str(protocol_name).lower():
                raise LinkValidationError(
                    f"backend {config.backend!r} não corresponde ao protocolo {protocol_name!r}."
                )
    else:
        predictions_text = str(predictions_path)
        if (
            config.expected_protocol_signature
            and config.expected_protocol_signature not in predictions_text
        ):
            raise LinkValidationError(
                f"assinatura de protocolo esperada {config.expected_protocol_signature!r} "
                f"não está no caminho de predictions: {predictions_text}"
            )
        if config.backend and config.backend.lower() not in predictions_text.lower():
            raise LinkValidationError(
                f"backend {config.backend!r} não identificado no caminho de predictions: {predictions_text}"
            )
    if scores_manifest is not None:
        manifest_backend = scores_manifest.get("backend")
        if manifest_backend is not None and manifest_backend != config.backend:
            raise LinkValidationError(
                f"backend das scores ({manifest_backend!r}) difere do configurado ({config.backend!r})."
            )


def _read_scores_manifest(config: LinkConfig) -> dict[str, Any] | None:
    manifest_path = config.example_scores_parquet.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_and_join(config: LinkConfig) -> pd.DataFrame:
    scores = pd.read_parquet(config.example_scores_parquet)
    preds = pd.read_parquet(config.predictions_parquet)

    id_col = config.example_id_column
    _require_columns(scores, [id_col, config.source_id_column, config.label_column], "scores")
    _require_columns(
        preds,
        [id_col, config.score_column, config.source_id_column, config.label_column],
        "predictions",
    )
    if config.quality_signal not in scores.columns:
        raise LinkValidationError(
            f"sinal de qualidade {config.quality_signal!r} ausente em scores; "
            f"disponíveis: {sorted(c for c in scores.columns if c != id_col)}"
        )

    _assert_unique(scores[id_col].astype(str), "scores")
    _assert_unique(preds[id_col].astype(str), "predictions")

    _validate_backend(config, _read_scores_manifest(config))

    score_ids = set(scores[id_col].astype(str))
    pred_ids = set(preds[id_col].astype(str))
    missing = pred_ids - score_ids
    if missing:
        raise LinkValidationError(
            f"{len(missing)} example_id de predictions não estão presentes em scores "
            f"(exemplos: {sorted(missing)[:5]}). Não é permitido join com perda silenciosa."
        )

    preds_small = preds[[id_col, config.score_column]].copy()
    preds_small["__label_pred"] = preds[config.label_column]
    preds_small["__source_id_pred"] = preds[config.source_id_column]

    merged = scores.merge(preds_small, on=id_col, how="inner", validate="one_to_one")
    if len(merged) != len(preds):
        raise LinkValidationError(
            f"join resultou em {len(merged)} linhas, mas predictions tem {len(preds)}."
        )

    label_mismatch = merged[
        merged[config.label_column].astype(str) != merged["__label_pred"].astype(str)
    ]
    if len(label_mismatch):
        raise LinkValidationError(
            f"{len(label_mismatch)} exemplos com label divergente entre scores e predictions "
            f"(exemplos: {list(label_mismatch[id_col].astype(str)[:5])})."
        )

    source_mismatch = merged[
        merged[config.source_id_column].astype(str) != merged["__source_id_pred"].astype(str)
    ]
    if len(source_mismatch):
        raise LinkValidationError(
            f"{len(source_mismatch)} exemplos com source_id divergente entre scores e predictions "
            f"(exemplos: {list(source_mismatch[id_col].astype(str)[:5])})."
        )

    return merged.drop(columns=["__label_pred", "__source_id_pred"])


def add_error_columns(
    frame: pd.DataFrame, config: LinkConfig, threshold: float
) -> pd.DataFrame:
    frame = frame.copy()
    y = frame[config.label_column].astype(int).to_numpy()
    p = frame[config.score_column].astype(float).clip(_EPS, 1 - _EPS).to_numpy()
    pred = (frame[config.score_column].astype(float).to_numpy() >= threshold).astype(int)
    frame["pred_at_threshold"] = pred
    frame["abs_error"] = np.abs(p - y)
    frame["log_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    frame["correct"] = (pred == y).astype(int)
    return frame


def _quality_axis(frame: pd.DataFrame, config: LinkConfig) -> pd.Series:
    signal = frame[config.quality_signal].astype(float)
    return signal if config.quality_direction == "higher_is_better" else -signal


def _bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    values: list[float] = []
    for _ in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        if metric == "AUPRC":
            values.append(float(average_precision_score(yb, sb)))
        elif metric == "AUROC":
            values.append(float(roc_auc_score(yb, sb)))
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def stratified_analysis(frame: pd.DataFrame, config: LinkConfig) -> pd.DataFrame:
    valid = frame[frame[config.quality_signal].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["quality_axis"] = _quality_axis(valid, config)
    try:
        valid["quality_bucket"] = pd.qcut(
            valid["quality_axis"],
            q=config.num_quality_buckets,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        valid["quality_bucket"] = 0

    rows: list[dict[str, Any]] = []
    for bucket, group in valid.groupby("quality_bucket", sort=True):
        y = group[config.label_column].astype(int).to_numpy()
        scores = group[config.score_column].astype(float).to_numpy()
        pred = group["pred_at_threshold"].astype(int).to_numpy()
        record: dict[str, Any] = {
            "quality_bucket": int(bucket),
            "n": int(len(group)),
            "quality_signal_min": float(group[config.quality_signal].min()),
            "quality_signal_max": float(group[config.quality_signal].max()),
            "positives": int(y.sum()),
        }
        if len(np.unique(y)) >= 2:
            point = binary_metrics(y, pred, scores)
            record.update({k: point[k] for k in ("AUPRC", "AUROC", "F1", "MCC", "FPR")})
            groups = group[config.source_id_column].astype(str).to_numpy()
            lo, hi = _bootstrap_ci(
                y, scores, groups, "AUPRC", config.bootstrap_samples, config.bootstrap_seed
            )
            record["AUPRC_ci_low"] = lo
            record["AUPRC_ci_high"] = hi
        rows.append(record)
    return pd.DataFrame(rows)


def filtered_curve(frame: pd.DataFrame, config: LinkConfig, steps: int = 20) -> pd.DataFrame:
    valid = frame[frame[config.quality_signal].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["quality_axis"] = _quality_axis(valid, config)
    valid = valid.sort_values("quality_axis")
    quality = valid["quality_axis"].to_numpy()
    y_all = valid[config.label_column].astype(int).to_numpy()
    s_all = valid[config.score_column].astype(float).to_numpy()

    rows: list[dict[str, Any]] = []
    for frac in np.linspace(0.0, 0.9, steps + 1):
        cut = float(np.quantile(quality, frac)) if frac > 0 else float(quality.min()) - 1.0
        keep = quality >= cut
        y, s = y_all[keep], s_all[keep]
        if len(np.unique(y)) < 2:
            continue
        rows.append(
            {
                "dropped_fraction": float(frac),
                "quality_axis_threshold": cut,
                "n_kept": int(keep.sum()),
                "AUPRC": float(average_precision_score(y, s)),
                "AUROC": float(roc_auc_score(y, s)),
            }
        )
    return pd.DataFrame(rows)


def correlation_and_regression(frame: pd.DataFrame, config: LinkConfig) -> dict[str, Any]:
    valid = frame.copy()
    valid["quality_axis"] = _quality_axis(valid, config)
    result: dict[str, Any] = {"n": int(len(valid))}
    if len(valid) < 3:
        return result

    loss = valid["log_loss"].to_numpy(dtype=float)
    correlations: dict[str, dict[str, float]] = {}
    for signal in CORRELATION_SIGNALS:
        if signal not in valid.columns:
            continue
        values = valid[signal].astype(float)
        mask = values.notna().to_numpy() & np.isfinite(loss)
        if mask.sum() < 3:
            continue
        x = values.to_numpy()[mask]
        y = loss[mask]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rq = pd.Series(x).rank().to_numpy()
        rl = pd.Series(y).rank().to_numpy()
        correlations[signal] = {
            "pearson_vs_log_loss": float(np.corrcoef(x, y)[0, 1]),
            "spearman_vs_log_loss": float(np.corrcoef(rq, rl)[0, 1]),
        }
    result["correlations"] = correlations

    error = (valid["correct"].to_numpy() == 0).astype(int)
    if len(np.unique(error)) >= 2:
        from sklearn.linear_model import LogisticRegression

        features = [valid["quality_axis"].to_numpy(dtype=float)]
        names = ["quality_axis"]
        if "any_truncation_introduced" in valid.columns:
            features.append(valid["any_truncation_introduced"].fillna(False).astype(float).to_numpy())
            names.append("truncation")
        X = np.column_stack(features)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, error)
        result["logistic_error_model"] = {
            "features": names,
            "coefficients": [float(c) for c in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
        }
    return result


@dataclass
class LinkResult:
    config: LinkConfig
    joined: pd.DataFrame
    threshold: float
    threshold_record: dict[str, Any]
    overall: dict[str, Any]
    stratified: pd.DataFrame
    curve: pd.DataFrame
    correlation: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)


def link_quality_to_detection(config: LinkConfig, *, write: bool = True) -> LinkResult:
    merged = load_and_join(config)
    if config.sample_limit is not None:
        merged = merged.sort_values(config.example_id_column).head(config.sample_limit).reset_index(drop=True)

    threshold, threshold_record = load_protocol_threshold(config)
    joined = add_error_columns(merged, config, threshold)

    y = joined[config.label_column].astype(int).to_numpy()
    scores = joined[config.score_column].astype(float).to_numpy()
    pred = joined["pred_at_threshold"].astype(int).to_numpy()
    overall = binary_metrics(y, pred, scores)

    stratified = stratified_analysis(joined, config)
    curve = filtered_curve(joined, config)
    correlation = correlation_and_regression(joined, config)

    summary = {
        "n_examples": int(len(joined)),
        "backend": config.backend,
        "quality_signal": config.quality_signal,
        "quality_direction": config.quality_direction,
        "threshold": float(threshold),
        "threshold_criterion": config.threshold_criterion,
        "threshold_source": str(config.thresholds_json),
        "threshold_rule": threshold_record.get("rule"),
        "overall": {k: overall[k] for k in ("AUPRC", "AUROC", "F1", "MCC", "FPR", "N")},
        "correlation": correlation,
        "quality_buckets": int(stratified["quality_bucket"].nunique()) if not stratified.empty else 0,
    }
    result = LinkResult(
        config, joined, float(threshold), threshold_record, overall, stratified, curve, correlation, summary
    )

    if write:
        _write_outputs(result)
    return result


def _write_outputs(result: LinkResult) -> None:
    config = result.config
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_atomic(run_dir / "linked_examples.parquet", result.joined)
    if not result.stratified.empty:
        write_csv_atomic(run_dir / "stratified_metrics.csv", result.stratified)
    if not result.curve.empty:
        write_csv_atomic(run_dir / "filtered_curve.csv", result.curve)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "signature": config.signature,
        "backend": config.backend,
        "config": config.to_dict(),
        "inputs": {
            "example_scores_sha256": sha256_file(config.example_scores_parquet),
            "predictions_sha256": sha256_file(config.predictions_parquet),
            "protocol_thresholds_sha256": sha256_file(config.thresholds_json),
        },
        "threshold": {
            "criterion": config.threshold_criterion,
            "value": result.threshold,
            "source": str(config.thresholds_json),
            "record": result.threshold_record,
        },
        "summary": result.summary,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_json_atomic(run_dir / "resolved_config.json", config.to_dict())
    write_json_atomic(run_dir / "link_summary.json", result.summary)
