from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifacts import atomic_csv, atomic_json, file_manifest, make_archive
from .config import PublicHearingConfig
from .data import download_dataset, normalize_publichearing, write_metadata_csv
from .metrics import all_metrics, grouped_bootstrap, ranking_metrics, select_operating_threshold
from .splits import FoldSplit, make_splits, split_frames, splits_hash
from .tokenization import build_or_load_token_cache
from .training import seed_signature, train_seed


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _experiment_identifier(config: PublicHearingConfig) -> str:
    if config.architecture == "set_transformer":
        return "publichearingbr_lora_set_transformer_mil_5fold"
    return "publichearingbr_lora_attention_mil_5fold"


@dataclass
class PreparedExperiment:
    config: PublicHearingConfig
    frame: pd.DataFrame
    rejected: pd.DataFrame
    audit: dict[str, Any]
    folds: list[FoldSplit]
    signature: str
    run_dir: Path
    cache: dict[str, Any]
    cache_path: Path


def _smoke_subset(frame: pd.DataFrame, config: PublicHearingConfig) -> pd.DataFrame:
    if config.smoke_max_hearings is None:
        return frame
    selected: list[str] = []
    seen = set()
    for row in frame.sort_values(["hearing_id", "sample_id"]).itertuples():
        if row.hearing_id not in seen and len(selected) < config.smoke_max_hearings:
            selected.append(row.hearing_id); seen.add(row.hearing_id)
    subset = frame.loc[frame.hearing_id.isin(selected)].copy().reset_index(drop=True)
    if subset.label.nunique() != 2:
        raise RuntimeError("Smoke subset não contém as duas classes; aumente smoke_max_hearings.")
    return subset


def prepare_experiment(config: PublicHearingConfig, force_tokenization: bool = False) -> PreparedExperiment:
    dataset_path = download_dataset(config)
    frame, rejected, audit = normalize_publichearing(dataset_path)
    frame = _smoke_subset(frame, config)
    if len(frame) != audit["modelable_examples"]:
        audit = {**audit, "full_modelable_examples": audit["modelable_examples"], "modelable_examples": len(frame), "hearings": int(frame.hearing_id.nunique()), "positives": int(frame.label.sum()), "prevalence": float(frame.label.mean()), "smoke_subset": True}
    folds = make_splits(frame, config)
    if config.mode == "smoke":
        if config.smoke_fold >= len(folds):
            raise ValueError(f"smoke_fold fora do intervalo: {config.smoke_fold}")
        folds = [folds[config.smoke_fold]]
    signature_payload = {"config": config.to_dict(), "dataset_sha256": audit["dataset_sha256"], "eligible_sample_ids": frame.sample_id.tolist(), "splits_sha256": splits_hash(frame, folds)}
    signature = _stable_hash(signature_payload)[:16]
    run_dir = Path(config.output_root) / signature
    outputs = run_dir / "outputs"; outputs.mkdir(parents=True, exist_ok=True)
    write_metadata_csv(frame, rejected, outputs)
    summary, membership = split_frames(frame, folds)
    atomic_csv(outputs / "split_summary.csv", summary); atomic_csv(outputs / "fold_membership.csv", membership)
    atomic_json(run_dir / "experiment_signature.json", {"signature": signature, "payload": signature_payload})
    import peft, sklearn, transformers
    run_config = {"experiment": _experiment_identifier(config), "protocol": "supervised in-domain grouped out-of-fold", "configuration": config.to_dict(), "dataset": {**audit, "path": str(dataset_path)}, "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "scikit_learn": sklearn.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gpu_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None}}
    atomic_json(run_dir / "run_config.json", run_config)
    cache, cache_path = build_or_load_token_cache(frame, audit["dataset_sha256"], config, Path(config.output_root) / "cache", force=force_tokenization)
    return PreparedExperiment(config, frame, rejected, audit, folds, signature, run_dir, cache, cache_path)


def train_experiment(prepared: PreparedExperiment, only_fold: int | None = None, only_seed: int | None = None) -> list[dict[str, Any]]:
    folds = [fold for fold in prepared.folds if only_fold is None or fold.outer_fold == only_fold]
    if not folds:
        raise ValueError(f"Fold inexistente: {only_fold}")
    seeds = [seed for seed in prepared.config.seeds if only_seed is None or seed == only_seed]
    if not seeds:
        raise ValueError(f"Seed não configurada: {only_seed}")
    metadata: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            seed_dir = prepared.run_dir / "folds" / f"fold_{fold.outer_fold}" / f"seed_{seed}"
            _, _, _, result, reused = train_seed(prepared.config, prepared.cache, prepared.frame, prepared.signature, fold.outer_fold, seed, fold.train, fold.validation, fold.test, seed_dir)
            print(f"fold={fold.outer_fold} seed={seed} {'reutilizado' if reused else 'concluído'} best_epoch={result['best_epoch']} val_AUPRC={result['best_validation_AUPRC']:.6f}")
            metadata.append({**result, "reused": reused})
    return metadata


def _read_complete_seed(prepared: PreparedExperiment, fold: FoldSplit, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = prepared.run_dir / "folds" / f"fold_{fold.outer_fold}" / f"seed_{seed}"
    signature = seed_signature(prepared.signature, fold.outer_fold, seed, prepared.frame, fold.train, fold.validation, fold.test)
    marker = path / "SEED_RUN_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"Resultado ausente ou parcial: fold={fold.outer_fold}, seed={seed}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("seed_signature") != signature:
        raise RuntimeError(f"Resultado incompatível: fold={fold.outer_fold}, seed={seed}")
    required = ["validation_predictions.csv", "test_predictions.csv", "training_history.csv", "metadata.json"]
    if not all((path / name).is_file() for name in required):
        raise RuntimeError(f"Resultado incompleto: {path}")
    return pd.read_csv(path / required[0]), pd.read_csv(path / required[1]), pd.read_csv(path / required[2]), json.loads((path / required[3]).read_text(encoding="utf-8"))


def _ensemble(frames: list[pd.DataFrame], expected: np.ndarray) -> pd.DataFrame:
    columns = [f"attention_chunk_{index}" for index in range(4)]
    combined = pd.concat(frames, ignore_index=True)
    if set(combined.row_index) != set(expected) or len(combined) != len(expected) * len(frames):
        raise RuntimeError("Predições de seed não cobrem exatamente o split esperado.")
    result = combined.groupby("row_index", as_index=False).agg({"label": "first", "probability": "mean", **{column: "mean" for column in columns}}).sort_values("row_index").reset_index(drop=True)
    if result.row_index.tolist() != sorted(expected.tolist()):
        raise RuntimeError("Ensemble desalinhado.")
    return result


def _with_metadata(predictions: pd.DataFrame, frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    metadata = frame.reset_index(names="row_index")[["row_index", "sample_id", "hearing_id", "subject", "person_name", "person_role", "opinion", "label"]]
    output = predictions.drop(columns=["label"], errors="ignore").merge(metadata, on="row_index", validate="one_to_one")
    output["outer_fold"] = fold
    return output.sort_values("row_index").reset_index(drop=True)


def validate_oof_coverage(oof: pd.DataFrame, frame: pd.DataFrame) -> None:
    if len(oof) != len(frame) or not oof.sample_id.is_unique or set(oof.sample_id) != set(frame.sample_id):
        raise RuntimeError("Cobertura OOF inválida: toda opinião modelável deve aparecer exatamente uma vez.")


def aggregate_experiment(prepared: PreparedExperiment) -> dict[str, Any]:
    outputs = prepared.run_dir / "outputs"
    all_validation, all_test, histories, metadata_rows, metric_rows = [], [], [], [], []
    for fold in prepared.folds:
        validation_frames, test_frames = [], []
        for seed in prepared.config.seeds:
            validation, test, history, metadata = _read_complete_seed(prepared, fold, seed)
            validation_frames.append(validation); test_frames.append(test); histories.append(history); metadata_rows.append(metadata)
        validation = _ensemble(validation_frames, fold.validation)
        test = _ensemble(test_frames, fold.test)
        labels_v, probabilities_v = validation.label.to_numpy(dtype=bool), validation.probability.to_numpy(dtype=float)
        labels_t, probabilities_t = test.label.to_numpy(dtype=bool), test.probability.to_numpy(dtype=float)
        thresholds: dict[str, Any] = {}
        for criterion in ("max_f1", "fpr_operational"):
            threshold, validation_at_threshold = select_operating_threshold(labels_v, probabilities_v, criterion, prepared.config.fpr_target)
            thresholds[criterion] = {"threshold": threshold, "validation_metrics": {**ranking_metrics(labels_v, probabilities_v), **validation_at_threshold}, "source": "internal_validation_only"}
            validation[f"prediction_{criterion}"] = probabilities_v >= threshold
            test[f"prediction_{criterion}"] = probabilities_t >= threshold
            validation[f"threshold_{criterion}"] = threshold; test[f"threshold_{criterion}"] = threshold
            for split, labels, probabilities in (("validation", labels_v, probabilities_v), ("test", labels_t, probabilities_t)):
                metric_rows.append({"outer_fold": fold.outer_fold, "split": split, "criterion": criterion, "examples": len(labels), "positives": int(labels.sum()), "prevalence": float(labels.mean()), "threshold": threshold, **all_metrics(labels, probabilities, threshold)})
        for split, labels, probabilities in (("validation", labels_v, probabilities_v), ("test", labels_t, probabilities_t)):
            metric_rows.append({"outer_fold": fold.outer_fold, "split": split, "criterion": "ranking", "examples": len(labels), "positives": int(labels.sum()), "prevalence": float(labels.mean()), **ranking_metrics(labels, probabilities)})
        fold_output = outputs / f"fold_{fold.outer_fold}"
        atomic_csv(fold_output / "validation_ensemble.csv", _with_metadata(validation, prepared.frame, fold.outer_fold))
        atomic_csv(fold_output / "test_ensemble.csv", _with_metadata(test, prepared.frame, fold.outer_fold))
        atomic_json(fold_output / "thresholds.json", thresholds)
        all_validation.append(_with_metadata(validation, prepared.frame, fold.outer_fold)); all_test.append(_with_metadata(test, prepared.frame, fold.outer_fold))
    oof = pd.concat(all_test, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    expected_oof = prepared.frame.iloc[np.concatenate([fold.test for fold in prepared.folds])]
    validate_oof_coverage(oof, expected_oof)
    validation_all, histories_all, metadata_all, metrics = pd.concat(all_validation, ignore_index=True), pd.concat(histories, ignore_index=True), pd.DataFrame(metadata_rows), pd.DataFrame(metric_rows)
    atomic_csv(outputs / "oof_predictions.csv", oof); atomic_csv(outputs / "all_validation_predictions.csv", validation_all); atomic_csv(outputs / "training_history.csv", histories_all); atomic_csv(outputs / "seed_run_metadata.csv", metadata_all); atomic_csv(outputs / "fold_metrics.csv", metrics)
    overall: list[dict[str, Any]] = [{"criterion": "ranking", "examples": len(oof), "hearings": oof.hearing_id.nunique(), "positives": int(oof.label.sum()), "prevalence": float(oof.label.mean()), **ranking_metrics(oof.label.to_numpy(dtype=bool), oof.probability.to_numpy(dtype=float))}]
    for criterion in ("max_f1", "fpr_operational"):
        threshold_metrics = all_metrics(oof.label.to_numpy(dtype=bool), oof.probability.to_numpy(dtype=float), 0.5)
        # Existing fold-calibrated decisions, rather than an OOF-derived threshold.
        from ..metrics import binary_metrics
        threshold_metrics.update(binary_metrics(oof.label.to_numpy(dtype=bool), oof[f"prediction_{criterion}"].to_numpy(dtype=bool)))
        overall.append({"criterion": criterion, "examples": len(oof), "hearings": oof.hearing_id.nunique(), "positives": int(oof.label.sum()), "prevalence": float(oof.label.mean()), **threshold_metrics})
    atomic_csv(outputs / "overall_oof_metrics.csv", pd.DataFrame(overall))
    samples, summary = grouped_bootstrap(oof, prepared.config.bootstrap_repetitions, prepared.config.split_seed + 50_000)
    atomic_csv(outputs / "cluster_bootstrap_samples.csv", samples); atomic_csv(outputs / "cluster_bootstrap_summary.csv", summary)
    _plots_and_attention(oof, outputs)
    manifest = file_manifest(outputs); atomic_json(outputs / "file_manifest.json", manifest)
    archive = make_archive(prepared.run_dir, prepared.config.include_checkpoints_in_zip)
    return {"run_dir": str(prepared.run_dir), "signature": prepared.signature, "oof_examples": len(oof), "archive": str(archive), "outputs": str(outputs)}


def _plots_and_attention(oof: pd.DataFrame, outputs: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    labels, probabilities = oof.label.to_numpy(dtype=bool), oof.probability.to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(8, 6)); plt.plot(recall, precision); plt.axhline(labels.mean(), linestyle="--"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.savefig(outputs / "oof_precision_recall_curve.png", dpi=160); plt.close()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.tight_layout(); plt.savefig(outputs / "oof_roc_curve.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 6)); plt.hist(probabilities[~labels], bins=40, alpha=.6, density=True); plt.hist(probabilities[labels], bins=40, alpha=.6, density=True); plt.xlabel("Probabilidade prevista"); plt.tight_layout(); plt.savefig(outputs / "oof_probability_histogram.png", dpi=160); plt.close()
    columns = [f"attention_chunk_{index}" for index in range(4)]
    summary = oof.melt(id_vars=["sample_id", "label"], value_vars=columns, var_name="chunk", value_name="attention").groupby(["label", "chunk"], as_index=False).attention.agg(["mean", "std", "count"]).reset_index()
    atomic_csv(outputs / "attention_summary.csv", summary)
    top = oof[["sample_id", "hearing_id", "label", "probability", "opinion", *columns]].copy(); top["selected_chunk"] = top[columns].to_numpy().argmax(axis=1); top["max_attention"] = top[columns].max(axis=1)
    atomic_csv(outputs / "top_200_predictions_with_attention.csv", top.sort_values("probability", ascending=False).head(200))


def validate_run(run_dir: Path) -> dict[str, Any]:
    signature_path, oof_path, membership_path = run_dir / "experiment_signature.json", run_dir / "outputs" / "oof_predictions.csv", run_dir / "outputs" / "fold_membership.csv"
    if not signature_path.is_file() or not oof_path.is_file() or not membership_path.is_file():
        raise RuntimeError("Run não possui os artefatos necessários para validação.")
    oof, membership = pd.read_csv(oof_path), pd.read_csv(membership_path)
    if not oof.sample_id.is_unique or not np.isfinite(oof.probability).all():
        raise RuntimeError("OOF inválido.")
    for fold, part in membership.groupby("outer_fold"):
        groups = {role: set(rows.hearing_id.astype(str)) for role, rows in part.groupby("role")}
        if groups.get("train", set()) & groups.get("validation", set()) or groups.get("train", set()) & groups.get("test", set()) or groups.get("validation", set()) & groups.get("test", set()):
            raise RuntimeError(f"Leakage no fold {fold}")
    return {"status": "valid", "signature": json.loads(signature_path.read_text())["signature"], "oof_examples": len(oof), "folds": int(oof.outer_fold.nunique())}
