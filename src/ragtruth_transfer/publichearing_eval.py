from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .dataset import BagCollator, EvidenceBagDataset
from .io_utils import sha256_file, write_json
from .metrics import binary_metrics, paired_cluster_bootstrap
from .modeling import build_model, load_head_state
from .publichearing_data import export_publichearing_examples
from .training import predict


def _coerce_boolean(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized.unique()).difference(mapping))
    if unknown:
        raise ValueError(f"Valores desconhecidos em {column_name}: {unknown[:10]}")
    return normalized.map(mapping).astype(bool)


def _load_frozen_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifesto inválido: {path}")
    return value


def evaluate_publichearing(
    frozen_manifest_path: Path,
    publichearing_path: Path,
    output_dir: Path,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    max_examples: int | None = None,
) -> dict[str, Any]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples deve ser positivo.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_manifest(frozen_manifest_path)
    config = ExperimentConfig.from_mapping(frozen["config"])
    examples_path = output_dir / "publichearing_examples.jsonl"
    full_rows = export_publichearing_examples(publichearing_path, examples_path)
    full_frame = pd.DataFrame(full_rows)
    evaluation_frame = full_frame.loc[full_frame["chunk_count"].eq(4)].copy().reset_index(drop=True)
    if len(evaluation_frame) != 4237:
        raise AssertionError(f"Esperavam-se 4.237 exemplos; obtidos {len(evaluation_frame)}")
    if int(evaluation_frame["manual_hallucination"].sum()) != 503:
        raise AssertionError("Número inesperado de positivos no PublicHearingBR.")

    full_examples = len(evaluation_frame)
    full_positives = int(evaluation_frame["manual_hallucination"].sum())
    if max_examples is not None:
        evaluation_frame = evaluation_frame.iloc[:max_examples].copy().reset_index(drop=True)

    dataset = EvidenceBagDataset(examples_path, limit=max_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_scores: list[np.ndarray] = []
    reference_ids: list[str] | None = None

    if "model_dirs" in frozen:
        run_dirs = [frozen_manifest_path.parent / value for value in frozen["model_dirs"]]
    else:
        run_dirs = [Path(value) for value in frozen["run_dirs"]]

    for run_dir in run_dirs:
        adapter_path = run_dir / "adapter" if config.encoder_mode == "lora" else None
        model, _ = build_model(config, adapter_path=adapter_path)
        load_head_state(model, run_dir / "head.pt")
        model.to(device)
        tokenizer_path = run_dir / "tokenizer"
        tokenizer_source = tokenizer_path if tokenizer_path.is_dir() else config.model_id
        tokenizer_kwargs = {"use_fast": True}
        if tokenizer_source == config.model_id and config.model_revision:
            tokenizer_kwargs["revision"] = config.model_revision
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        loader = DataLoader(
            dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=BagCollator(tokenizer, config.max_length),
            pin_memory=torch.cuda.is_available(),
        )
        predictions = predict(model, loader, device)
        ids = predictions["example_id"].astype(str).tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError("Previsões de seeds desalinhadas.")
        seed_scores.append(predictions["score"].to_numpy(dtype=float))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = np.mean(np.stack(seed_scores), axis=0)
    if reference_ids != evaluation_frame["sample_id"].astype(str).tolist():
        raise RuntimeError("Ordem das previsões incompatível com o PublicHearingBR.")
    evaluation_frame["external_score"] = scores
    y_true = evaluation_frame["manual_hallucination"].to_numpy(dtype=bool)
    groups = evaluation_frame["hearing_id"].astype(str).to_numpy()

    metric_rows: list[dict[str, Any]] = []
    model_predictions: dict[str, np.ndarray] = {}
    for criterion in ("f1", "fpr10"):
        threshold = float(frozen["thresholds"][criterion]["threshold"])
        prediction = scores >= threshold
        model_predictions[criterion] = prediction
        evaluation_frame[f"external_pred_{criterion}"] = prediction
        metric_rows.append(
            {
                "system": "RAGTruth external ensemble",
                "criterion": criterion,
                "threshold_source": "RAGTruth validation",
                "threshold": threshold,
                **binary_metrics(y_true, prediction, scores),
            }
        )

    llm_columns = sorted(column for column in evaluation_frame.columns if column.startswith("prompt_"))
    llm_predictions: dict[str, np.ndarray] = {}
    for column in llm_columns:
        prediction = _coerce_boolean(evaluation_frame[column], column).to_numpy(dtype=bool)
        llm_predictions[column] = prediction
        metric_rows.append(
            {
                "system": column,
                "criterion": "fixed_binary",
                "threshold_source": "authors",
                "threshold": np.nan,
                **binary_metrics(y_true, prediction),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    llm_metrics = metrics_frame.loc[metrics_frame["criterion"].eq("fixed_binary")]
    best_llm_f1 = str(llm_metrics.sort_values("F1", ascending=False).iloc[0]["system"])
    operational = llm_metrics.loc[llm_metrics["FPR"] <= 0.10]
    best_llm_operational = str(
        operational.sort_values(["Recall", "F1"], ascending=False).iloc[0]["system"]
    )

    bootstrap_rows: list[dict[str, Any]] = []
    for criterion, model_prediction in model_predictions.items():
        for comparison_name in sorted({best_llm_f1, best_llm_operational}):
            comparison_prediction = llm_predictions[comparison_name]
            rows = paired_cluster_bootstrap(
                y_true=y_true,
                groups=groups,
                first_pred=model_prediction,
                first_scores=scores,
                second_pred=comparison_prediction,
                second_scores=comparison_prediction.astype(float),
                metrics=("F1", "Recall", "FPR", "MCC"),
                n_resamples=bootstrap_resamples,
                seed=seed,
            )
            for row in rows:
                row.update(
                    {
                        "model_criterion": criterion,
                        "comparison_system": comparison_name,
                    }
                )
                bootstrap_rows.append(row)

    predictions_path = output_dir / "publichearing_predictions.csv"
    metrics_path = output_dir / "publichearing_metrics.csv"
    bootstrap_path = output_dir / "paired_cluster_bootstrap.csv"
    evaluation_frame.to_csv(predictions_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)

    result_manifest = {
        "protocol": "zero-shot transfer: RAGTruth training/calibration, PublicHearingBR evaluation only",
        "frozen_manifest": str(frozen_manifest_path.resolve()),
        "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
        "publichearing_path": str(publichearing_path.resolve()),
        "publichearing_sha256": sha256_file(publichearing_path),
        "examples": len(evaluation_frame),
        "positives": int(y_true.sum()),
        "full_examples": full_examples,
        "full_positives": full_positives,
        "max_examples": max_examples,
        "hearings": int(evaluation_frame["hearing_id"].nunique()),
        "best_llm_f1": best_llm_f1,
        "best_llm_operational": best_llm_operational,
        "bootstrap_resamples": bootstrap_resamples,
        "outputs": {
            predictions_path.name: sha256_file(predictions_path),
            metrics_path.name: sha256_file(metrics_path),
            bootstrap_path.name: sha256_file(bootstrap_path),
        },
    }
    write_json(output_dir / "evaluation_manifest.json", result_manifest)
    return result_manifest
