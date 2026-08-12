from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .io_utils import sha256_file, write_json
from .metrics import binary_metrics, select_threshold


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifesto inválido: {path}")
    return value


def _aligned_average(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = [pd.read_csv(run_dir / filename) for run_dir in run_dirs]
    reference = frames[0][["example_id", "source_id", "task_type", "label"]].copy()
    scores = []
    for run_dir, frame in zip(run_dirs, frames):
        current = frame[["example_id", "source_id", "task_type", "label"]]
        if not reference.equals(current):
            raise ValueError(f"Previsões desalinhadas em {run_dir / filename}")
        scores.append(frame["score"].to_numpy(dtype=float))
    reference["score"] = np.mean(np.stack(scores), axis=0)
    return reference


def freeze_ensemble(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("Informe ao menos um diretório de execução.")
    manifests = [_load_manifest(run_dir / "run_manifest.json") for run_dir in run_dirs]
    reference_config = manifests[0]["config"]
    reference_data_hash = manifests[0]["data_manifest_sha256"]
    for run_dir, manifest in zip(run_dirs, manifests):
        if manifest["config"] != reference_config:
            raise ValueError(f"Configuração divergente em {run_dir}")
        if manifest["data_manifest_sha256"] != reference_data_hash:
            raise ValueError(f"Dataset divergente em {run_dir}")
        if any(value is not None for value in manifest.get("limits", {}).values()):
            raise ValueError(f"Não congele ensemble produzido com limites de smoke test: {run_dir}")

    validation = _aligned_average(run_dirs, "validation_predictions.csv")
    test = _aligned_average(run_dirs, "test_predictions.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "validation_ensemble_predictions.csv", index=False)
    test.to_csv(output_dir / "test_ensemble_predictions.csv", index=False)

    y_validation = validation["label"].to_numpy(dtype=bool)
    score_validation = validation["score"].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=bool)
    score_test = test["score"].to_numpy(dtype=float)
    thresholds: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    for criterion in ("f1", "fpr10"):
        threshold, development_metrics, feasible = select_threshold(
            y_validation, score_validation, criterion
        )
        thresholds[criterion] = {
            "threshold": threshold,
            "development_metrics": development_metrics,
            "constraint_feasible": feasible,
        }
        test_metrics[criterion] = binary_metrics(y_test, score_test >= threshold, score_test)

    config = ExperimentConfig.from_mapping(reference_config)

    model_dirs: list[str] = []
    models_root = output_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    for run_dir, manifest in zip(run_dirs, manifests):
        seed = int(manifest["seed"])
        destination = models_root / f"seed_{seed}"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        shutil.copy2(run_dir / "head.pt", destination / "head.pt")
        shutil.copy2(run_dir / "run_manifest.json", destination / "run_manifest.json")
        shutil.copytree(run_dir / "tokenizer", destination / "tokenizer")
        if config.encoder_mode == "lora":
            shutil.copytree(run_dir / "adapter", destination / "adapter")
        model_dirs.append(str(destination.relative_to(output_dir)))

    source_data_manifest = Path(manifests[0]["data_dir"]) / "manifest.json"
    shutil.copy2(source_data_manifest, output_dir / "ragtruth_data_manifest.json")

    frozen_manifest = {
        "protocol": "RAGTruth-only model selection and threshold calibration",
        "config": config.to_dict(),
        "model_dirs": model_dirs,
        "seeds": [int(manifest["seed"]) for manifest in manifests],
        "data_manifest_sha256": reference_data_hash,
        "thresholds": thresholds,
        "ragtruth_test_metrics": test_metrics,
        "artifacts": {
            "validation_predictions_sha256": sha256_file(
                output_dir / "validation_ensemble_predictions.csv"
            ),
            "test_predictions_sha256": sha256_file(output_dir / "test_ensemble_predictions.csv"),
            "ragtruth_data_manifest_sha256": sha256_file(output_dir / "ragtruth_data_manifest.json"),
        },
    }
    write_json(output_dir / "frozen_manifest.json", frozen_manifest)
    write_json(output_dir / "ragtruth_test_metrics.json", test_metrics)
    return frozen_manifest
