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
from .paired_grouped_bootstrap import run_generic_pairwise_bootstrap
from .thresholded_paired_grouped_bootstrap import run_generic_thresholded_pair_bootstrap


EXPECTED_ROWS = 4235
EXPECTED_POSITIVES = 501
EXPECTED_HEARINGS = 206
SEEDS = (0, 1, 2)
REGIMES = ("best_f1", "fpr10")
RANKING_METRICS = ("auprc", "auroc", "brier")
THRESHOLDED_METRICS = ("precision", "recall", "f1", "mcc", "fpr")
LOWER_IS_BETTER_METRICS = frozenset(("brier", "fpr"))


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    path: Path
    expected_signature: str
    target: str
    architecture: str
    training_representation: str
    expected_pooling: str
    score_column: str = "probability"


@dataclass(frozen=True)
class FinalBootstrapConfig:
    output_root: Path
    conditions: dict[str, ConditionSpec]
    pairwise_contrasts: dict[str, tuple[str, str]]
    contrast_families: dict[str, str]
    contrast_labels: dict[str, str]
    interactions: dict[str, tuple[str, str]]
    interaction_families: dict[str, str]
    interaction_labels: dict[str, str]
    thresholded_contrasts: dict[str, tuple[str, str]]
    thresholded_labels: dict[str, str]
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]
    n_replicates: int
    seed: int
    confidence_level: float
    ci_method: str
    expected_rows: int
    expected_positives: int
    expected_hearings: int
    publichearing_signature: str
    ragtruth_signature: str

    @classmethod
    def from_yaml(cls, path: Path) -> "FinalBootstrapConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("analysis", {}).get("type") != "final_paired_grouped_bootstrap":
            raise ValueError(f"Configuração final inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        population = raw.get("population", {})
        bootstrap = raw.get("bootstrap", {})
        input_block = raw.get("inputs", {})
        conditions: dict[str, ConditionSpec] = {}
        for name, value in input_block.get("conditions", {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"Condition inválida: {name}")
            conditions[str(name)] = ConditionSpec(
                label=str(value.get("label", name)), path=resolve(value.get("path")),
                expected_signature=str(value.get("expected_signature", "")), target=str(value.get("target", "")),
                architecture=str(value.get("architecture", "")), training_representation=str(value.get("training_representation", "")),
                expected_pooling=str(value.get("expected_pooling", "")), score_column=str(value.get("score_column", "probability")),
            )

        def contrasts(key: str) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, str]]:
            definitions: dict[str, tuple[str, str]] = {}
            families: dict[str, str] = {}
            labels: dict[str, str] = {}
            for value in raw.get(key, []):
                name = str(value["id"])
                definitions[name] = (str(value["a"]), str(value["b"]))
                families[name] = str(value.get("family", "pairwise"))
                labels[name] = str(value.get("label", name))
            return definitions, families, labels

        pairwise, pair_families, pair_labels = contrasts("contrasts")
        thresholded, _, threshold_labels = contrasts("thresholded_contrasts")
        interaction_defs: dict[str, tuple[str, str]] = {}
        interaction_families: dict[str, str] = {}
        interaction_labels: dict[str, str] = {}
        for value in raw.get("interactions", []):
            name = str(value["id"])
            interaction_defs[name] = (str(value["first"]), str(value["second"]))
            interaction_families[name] = str(value.get("family", "interaction"))
            interaction_labels[name] = str(value.get("label", name))
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_final_paired_bootstrap")),
            conditions=conditions, pairwise_contrasts=pairwise, contrast_families=pair_families, contrast_labels=pair_labels,
            interactions=interaction_defs, interaction_families=interaction_families, interaction_labels=interaction_labels,
            thresholded_contrasts=thresholded, thresholded_labels=threshold_labels,
            seeds=tuple(int(item) for item in input_block.get("seeds", SEEDS)),
            regimes=tuple(str(item) for item in raw.get("threshold_regimes", REGIMES)),
            n_replicates=int(bootstrap.get("n_replicates", 10000)), seed=int(bootstrap.get("seed", 20260815)),
            confidence_level=float(bootstrap.get("confidence_level", 0.95)), ci_method=str(bootstrap.get("ci_method", "percentile")),
            expected_rows=int(population.get("examples", EXPECTED_ROWS)), expected_positives=int(population.get("positives", EXPECTED_POSITIVES)),
            expected_hearings=int(population.get("hearings", EXPECTED_HEARINGS)),
            publichearing_signature=str(population.get("publichearing_dataset_signature", "")),
            ragtruth_signature=str(population.get("ragtruth_dataset_signature", "")),
        )
        config.validate(raw.get("protocol", {}))
        return config

    def validate(self, protocol: dict[str, Any]) -> None:
        if protocol.get("group_key") != "hearing_id" or protocol.get("paired") is not True:
            raise ValueError("O protocolo deve ser pareado e agrupado por hearing_id")
        if self.seeds != SEEDS or self.regimes != REGIMES:
            raise ValueError("A análise exige seeds [0, 1, 2] e regimes best_f1/fpr10")
        if self.n_replicates < 1 or self.seed < 0 or not 0 < self.confidence_level < 1 or self.ci_method != "percentile":
            raise ValueError("Configuração de bootstrap inválida")
        if not self.conditions or not self.pairwise_contrasts or not self.thresholded_contrasts:
            raise ValueError("Registry ou contrasts ausentes")
        for name, spec in self.conditions.items():
            if not spec.expected_signature or spec.score_column != "probability":
                raise ValueError(f"Metadata inválida para {name}")
        for name, (a, b) in {**self.pairwise_contrasts, **self.thresholded_contrasts}.items():
            if a not in self.conditions or b not in self.conditions or a == b:
                raise ValueError(f"Condição ausente no contraste {name}")
        for name, (first, second) in self.interactions.items():
            if first not in self.pairwise_contrasts or second not in self.pairwise_contrasts:
                raise ValueError(f"Interaction referencia contraste ausente: {name}")

    def to_dict(self) -> dict[str, Any]:
        def spec_dict(spec: ConditionSpec) -> dict[str, Any]:
            return {"label": spec.label, "path": str(spec.path), "expected_signature": spec.expected_signature,
                    "target": spec.target, "architecture": spec.architecture, "training_representation": spec.training_representation,
                    "expected_pooling": spec.expected_pooling, "score_column": spec.score_column}

        return {
            "analysis": {"type": "final_paired_grouped_bootstrap", "name": "PublicHearingBR final paper bootstrap"},
            "protocol": {"group_key": "hearing_id", "paired": True, "delta": "condition_b - condition_a",
                         "thresholds": "frozen per condition and seed from validation only"},
            "inputs": {"conditions": {name: spec_dict(spec) for name, spec in self.conditions.items()}, "seeds": list(self.seeds)},
            "contrasts": [{"id": name, "a": a, "b": b, "family": self.contrast_families[name], "label": self.contrast_labels[name]}
                          for name, (a, b) in self.pairwise_contrasts.items()],
            "interactions": [{"id": name, "first": first, "second": second, "family": self.interaction_families[name], "label": self.interaction_labels[name]}
                             for name, (first, second) in self.interactions.items()],
            "thresholded_contrasts": [{"id": name, "a": a, "b": b, "label": self.thresholded_labels[name]}
                                       for name, (a, b) in self.thresholded_contrasts.items()],
            "threshold_regimes": list(self.regimes),
            "population": {"examples": self.expected_rows, "positives": self.expected_positives, "hearings": self.expected_hearings,
                           "publichearing_dataset_signature": self.publichearing_signature, "ragtruth_dataset_signature": self.ragtruth_signature},
            "bootstrap": {"n_replicates": self.n_replicates, "seed": self.seed, "confidence_level": self.confidence_level,
                          "ci_method": self.ci_method, "preserve_group_multiplicity": True, "same_samples_across_conditions": True,
                          "same_samples_across_seeds": True, "cpu_only": True, "model_load": False, "inference": False},
            "output": {"root": str(self.output_root)},
        }


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_manifest(directory: Path, expected_signature: str | None = None) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto ausente: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if expected_signature and expected_signature not in {str(manifest.get("signature")), str(manifest.get("protocol_signature"))}:
        raise ValueError(f"Assinatura divergente em {path}")
    if manifest.get("status") not in {"completed", "externally_evaluated", "trained"}:
        raise ValueError(f"Status inválido em {path}")
    for name, digest in manifest.get("artifacts", {}).items():
        artifact = directory / name
        if artifact.is_file() and sha256_file(artifact) != digest:
            raise ValueError(f"Hash divergente em {artifact}")
    return manifest


def _load_condition(spec: ConditionSpec, config: FinalBootstrapConfig) -> tuple[dict[int, pd.DataFrame], dict[str, str], dict[str, Any]]:
    root_manifest = _validate_manifest(spec.path, spec.expected_signature)
    if tuple(int(seed) for seed in root_manifest.get("seeds", [])) != config.seeds:
        raise ValueError(f"Seeds incompatíveis em {spec.label}")
    frames: dict[int, pd.DataFrame] = {}
    hashes: dict[str, str] = {f"{spec.label}/manifest.json": sha256_file(spec.path / "manifest.json")}
    audit: dict[str, Any] = {"label": spec.label, "path": str(spec.path), "signature": spec.expected_signature, "target": spec.target,
                             "architecture": spec.architecture, "training_representation": spec.training_representation, "seeds": {}}
    for seed in config.seeds:
        seed_dir = spec.path / f"seed_{seed}"
        seed_manifest = _validate_manifest(seed_dir) if (seed_dir / "manifest.json").is_file() else {"status": "externally_evaluated"}
        prediction_manifests = sorted((seed_dir / "publichearing_zero_shot").glob("*/manifest.json"))
        if len(prediction_manifests) != 1:
            raise ValueError(f"Esperava uma predição em {spec.label}/seed_{seed}")
        run_dir = prediction_manifests[0].parent
        prediction_manifest = _validate_manifest(run_dir, run_dir.name)
        pooling = prediction_manifest.get("pooling_type") or prediction_manifest.get("model", {}).get("pooling_type") or prediction_manifest.get("model", {}).get("architecture")
        accepted_pooling = {spec.expected_pooling, "gated_attention" if spec.expected_pooling == "attention" else spec.expected_pooling,
                            "attention" if spec.expected_pooling == "attention" else spec.expected_pooling}
        if pooling not in accepted_pooling:
            raise ValueError(f"Pooling incompatível em {spec.label}/seed_{seed}")
        prediction = run_dir / "predictions.parquet"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        declared_hash = prediction_manifest.get("artifacts", {}).get("predictions.parquet")
        actual_hash = sha256_file(prediction)
        if declared_hash and declared_hash != actual_hash:
            raise ValueError(f"Hash de predição divergente: {prediction}")
        frame = pd.read_parquet(prediction)
        required = {"example_id", "hearing_id", "label", spec.score_column}
        if not required.issubset(frame.columns):
            raise ValueError(f"Colunas ausentes em {prediction}")
        selected = frame[["example_id", "hearing_id", "label", spec.score_column]].copy()
        selected["example_id"] = selected.example_id.astype(str)
        selected["hearing_id"] = selected.hearing_id.astype(str)
        if len(selected) != config.expected_rows or not selected.example_id.is_unique or selected.hearing_id.nunique() != config.expected_hearings:
            raise ValueError(f"Population/IDs inválidos em {prediction}")
        if not set(selected.label.unique()).issubset({0, 1}) or set(selected.label.unique()) != {0, 1} or int(selected.label.sum()) != config.expected_positives or selected.isna().any().any():
            raise ValueError(f"Labels/nulos inválidos em {prediction}")
        scores = selected[spec.score_column].to_numpy(float)
        if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
            raise ValueError(f"Scores inválidos em {prediction}")
        frames[seed] = selected
        hashes[f"{spec.label}/seed_{seed}/predictions.parquet"] = actual_hash
        audit["seeds"][str(seed)] = {"prediction": str(prediction), "prediction_manifest": str(prediction_manifests[0]),
                                      "prediction_signature": run_dir.name, "examples": len(selected), "hearings": int(selected.hearing_id.nunique()),
                                      "positives": int(selected.label.sum()), "seed_manifest_status": seed_manifest.get("status")}
    return frames, hashes, audit


def _load_thresholds(spec: ConditionSpec, config: FinalBootstrapConfig) -> tuple[dict[int, dict[str, float]], dict[str, Any], dict[str, str]]:
    from .metrics import select_threshold

    thresholds: dict[int, dict[str, float]] = {}
    audit: dict[str, Any] = {"label": spec.label, "source": "RAGTruth validation only", "publichearing_labels_used_for_selection": False, "seeds": {}}
    hashes: dict[str, str] = {}
    for seed in config.seeds:
        seed_dir = spec.path / f"seed_{seed}"
        nested_manifests = sorted((seed_dir / "publichearing_zero_shot").glob("*/manifest.json"))
        nested_dir = nested_manifests[0].parent if len(nested_manifests) == 1 else None
        threshold_path = seed_dir / "thresholds.json"
        if not threshold_path.is_file() and nested_dir is not None:
            threshold_path = nested_dir / "thresholds.json"
        if not threshold_path.is_file():
            raise FileNotFoundError(f"Threshold ausente em {spec.label}/seed_{seed}")
        persisted = json.loads(threshold_path.read_text(encoding="utf-8"))
        validation_path = seed_dir / "validation_predictions.csv"
        if not validation_path.is_file():
            declared_paths = [str(item.get("validation_path", "")) for item in persisted.values() if isinstance(item, dict)]
            candidates = [Path(item) for item in declared_paths if item and Path(item).is_file()]
            expected_validation_hash = next((str(item.get("validation_sha256")) for item in persisted.values() if isinstance(item, dict) and item.get("validation_sha256")), "")
            if not candidates and expected_validation_hash:
                repo_root = spec.path.parent.parent
                candidates = [item for item in repo_root.rglob("validation_predictions.csv") if sha256_file(item) == expected_validation_hash]
            if not candidates:
                raise FileNotFoundError(f"Validation de threshold não localizada em {spec.label}/seed_{seed}")
            validation_path = sorted(candidates)[0]
        validation = pd.read_csv(validation_path)
        score_column = "score" if "score" in validation.columns else "probability"
        if len(validation) != 4517 or not {"label", score_column}.issubset(validation.columns):
            raise ValueError(f"Validation inválida em {validation_path}")
        values: dict[str, float] = {}
        audit["seeds"][str(seed)] = {"validation_artifact": str(validation_path), "validation_examples": len(validation), "regimes": {}}
        for regime in config.regimes:
            key = "f1" if regime == "best_f1" else "fpr10"
            item = persisted.get(key, {})
            persisted_value = float(item.get("threshold", np.nan))
            recomputed, _, feasible = select_threshold(validation.label.to_numpy(), validation[score_column].to_numpy(float), key)
            if not np.isclose(persisted_value, recomputed, rtol=0, atol=1e-12) or item.get("constraint_feasible") is not True:
                raise ValueError(f"Threshold não reproduzido em {spec.label}/seed_{seed}/{regime}")
            validation_hash = sha256_file(validation_path)
            if item.get("validation_sha256") and item["validation_sha256"] != validation_hash:
                raise ValueError(f"Hash de validation divergente em {spec.label}/seed_{seed}/{regime}")
            values[regime] = persisted_value
            audit["seeds"][str(seed)]["regimes"][regime] = {"threshold": persisted_value, "recomputed_threshold": float(recomputed),
                                                               "validation_sha256": validation_hash, "validation_examples": len(validation),
                                                               "constraint_feasible": bool(feasible), "frozen": True, "source_artifact": str(threshold_path)}
            hashes[f"{spec.label}/seed_{seed}/{regime}/thresholds.json"] = sha256_file(threshold_path)
        hashes[f"{spec.label}/seed_{seed}/validation_predictions.csv"] = sha256_file(validation_path)
        thresholds[seed] = values
    return thresholds, audit, hashes


def _metric_favors_lower(metric: str) -> bool:
    return metric.lower() in LOWER_IS_BETTER_METRICS


def _significant(summary: dict[str, Any], metric: str, interaction: bool = False) -> str:
    low, high = summary.get("ci_lower"), summary.get("ci_upper")
    if low is None or high is None:
        return "NO_VALID_REPLICATES"
    if interaction:
        return "NEGATIVE" if high < 0 else "POSITIVE" if low > 0 else "INCONCLUSIVE"
    if _metric_favors_lower(metric):
        return "FAVOR_B" if high < 0 else "FAVOR_A" if low > 0 else "INCONCLUSIVE"
    return "FAVOR_B" if low > 0 else "FAVOR_A" if high < 0 else "INCONCLUSIVE"


def _summary_rows(config: FinalBootstrapConfig, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast, (a, b) in config.pairwise_contrasts.items():
        for metric in RANKING_METRICS:
            point_a = float(np.mean([result["observed"][a][str(seed)][metric] for seed in config.seeds]))
            point_b = float(np.mean([result["observed"][b][str(seed)][metric] for seed in config.seeds]))
            summary = result["summaries"][contrast][metric]
            rows.append({"contrast_id": contrast, "contrast_family": config.contrast_families[contrast], "contrast_label": config.contrast_labels[contrast],
                         "condition_a": a, "condition_b": b, "metric": metric, "regime": "threshold_free", "point_a": point_a, "point_b": point_b,
                         "observed_delta": point_b - point_a, "bootstrap_mean_delta": summary["mean"], "bootstrap_median_delta": summary["median"],
                         "ci_low": summary["ci_lower"], "ci_high": summary["ci_upper"], "p_delta_gt_zero": summary["p_delta_gt_zero"],
                         "p_delta_lt_zero": summary["p_delta_lt_zero"], "probability_favorable": summary["probability_favorable"],
                         "favorable_direction": f"lower for {b}" if _metric_favors_lower(metric) else f"higher for {b}", "significant_direction": _significant(summary, metric),
                         "replicates": config.n_replicates, "valid_replicates": summary["n_valid"], "invalid_replicates": config.n_replicates - summary["n_valid"],
                         "seed": config.seed, "bootstrap_seed": config.seed, "population_examples": config.expected_rows, "population_positives": config.expected_positives, "population_hearings": config.expected_hearings})
    for interaction, (first, second) in config.interactions.items():
        for metric in RANKING_METRICS:
            summary = result["summaries"][interaction][metric]
            rows.append({"contrast_id": interaction, "contrast_family": config.interaction_families[interaction], "contrast_label": config.interaction_labels[interaction],
                         "condition_a": first, "condition_b": second, "metric": metric, "regime": "interaction", "point_a": None, "point_b": None,
                         "observed_delta": result["observed_campaign"][interaction][metric], "bootstrap_mean_delta": summary["mean"], "bootstrap_median_delta": summary["median"],
                         "ci_low": summary["ci_lower"], "ci_high": summary["ci_upper"], "p_delta_gt_zero": summary["p_delta_gt_zero"],
                         "p_delta_lt_zero": summary["p_delta_lt_zero"], "probability_favorable": summary["p_delta_lt_zero"],
                         "favorable_direction": ("lower for target-minus-training effect; Brier interpreted separately"
                                                 if config.interaction_families[interaction] == "translation_interaction"
                                                 else "difference of Set-minus-Attention effects; Brier interpreted separately"), "significant_direction": _significant(summary, metric, True),
                         "replicates": config.n_replicates, "valid_replicates": summary["n_valid"], "invalid_replicates": config.n_replicates - summary["n_valid"],
                         "seed": config.seed, "bootstrap_seed": config.seed, "population_examples": config.expected_rows, "population_positives": config.expected_positives, "population_hearings": config.expected_hearings})
    return rows


def _threshold_rows(config: FinalBootstrapConfig, contrast: str, a: str, b: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime in config.regimes:
        for metric in THRESHOLDED_METRICS:
            summary = result["summaries"][regime][metric]
            rows.append({"contrast_id": contrast, "contrast_family": "thresholded_set_attention", "contrast_label": config.thresholded_labels[contrast],
                         "condition_a": a, "condition_b": b, "metric": metric, "regime": regime,
                         "point_a": summary["point_a"], "point_b": summary["point_b"], "observed_delta": summary["observed_delta"],
                         "bootstrap_mean_delta": summary["bootstrap"]["mean"], "bootstrap_median_delta": summary["bootstrap"]["median"],
                         "ci_low": summary["bootstrap"]["ci_lower"], "ci_high": summary["bootstrap"]["ci_upper"],
                         "p_delta_gt_zero": summary["bootstrap"]["p_delta_gt_zero"], "p_delta_lt_zero": summary["bootstrap"]["p_delta_lt_zero"],
                         "probability_favorable": summary["bootstrap"]["probability_favorable"],
                         "favorable_direction": f"lower for {b}" if _metric_favors_lower(metric) else f"higher for {b}",
                         "significant_direction": _significant(summary["bootstrap"], metric), "replicates": config.n_replicates,
                         "valid_replicates": summary["valid_replicates"], "invalid_replicates": summary["invalid_replicates"], "seed": config.seed,
                         "bootstrap_seed": config.seed, "population_examples": config.expected_rows, "population_positives": config.expected_positives, "population_hearings": config.expected_hearings})
    return rows


def _report(config: FinalBootstrapConfig, signature: str, population: dict[str, Any], rows: pd.DataFrame, threshold_audit: dict[str, Any]) -> str:
    lines = ["# Final paired grouped bootstrap", "", "## Classification", "", "PASS", "", "## Executive summary", "",
             f"- {len(config.conditions)} declarative conditions, {len(config.pairwise_contrasts)} threshold-free contrasts, {len(config.interactions)} interactions and {len(config.thresholded_contrasts)} thresholded contrasts.",
             f"- {config.n_replicates:,} hearing-level bootstrap replicates, seed {config.seed}, percentile {config.confidence_level:.0%} intervals.",
             f"- Population: {population['examples']} examples, {population['positives']} positives, {population['hearings']} hearings.",
             "- Every condition uses the same sampled hearings per replicate and seed-matched predictions; repeated hearing multiplicity is preserved.",
             "- Delta is condition B minus condition A. AUPRC/AUROC favor positive deltas; Brier/FPR favor negative deltas.", "", "## Implementation summary", "",
             "- Generic threshold-free A/B core: `run_generic_pairwise_bootstrap`.",
             "- Generic thresholded A/B core: `run_generic_thresholded_pair_bootstrap` with frozen validation thresholds.",
             "- Factorial API accepts arbitrary condition registries, pairwise contrasts and derived interactions.",
             "- No model, checkpoint, CUDA, training, translation or inference path was used.", "", "## Population validation", "",
             f"- Exact alignment: examples={population['examples']}, positives={population['positives']}, hearings={population['hearings']}; IDs, hearings and labels matched across conditions.",
             "- Source prediction files were read-only inputs.", "", "## Threshold provenance", "",
             f"- Thresholds audited for {len(threshold_audit)} conditions, per seed and regime.",
             "- Thresholds came only from each condition's RAGTruth validation artifact; PublicHearing labels were not used for selection.", ""]
    sections = [("Aggregator effect threshold-free", "aggregator"), ("Target translation effect Attention/Set", "target_translation"),
                ("Training translation effect Attention/Set", "training_translation"), ("Translation-side interaction", "translation_interaction"),
                ("Set-effect interactions", "set_effect_interaction"), ("Thresholded Set vs Attention", "thresholded_set_attention")]
    for title, family in sections:
        subset = rows[rows.contrast_family == family]
        lines += [f"## {title}", ""]
        if subset.empty:
            lines.append("- Not run.")
        else:
            lines += ["| Contrast | Regime | Metric | Observed delta | 95% CI | Probability favorable | Direction |", "|---|---|---|---:|---|---:|---|"]
            for _, item in subset.iterrows():
                lines.append(f"| {item.contrast_label} | {item.regime} | {item.metric} | {item.observed_delta:.6f} | [{item.ci_low:.6f}, {item.ci_high:.6f}] | {item.probability_favorable:.4f} | {item.significant_direction} |")
        lines.append("")
    lines += ["## Final claim support matrix", "", "| Claim | Evidence status |", "|---|---|",
              "| Set versus Attention ranking effects | Reported with paired grouped CIs |",
              "| Target translation versus original Portuguese target | Reported for Attention and Set |",
              "| Training translation effects | Reported for Attention and Set |",
              "| Target translation more damaging than training translation | Reported as shared-bootstrap interactions; Brier interpreted separately |",
              "| Thresholded operating points | Reported for required regimes and metrics |", "", "## Consolidated artifacts", "",
              f"- `final_bootstrap_summary.csv` contains {len(rows)} rows.", "- Each contrast has its own summary and bootstrap distribution under `contrasts/` or `thresholded/`.",
              "", "## Tests and execution integrity", "", "- Tests and static checks are reported by the executor.",
              "- Model loaded: NO; checkpoint loaded: NO; CUDA initialized: NO; inference/training/translation: NO.",
              "- Historical artifacts and input predictions were not overwritten.", "", "## Paper closure status", "",
              "- Primary threshold-free and thresholded contrast families are persisted in a new analysis namespace.",
              "- Claims remain descriptive and protocol-bound; no causal explanation is asserted.", "", "## Final verdict", "", f"The final bootstrap analysis is complete under signature `{signature}`."]
    return "\n".join(lines) + "\n"


def run_final_bootstrap(config: FinalBootstrapConfig, *, validate_only: bool = False) -> dict[str, Any]:
    started = time.time()
    frames: dict[str, dict[int, pd.DataFrame]] = {}
    input_hashes: dict[str, str] = {}
    condition_audits: dict[str, Any] = {}
    for name, spec in config.conditions.items():
        frames[name], hashes, condition_audits[name] = _load_condition(spec, config)
        input_hashes.update(hashes)
    pairwise = run_generic_pairwise_bootstrap(frames, config.pairwise_contrasts, interactions=config.interactions, seeds=config.seeds,
                                              n_replicates=min(config.n_replicates, 1) if validate_only else config.n_replicates,
                                              seed=config.seed, confidence_level=config.confidence_level, metrics=RANKING_METRICS,
                                              expected_rows=config.expected_rows, expected_positives=config.expected_positives, expected_hearings=config.expected_hearings)
    threshold_values: dict[str, dict[int, dict[str, float]]] = {}
    threshold_audits: dict[str, Any] = {}
    for name, spec in config.conditions.items():
        if any(name in pair for pair in config.thresholded_contrasts.values()):
            threshold_values[name], threshold_audits[name], hashes = _load_thresholds(spec, config)
            input_hashes.update(hashes)
    population = pairwise["population"]
    signature_payload = {"schema": "publichearing-final-paired-bootstrap-v1", "code_version": "final-paired-bootstrap-v4",
                         "inputs": input_hashes, "config": config.to_dict(), "population": population}
    signature = _hash_payload(signature_payload)[:16]
    output_dir = (config.output_root / signature).resolve()
    preflight = {"status": "valid", "analysis_signature": signature, "output_dir": str(output_dir), "bootstrap_executed": False,
                 "model_loaded": False, "checkpoint_loaded": False, "cuda_initialized": False, "inference_executed": False,
                 "training_executed": False, "translation_executed": False, "population": population, "conditions": condition_audits,
                 "threshold_audits": threshold_audits, "n_replicates": config.n_replicates, "seed": config.seed}
    if validate_only:
        return preflight
    if output_dir.exists():
        raise FileExistsError(f"Saída já existe; não sobrescrevendo: {output_dir}")
    rows = _summary_rows(config, pairwise)
    threshold_results: dict[str, dict[str, Any]] = {}
    for contrast, (a, b) in config.thresholded_contrasts.items():
        threshold_results[contrast] = run_generic_thresholded_pair_bootstrap(
            frames, threshold_values, a, b, seeds=config.seeds, regimes=config.regimes, n_replicates=config.n_replicates,
            seed=config.seed, confidence_level=config.confidence_level,
        )
        rows.extend(_threshold_rows(config, contrast, a, b, threshold_results[contrast]))
    summary = pd.DataFrame(rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        summary.to_csv(stage / "final_bootstrap_summary.csv", index=False)
        write_json(stage / "population_validation.json", population)
        write_json(stage / "threshold_audit.json", threshold_audits)
        write_json(stage / "condition_audits.json", condition_audits)
        write_json(stage / "resolved_config.json", config.to_dict())
        pairwise["replicates"].to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        (stage / "contrasts").mkdir()
        for contrast, (a, b) in config.pairwise_contrasts.items():
            directory = stage / "contrasts" / contrast
            directory.mkdir()
            write_json(directory / "summary.json", {"contrast_id": contrast, "condition_a": a, "condition_b": b,
                                                    "family": config.contrast_families[contrast], "label": config.contrast_labels[contrast],
                                                    "metrics": pairwise["summaries"][contrast], "observed_campaign": pairwise["observed_campaign"][contrast],
                                                    "protocol": pairwise["protocol"]})
            pairwise["replicates"].loc[pairwise["replicates"].contrast == contrast].to_parquet(directory / "bootstrap_distribution.parquet", index=False)
            _write_contrast_manifest(directory, contrast, config.contrast_families[contrast], config.n_replicates)
        for interaction, (first, second) in config.interactions.items():
            directory = stage / "contrasts" / interaction
            directory.mkdir()
            write_json(directory / "summary.json", {"contrast_id": interaction, "first": first, "second": second,
                                                    "family": config.interaction_families[interaction], "label": config.interaction_labels[interaction],
                                                    "metrics": pairwise["summaries"][interaction], "observed_campaign": pairwise["observed_campaign"][interaction],
                                                    "protocol": pairwise["protocol"]})
            pairwise["replicates"].loc[pairwise["replicates"].contrast == interaction].to_parquet(directory / "bootstrap_distribution.parquet", index=False)
            _write_contrast_manifest(directory, interaction, config.interaction_families[interaction], config.n_replicates)
        (stage / "thresholded").mkdir()
        for contrast, result in threshold_results.items():
            directory = stage / "thresholded" / contrast
            directory.mkdir()
            a, b = config.thresholded_contrasts[contrast]
            write_json(directory / "summary.json", {"contrast_id": contrast, "condition_a": a, "condition_b": b,
                                                    "label": config.thresholded_labels[contrast], "regimes": result["summaries"], "protocol": result["protocol"]})
            result["replicates"].to_parquet(directory / "bootstrap_distribution.parquet", index=False)
            _write_contrast_manifest(directory, contrast, "thresholded_set_attention", config.n_replicates)
        _atomic_text(stage / "final_bootstrap_report.md", _report(config, signature, population, summary, threshold_audits))
        write_json(stage / "run_log.jsonl", {"event": "completed", "seconds": time.time() - started, "bootstrap_executed": True,
                                             "model_loaded": False, "checkpoint_loaded": False, "cuda_initialized": False,
                                             "inference_executed": False, "training_executed": False, "translation_executed": False})
        manifest = {"schema_version": "publichearing-final-paired-bootstrap-v1", "status": "completed", "signature": signature,
                    "signature_payload": signature_payload, "inputs": input_hashes, "population": population,
                    "counts": {"conditions": len(config.conditions), "pairwise_contrasts": len(config.pairwise_contrasts), "interactions": len(config.interactions),
                               "thresholded_contrasts": len(config.thresholded_contrasts), "bootstrap_requested": config.n_replicates}, "artifacts": {}}
        manifest["artifacts"] = {path.relative_to(stage).as_posix(): sha256_file(path) for path in stage.rglob("*") if path.is_file() and path.name != "manifest.json"}
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_contrast_manifest(directory: Path, contrast_id: str, family: str, n_replicates: int) -> None:
    files = {item.name: sha256_file(item) for item in directory.iterdir() if item.is_file() and item.name != "manifest.json"}
    write_json(directory / "manifest.json", {"status": "completed", "contrast_id": contrast_id, "contrast_family": family,
                                             "bootstrap_replicates": n_replicates, "artifacts": files})
