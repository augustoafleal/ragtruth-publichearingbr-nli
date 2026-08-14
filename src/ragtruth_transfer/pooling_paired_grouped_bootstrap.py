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
from .paired_grouped_bootstrap import (
    EXPECTED_HEARINGS,
    EXPECTED_POSITIVES,
    EXPECTED_ROWS,
    EXPECTED_SEEDS,
    _metric_value,
    _summary,
    _validate_manifest,
    _validate_prediction,
)


SCHEMA_VERSION = "publichearing-pooling-paired-grouped-bootstrap-v1"
CODE_VERSION = "pooling-paired-grouped-bootstrap-v1"
METRICS = ("auprc", "auroc", "brier")
EXPECTED_CAMPAIGNS = {
    "gated_attention": "4e12933c51136624",
    "mean": "6bb8a5d4e8041215",
    "max": "53f48ce4154f075e",
}
EXPECTED_COMPARISONS = (
    ("gated_attention", "max"),
    ("gated_attention", "mean"),
    ("mean", "max"),
)
EXPECTED_DATASET_SIGNATURE = "0cdf598fa866741d"
EXPECTED_SPLIT_SIGNATURE = "525edec2966a4fac"


@dataclass(frozen=True)
class CampaignInput:
    name: str
    path: Path
    signature: str
    score_column: str


@dataclass(frozen=True)
class PoolingBootstrapConfig:
    output_root: Path
    campaigns: dict[str, CampaignInput]
    comparisons: tuple[tuple[str, str], ...]
    seeds: tuple[int, ...]
    n_replicates: int
    seed: int
    confidence_level: float
    ci_method: str
    preserve_group_multiplicity: bool
    same_samples_across_methods: bool
    same_samples_across_seeds: bool
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "PoolingBootstrapConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuração inválida: {path}")

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            return candidate.resolve()

        campaign_raw = raw.get("inputs", {}).get("campaigns", {})
        campaigns = {
            str(name): CampaignInput(
                name=str(name),
                path=resolve(value.get("path")),
                signature=str(value.get("expected_signature")),
                score_column=str(value.get("score_column", "probability")),
            )
            for name, value in campaign_raw.items()
            if isinstance(value, dict)
        }
        comparisons = tuple(
            (str(value.get("left")), str(value.get("right")))
            for value in raw.get("comparisons", [])
            if isinstance(value, dict)
        )
        bootstrap = raw.get("bootstrap", {})
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_pooling_paired_bootstrap")),
            campaigns=campaigns,
            comparisons=comparisons,
            seeds=tuple(int(seed) for seed in raw.get("inputs", {}).get("seeds", [0, 1, 2])),
            n_replicates=int(bootstrap.get("n_replicates", 10000)),
            seed=int(bootstrap.get("seed", 20260813)),
            confidence_level=float(bootstrap.get("confidence_level", 0.95)),
            ci_method=str(bootstrap.get("ci_method", "percentile")),
            preserve_group_multiplicity=bool(bootstrap.get("preserve_group_multiplicity", True)),
            same_samples_across_methods=bool(bootstrap.get("same_samples_across_methods", True)),
            same_samples_across_seeds=bool(bootstrap.get("same_samples_across_seeds", True)),
            metrics=tuple(str(metric) for metric in raw.get("metrics", METRICS)),
        )
        config.validate(raw.get("protocol", {}))
        return config

    def validate(self, protocol: dict[str, Any] | None = None) -> None:
        protocol = protocol or {}
        if protocol.get("analysis_type") not in {None, "paired_grouped_bootstrap"}:
            raise ValueError("analysis_type deve ser paired_grouped_bootstrap")
        if protocol.get("group_key", "hearing_id") != "hearing_id" or protocol.get("paired", True) is not True:
            raise ValueError("O protocolo deve ser pareado e agrupado por hearing_id")
        if set(self.campaigns) != set(EXPECTED_CAMPAIGNS):
            raise ValueError(f"Campanhas esperadas: {sorted(EXPECTED_CAMPAIGNS)}")
        if {name: campaign.signature for name, campaign in self.campaigns.items()} != EXPECTED_CAMPAIGNS:
            raise ValueError("Assinaturas das campanhas de pooling não correspondem aos runs congelados")
        if any(campaign.score_column != "probability" for campaign in self.campaigns.values()):
            raise ValueError("As comparações de pooling exigem a coluna probability")
        if self.comparisons != EXPECTED_COMPARISONS:
            raise ValueError(f"Comparações esperadas: {EXPECTED_COMPARISONS}")
        if self.seeds != EXPECTED_SEEDS:
            raise ValueError("A análise exige exatamente as seeds [0, 1, 2]")
        if self.metrics != METRICS:
            raise ValueError(f"Métricas esperadas: {METRICS}")
        if self.n_replicates < 1 or self.seed < 0 or not 0 < self.confidence_level < 1 or self.ci_method != "percentile":
            raise ValueError("Configuração de bootstrap inválida")
        if not self.preserve_group_multiplicity or not self.same_samples_across_methods or not self.same_samples_across_seeds:
            raise ValueError("O protocolo exige multiplicidade e amostras compartilhadas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {"name": "publichearing_pooling_ablation_paired_bootstrap", "analysis_type": "paired_grouped_bootstrap", "group_key": "hearing_id", "paired": True},
            "inputs": {
                "campaigns": {
                    name: {"path": str(campaign.path), "expected_signature": campaign.signature, "score_column": campaign.score_column}
                    for name, campaign in self.campaigns.items()
                },
                "seeds": list(self.seeds),
            },
            "comparisons": [{"left": left, "right": right} for left, right in self.comparisons],
            "bootstrap": {
                "n_replicates": self.n_replicates,
                "seed": self.seed,
                "confidence_level": self.confidence_level,
                "ci_method": self.ci_method,
                "preserve_group_multiplicity": self.preserve_group_multiplicity,
                "same_samples_across_methods": self.same_samples_across_methods,
                "same_samples_across_seeds": self.same_samples_across_seeds,
            },
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


def _validate_campaign_manifest(campaign: CampaignInput) -> dict[str, Any]:
    manifest = _validate_manifest(campaign.path, campaign.signature)
    data_audit = manifest.get("data_audit", {})
    if data_audit.get("metadata", {}).get("signature") != EXPECTED_DATASET_SIGNATURE:
        raise ValueError(f"Dataset RAGTruth divergente em {campaign.name}")
    if data_audit.get("split", {}).get("signature") != EXPECTED_SPLIT_SIGNATURE:
        raise ValueError(f"Split RAGTruth divergente em {campaign.name}")
    pooling_type = str(manifest.get("pooling_type", "gated_attention"))
    if pooling_type != campaign.name:
        raise ValueError(f"Pooling divergente em {campaign.name}: {pooling_type}")
    return manifest


def _locate_seed_prediction(campaign: CampaignInput, seed: int) -> tuple[Path, dict[str, Any], str]:
    seed_root = campaign.path / f"seed_{seed}" / "publichearing_zero_shot"
    manifests = sorted(seed_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"Esperava um manifesto zero-shot em {campaign.name}, seed {seed}; encontrei {len(manifests)}")
    run_dir = manifests[0].parent
    manifest = _validate_manifest(run_dir, run_dir.name)
    prediction = run_dir / "predictions.parquet"
    if not prediction.is_file():
        raise FileNotFoundError(f"Predição ausente: {prediction}")
    digest = sha256_file(prediction)
    expected_hash = manifest.get("artifacts", {}).get("predictions.parquet")
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Hash da predição divergente: {prediction}")
    return prediction, manifest, digest


def _strict_join_campaigns(frames: dict[str, dict[int, pd.DataFrame]], config: PoolingBootstrapConfig) -> tuple[pd.DataFrame, dict[str, Any], str]:
    reference_name = "gated_attention"
    reference = frames[reference_name][config.seeds[0]][["example_id", "hearing_id", "label"]].copy()
    reference["example_id"] = reference["example_id"].astype(str)
    if not reference["example_id"].is_unique:
        raise ValueError("example_id duplicado na referência")
    reference_ids = set(reference["example_id"])
    audit: dict[str, Any] = {}
    publichearing_signatures: set[str] = set()
    for name, per_seed in frames.items():
        audit[name] = {}
        for seed, frame in per_seed.items():
            current = frame[["example_id", "hearing_id", "label", "probability", "publichearing_dataset_signature", "model_run_signature"]].copy()
            current["example_id"] = current["example_id"].astype(str)
            if not current["example_id"].is_unique:
                raise ValueError(f"example_id duplicado em {name}, seed {seed}")
            ids = set(current["example_id"])
            left = reference.set_index("example_id").loc[sorted(reference_ids & ids)]
            right = current.set_index("example_id").loc[sorted(reference_ids & ids)]
            label_mismatches = int((left["label"].astype(int) != right["label"].astype(int)).sum())
            hearing_mismatches = int((left["hearing_id"].astype(str) != right["hearing_id"].astype(str)).sum())
            run_signatures = set(current["model_run_signature"].astype(str).unique())
            audit[name][str(seed)] = {
                "reference_only_ids": len(reference_ids - ids),
                "campaign_only_ids": len(ids - reference_ids),
                "label_mismatches": label_mismatches,
                "hearing_id_mismatches": hearing_mismatches,
                "duplicate_ids": int((not reference["example_id"].is_unique) or (not current["example_id"].is_unique)),
                "pairable_examples": len(reference_ids & ids),
                "model_run_signatures": sorted(run_signatures),
            }
            if reference_ids - ids or ids - reference_ids or label_mismatches or hearing_mismatches:
                raise ValueError(f"Previsões não pareáveis em {name}, seed {seed}")
            if len(run_signatures) != 1:
                raise ValueError(f"model_run_signature inconsistente em {name}, seed {seed}")
            signatures = set(current["publichearing_dataset_signature"].astype(str).unique())
            if len(signatures) != 1:
                raise ValueError(f"Assinatura PublicHearingBR inconsistente em {name}, seed {seed}")
            publichearing_signatures.update(signatures)
    if len(publichearing_signatures) != 1:
        raise ValueError("As campanhas não usaram o mesmo dataset PublicHearingBR")
    joined = reference.sort_values("example_id").reset_index(drop=True)
    for name, per_seed in frames.items():
        for seed, frame in per_seed.items():
            current = frame[["example_id", "probability"]].copy()
            current["example_id"] = current["example_id"].astype(str)
            column = f"{name}_seed_{seed}"
            joined = joined.merge(current, on="example_id", how="left", validate="one_to_one").rename(columns={"probability": column})
    score_columns = [f"{name}_seed_{seed}" for name in config.campaigns for seed in config.seeds]
    if joined[score_columns].isna().any().any():
        raise ValueError("Join deixou scores ausentes")
    return joined, audit, next(iter(publichearing_signatures))


def _input_signature(config: PoolingBootstrapConfig, input_hashes: dict[str, str], joined: pd.DataFrame, publichearing_signature: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "campaign_signatures": {name: campaign.signature for name, campaign in config.campaigns.items()},
        "input_hashes": input_hashes,
        "seeds": list(config.seeds),
        "comparisons": [{"left": left, "right": right, "delta_definition": "left - right"} for left, right in config.comparisons],
        "example_ids_hash": _canonical_hash(joined.example_id.astype(str).tolist()),
        "hearing_ids_hash": _canonical_hash(sorted(joined.hearing_id.astype(str).unique().tolist())),
        "publichearing_dataset_signature": publichearing_signature,
        "ragtruth_dataset_signature": EXPECTED_DATASET_SIGNATURE,
        "ragtruth_split_signature": EXPECTED_SPLIT_SIGNATURE,
        "score_column": "probability",
        "metrics": list(config.metrics),
        "n_replicates": config.n_replicates,
        "bootstrap_seed": config.seed,
        "confidence_level": config.confidence_level,
        "ci_method": config.ci_method,
        "group_key": "hearing_id",
        "preserve_group_multiplicity": config.preserve_group_multiplicity,
        "same_samples_across_methods": config.same_samples_across_methods,
        "same_samples_across_seeds": config.same_samples_across_seeds,
    }
    return _canonical_hash(payload)[:16], payload


def _observed_metrics(labels: np.ndarray, joined: pd.DataFrame, config: PoolingBootstrapConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    observed: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for name in config.campaigns:
        per_seed = {
            str(seed): {metric: _metric_value(metric, labels, joined[f"{name}_seed_{seed}"].to_numpy(dtype=float)) for metric in config.metrics}
            for seed in config.seeds
        }
        observed[name] = {
            "per_seed": per_seed,
            "mean_across_seeds": {metric: float(np.mean([per_seed[str(seed)][metric] for seed in config.seeds])) for metric in config.metrics},
        }
    for left, right in config.comparisons:
        key = f"{left}_vs_{right}"
        comparisons[key] = {
            "left": left,
            "right": right,
            "delta_definition": "left - right",
            "per_seed_delta": {str(seed): {metric: float(observed[left]["per_seed"][str(seed)][metric] - observed[right]["per_seed"][str(seed)][metric]) for metric in config.metrics} for seed in config.seeds},
            "mean_delta_across_seeds": {metric: float(observed[left]["mean_across_seeds"][metric] - observed[right]["mean_across_seeds"][metric]) for metric in config.metrics},
        }
    return observed, comparisons


def _with_ci_interpretation(summary: dict[str, Any], metric: str) -> dict[str, Any]:
    result = dict(summary)
    result["ci_crosses_zero"] = bool(result["ci_lower"] <= 0 <= result["ci_upper"])
    result["left_favorable_direction"] = "positive" if metric in {"auprc", "auroc"} else "negative"
    return result


def _build_summaries(replicates: pd.DataFrame, config: PoolingBootstrapConfig, observed: dict[str, Any], observed_comparisons: dict[str, Any]) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    reference = replicates[(replicates["comparison"] == "gated_attention_vs_max") & (replicates["seed"] == config.seeds[0]) & (replicates["metric"] == config.metrics[0])]
    valid_count = int(reference["valid"].sum())
    invalid_count = config.n_replicates - valid_count
    invalid_reasons = {str(key): int(value) for key, value in reference.loc[~reference["valid"], "invalid_reason"].value_counts().items()}
    results: dict[str, Any] = {}
    for left, right in config.comparisons:
        key = f"{left}_vs_{right}"
        per_seed: dict[str, Any] = {}
        for seed in config.seeds:
            per_seed[str(seed)] = {
                metric: _with_ci_interpretation(
                    _summary(
                        replicates.loc[(replicates["comparison"] == key) & (replicates["seed"] == seed) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float),
                        config.confidence_level,
                        replicates.loc[(replicates["comparison"] == key) & (replicates["seed"] == seed) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float) > 0 if metric in {"auprc", "auroc"} else replicates.loc[(replicates["comparison"] == key) & (replicates["seed"] == seed) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float) < 0,
                    ),
                    metric,
                )
                for metric in config.metrics
            }
        campaign: dict[str, Any] = {}
        for metric in config.metrics:
            values = replicates.loc[(replicates["comparison"] == key) & (replicates["seed"] == -1) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float)
            campaign[metric] = _with_ci_interpretation(_summary(values, config.confidence_level, values > 0 if metric in {"auprc", "auroc"} else values < 0), metric)
        results[key] = {
            "left": left,
            "right": right,
            "delta_definition": "left - right",
            "observed": {
                "left_mean_across_seeds": observed[left]["mean_across_seeds"],
                "right_mean_across_seeds": observed[right]["mean_across_seeds"],
                "delta_mean_across_seeds": observed_comparisons[key]["mean_delta_across_seeds"],
            },
            "per_seed": per_seed,
            "campaign": campaign,
        }
    return results, valid_count, invalid_count, invalid_reasons


def _report(signature: str, config: PoolingBootstrapConfig, results: dict[str, Any], valid: int, invalid: int, seconds: float) -> str:
    lines = [
        "# Pooling paired grouped bootstrap",
        "",
        f"- Analysis signature: `{signature}`",
        f"- Replicates requested: {config.n_replicates}; valid: {valid}; invalid: {invalid}",
        "- Resampling unit: `hearing_id`, with replacement and group multiplicity preserved.",
        "- The same sampled hearings are used for all poolings and seeds. Campaign effects are means over seeds 0, 1, and 2.",
        "- Delta is always `left - right`; positive is favorable to the left only for AUPRC/AUROC, while negative is favorable to the left for Brier.",
        "",
        "## Campaign-level paired effects",
        "",
        "| Comparison | Metric | Left point | Right point | Delta | 95% CI | CI crosses zero |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for key, comparison in results.items():
        for metric, summary in comparison["campaign"].items():
            observed = comparison["observed"]
            lines.append(
                f"| {key} | {metric} | {observed['left_mean_across_seeds'][metric]:.6f} | {observed['right_mean_across_seeds'][metric]:.6f} | {observed['delta_mean_across_seeds'][metric]:+.6f} | [{summary['ci_lower']:+.6f}, {summary['ci_upper']:+.6f}] | {'yes' if summary['ci_crosses_zero'] else 'no'} |"
            )
    lines += ["", f"Execution time: {seconds:.3f} s.", "No model, tokenizer, threshold, ensemble, or checkpoint selection was used by this analysis."]
    return "\n".join(lines) + "\n"


def _verify_completed(output_dir: Path, signature: str) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("signature") != signature or manifest.get("status") != "completed":
        raise ValueError("Manifesto de análise incompatível")
    for name, expected in manifest.get("artifacts", {}).items():
        artifact = output_dir / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Artefato ausente/corrompido: {artifact}")
    return manifest


def run_pooling_bootstrap(config: PoolingBootstrapConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    frames: dict[str, dict[int, pd.DataFrame]] = {}
    input_hashes: dict[str, str] = {}
    input_paths: dict[str, dict[str, str]] = {}
    for name, campaign in config.campaigns.items():
        _validate_campaign_manifest(campaign)
        frames[name] = {}
        input_paths[name] = {}
        for seed in config.seeds:
            prediction_path, _, digest = _locate_seed_prediction(campaign, seed)
            frames[name][seed] = _validate_prediction(prediction_path, campaign.score_column)
            input_hashes[f"{name}_seed_{seed}_predictions.parquet"] = digest
            input_paths[name][str(seed)] = str(prediction_path)
    joined, pairing, publichearing_signature = _strict_join_campaigns(frames, config)
    signature, signature_payload = _input_signature(config, input_hashes, joined, publichearing_signature)
    labels = joined["label"].to_numpy(dtype=int)
    observed, observed_comparisons = _observed_metrics(labels, joined, config)
    output_dir = (config.output_root / signature).resolve()
    common_result = {
        "n_examples": len(joined),
        "n_hearings": int(joined["hearing_id"].nunique()),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "seeds": list(config.seeds),
        "comparisons": [f"{left}_vs_{right}" for left, right in config.comparisons],
        "n_bootstrap": config.n_replicates,
        "group_key": "hearing_id",
        "analysis_signature": signature,
        "output_dir": str(output_dir),
        "pairing": pairing,
        "input_hashes": input_hashes,
    }
    if validate_only:
        return {"status": "valid", "analysis_executed": False, "bootstrap_executed": False, "models_loaded": False, "cuda_initialized": False, **common_result}
    if output_dir.exists():
        if resume:
            return _verify_completed(output_dir, signature)
        raise FileExistsError(f"Saída já existe; use --resume: {output_dir}")

    group_values = joined["hearing_id"].astype(str).to_numpy()
    groups = sorted(np.unique(group_values).tolist())
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    scores = {name: {seed: joined[f"{name}_seed_{seed}"].to_numpy(dtype=float) for seed in config.seeds} for name in config.campaigns}
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    for replicate_id in range(config.n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([group_indices[position] for position in sampled_positions])
        sampled_labels = labels[indices]
        valid = np.unique(sampled_labels).size == 2
        common = {
            "replicate_id": replicate_id,
            "bootstrap_seed": config.seed,
            "valid": bool(valid),
            "invalid_reason": None if valid else "single_class_resample",
            "n_rows": int(len(indices)),
            "n_sampled_groups": len(groups),
            "n_unique_groups": int(len(set(sampled_positions.tolist()))),
            "n_positive": int(sampled_labels.sum()),
            "n_negative": int((sampled_labels == 0).sum()),
        }
        sampled_values: dict[str, dict[int, dict[str, float]]] = {}
        if valid:
            for name in config.campaigns:
                sampled_values[name] = {}
                for seed in config.seeds:
                    sampled_values[name][seed] = {metric: _metric_value(metric, sampled_labels, scores[name][seed][indices]) for metric in config.metrics}
        for left, right in config.comparisons:
            key = f"{left}_vs_{right}"
            campaign_deltas = {metric: 0.0 for metric in config.metrics}
            campaign_left = {metric: 0.0 for metric in config.metrics}
            campaign_right = {metric: 0.0 for metric in config.metrics}
            for seed in config.seeds:
                for metric in config.metrics:
                    left_value = sampled_values[left][seed][metric] if valid else None
                    right_value = sampled_values[right][seed][metric] if valid else None
                    delta = float(left_value - right_value) if valid else None
                    records.append({**common, "comparison": key, "left": left, "right": right, "seed": seed, "seed_label": f"seed_{seed}", "metric": metric, "left_value": left_value, "right_value": right_value, "delta": delta})
                    if valid:
                        campaign_deltas[metric] += delta / len(config.seeds)
                        campaign_left[metric] += left_value / len(config.seeds)
                        campaign_right[metric] += right_value / len(config.seeds)
            for metric in config.metrics:
                records.append({**common, "comparison": key, "left": left, "right": right, "seed": -1, "seed_label": "mean_seed_delta", "metric": metric, "left_value": campaign_left[metric] if valid else None, "right_value": campaign_right[metric] if valid else None, "delta": campaign_deltas[metric] if valid else None})
    replicate_frame = pd.DataFrame.from_records(records)
    results, valid_count, invalid_count, invalid_reasons = _build_summaries(replicate_frame, config, observed, observed_comparisons)
    reloaded_hashes = {key: sha256_file(Path(path)) for name, seeds in input_paths.items() for seed, path in seeds.items() for key in [f"{name}_seed_{seed}_predictions.parquet"]}
    if reloaded_hashes != input_hashes:
        raise RuntimeError("Uma das predições de entrada foi alterada durante a análise")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        replicate_frame.to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        write_json(stage / "observed_metrics.json", {"campaigns": observed, "comparisons": observed_comparisons})
        write_json(stage / "comparison_summary.json", results)
        write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {
            "inputs": input_hashes,
            "inputs_reloaded_hashes": reloaded_hashes,
            "input_paths": input_paths,
            "pairing": pairing,
            "publichearing_dataset_signature": publichearing_signature,
            "ragtruth_dataset_signature": EXPECTED_DATASET_SIGNATURE,
            "ragtruth_split_signature": EXPECTED_SPLIT_SIGNATURE,
            "n_examples": len(joined),
            "n_hearings": len(groups),
            "n_bootstrap_requested": config.n_replicates,
            "n_bootstrap_valid": valid_count,
            "n_bootstrap_invalid": invalid_count,
            "invalid_reasons": invalid_reasons,
            "paired": True,
            "grouped": True,
            "group_multiplicity_preserved": True,
            "same_indices_across_methods": True,
            "same_indices_across_seeds": True,
            "models_loaded": False,
            "tokenizers_loaded": False,
            "cuda_initialized": False,
        }
        write_json(stage / "integrity_audit.json", integrity)
        _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "seconds": time.time() - started, "bootstrap_executed": True, "models_loaded": False, "bootstrap_seed": config.seed}, ensure_ascii=False) + "\n")
        _atomic_text(stage / "report.md", _report(signature, config, results, valid_count, invalid_count, time.time() - started))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "signature": signature,
            "signature_payload": signature_payload,
            "inputs": {"campaign_signatures": {name: campaign.signature for name, campaign in config.campaigns.items()}, "hashes": input_hashes},
            "counts": {"examples": len(joined), "hearings": len(groups), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "bootstrap_requested": config.n_replicates, "bootstrap_valid": valid_count, "bootstrap_invalid": invalid_count},
            "artifacts": {},
        }
        manifest["artifacts"] = _file_hashes(stage)
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
