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
from pandas.api.types import is_integer_dtype

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


SCHEMA_VERSION = "publichearing-set-transformer-paired-grouped-bootstrap-v1"
CODE_VERSION = "set-transformer-paired-grouped-bootstrap-v1"
METRICS = ("auprc", "auroc", "brier")
EXPECTED_CAMPAIGNS = {
    "gated_attention": "4e12933c51136624",
    "set_transformer": "28323e6cca11feb6",
}
EXPECTED_DATASET_SIGNATURE = "0cdf598fa866741d"
EXPECTED_SPLIT_SIGNATURE = "525edec2966a4fac"


@dataclass(frozen=True)
class CampaignInput:
    name: str
    path: Path
    signature: str
    score_column: str


@dataclass(frozen=True)
class SetTransformerBootstrapConfig:
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
    def from_yaml(cls, path: Path) -> "SetTransformerBootstrapConfig":
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
        bootstrap = raw.get("bootstrap", {})
        config = cls(
            output_root=resolve(raw.get("output", {}).get("root", "../runs/publichearing_set_transformer_paired_bootstrap")),
            campaigns=campaigns,
            comparisons=tuple(
                (str(item.get("left")), str(item.get("right")))
                for item in raw.get("comparisons", [])
                if isinstance(item, dict)
            ),
            seeds=tuple(int(seed) for seed in raw.get("inputs", {}).get("seeds", [0, 1, 2])),
            n_replicates=int(bootstrap.get("n_replicates", 10000)),
            seed=int(bootstrap.get("seed", 20260815)),
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
            raise ValueError("Assinaturas das campanhas Gated/Set não correspondem aos runs esperados")
        if any(campaign.score_column != "probability" for campaign in self.campaigns.values()):
            raise ValueError("As duas campanhas devem usar a coluna probability")
        if self.comparisons != (("set_transformer", "gated_attention"),):
            raise ValueError("A comparação deve ser Set Transformer − Gated Attention")
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
            "protocol": {"name": "publichearing_set_transformer_paired_grouped_bootstrap", "analysis_type": "paired_grouped_bootstrap", "group_key": "hearing_id", "paired": True},
            "inputs": {
                "campaigns": {
                    name: {"path": str(campaign.path), "expected_signature": campaign.signature, "score_column": campaign.score_column}
                    for name, campaign in self.campaigns.items()
                },
                "seeds": list(self.seeds),
            },
            "comparisons": [{"left": left, "right": right, "delta_definition": "left - right"} for left, right in self.comparisons],
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
            "delta_convention": "set_transformer - gated_attention for auprc, auroc, and brier; negative brier favors Set Transformer",
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


def _campaign_pooling_type(campaign: CampaignInput, manifest: dict[str, Any]) -> str:
    if campaign.name == "gated_attention":
        # Historical manifests vary between architecture=gated_attention and pooling=attention.
        resolved = campaign.path / "resolved_config.json"
        architecture = json.loads(resolved.read_text(encoding="utf-8")).get("architecture") if resolved.is_file() else None
        pooling = str(manifest.get("pooling_type", ""))
        if architecture in {"gated_attention", "attention"} or pooling in {"gated_attention", "attention"}:
            return "gated_attention"
    if campaign.name == "set_transformer" and str(manifest.get("pooling_type")) == "set_transformer":
        return "set_transformer"
    raise ValueError(f"Alias/pooling incompatível para {campaign.name}")


def _validate_campaign_manifest(campaign: CampaignInput) -> dict[str, Any]:
    manifest = _validate_manifest(campaign.path, campaign.signature)
    data_audit = manifest.get("data_audit", {})
    if data_audit.get("metadata", {}).get("signature") != EXPECTED_DATASET_SIGNATURE:
        raise ValueError(f"Dataset RAGTruth divergente em {campaign.name}")
    if data_audit.get("split", {}).get("signature") != EXPECTED_SPLIT_SIGNATURE:
        raise ValueError(f"Split RAGTruth divergente em {campaign.name}")
    _campaign_pooling_type(campaign, manifest)
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
    frame = _validate_prediction(prediction, campaign.score_column)
    _validate_prediction_seed_column(frame, campaign, seed)
    return prediction, manifest, digest


def _validate_prediction_seed_column(frame: pd.DataFrame, campaign: CampaignInput, seed: int) -> None:
    if campaign.name == "set_transformer":
        if "seed" not in frame.columns or not is_integer_dtype(frame["seed"]):
            raise ValueError(f"Set Transformer seed {seed} não possui coluna seed inteira")
        if set(frame["seed"].astype(int).unique()) != {seed}:
            raise ValueError(f"Coluna seed divergente em Set Transformer seed {seed}")
    elif "seed" in frame.columns and set(frame["seed"].astype(int).unique()) != {seed}:
        raise ValueError(f"Coluna seed histórica divergente em Gated seed {seed}")


def _strict_join_campaigns(
    frames: dict[str, dict[int, pd.DataFrame]], config: SetTransformerBootstrapConfig
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    reference = frames["gated_attention"][config.seeds[0]][["example_id", "hearing_id", "label"]].copy()
    reference["example_id"] = reference["example_id"].astype(str)
    if not reference["example_id"].is_unique:
        raise ValueError("example_id duplicado na referência Gated")
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
            common = sorted(reference_ids & ids)
            left = reference.set_index("example_id").loc[common]
            right = current.set_index("example_id").loc[common]
            label_mismatches = int((left["label"].astype(int) != right["label"].astype(int)).sum())
            hearing_mismatches = int((left["hearing_id"].astype(str) != right["hearing_id"].astype(str)).sum())
            signatures = set(current["publichearing_dataset_signature"].astype(str).unique())
            run_signatures = set(current["model_run_signature"].astype(str).unique())
            seed_status = "explicit" if "seed" in frame.columns else "historical_path"
            audit[name][str(seed)] = {
                "seed_identification": seed_status,
                "reference_only_ids": len(reference_ids - ids),
                "campaign_only_ids": len(ids - reference_ids),
                "label_mismatches": label_mismatches,
                "hearing_id_mismatches": hearing_mismatches,
                "duplicate_ids": int(not current["example_id"].is_unique),
                "pairable_examples": len(common),
                "model_run_signatures": sorted(run_signatures),
                "publichearing_dataset_signatures": sorted(signatures),
            }
            if reference_ids - ids or ids - reference_ids or label_mismatches or hearing_mismatches:
                raise ValueError(f"Previsões não pareáveis em {name}, seed {seed}")
            if len(run_signatures) != 1 or len(signatures) != 1:
                raise ValueError(f"Signatures inconsistentes em {name}, seed {seed}")
            publichearing_signatures.update(signatures)
    if len(publichearing_signatures) != 1:
        raise ValueError("As campanhas não usaram o mesmo dataset PublicHearingBR")
    joined = reference.sort_values("example_id").reset_index(drop=True)
    for name, per_seed in frames.items():
        for seed, frame in per_seed.items():
            current = frame[["example_id", "probability"]].copy()
            current["example_id"] = current["example_id"].astype(str)
            joined = joined.merge(current, on="example_id", how="left", validate="one_to_one").rename(columns={"probability": f"{name}_seed_{seed}"})
    score_columns = [f"{name}_seed_{seed}" for name in config.campaigns for seed in config.seeds]
    if joined[score_columns].isna().any().any():
        raise ValueError("Join deixou scores ausentes")
    return joined, audit, next(iter(publichearing_signatures))


def _input_signature(config: SetTransformerBootstrapConfig, input_hashes: dict[str, str], joined: pd.DataFrame, publichearing_signature: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "campaign_signatures": {name: campaign.signature for name, campaign in config.campaigns.items()},
        "input_hashes": input_hashes,
        "seeds": list(config.seeds),
        "comparisons": [{"left": left, "right": right, "delta_definition": "set_transformer - gated_attention"} for left, right in config.comparisons],
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


def _observed_metrics(labels: np.ndarray, joined: pd.DataFrame, config: SetTransformerBootstrapConfig) -> tuple[dict[str, Any], dict[str, Any]]:
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
            "delta_definition": "set_transformer - gated_attention",
            "per_seed_delta": {
                str(seed): {metric: float(observed[left]["per_seed"][str(seed)][metric] - observed[right]["per_seed"][str(seed)][metric]) for metric in config.metrics}
                for seed in config.seeds
            },
            "mean_delta_across_seeds": {
                metric: float(observed[left]["mean_across_seeds"][metric] - observed[right]["mean_across_seeds"][metric])
                for metric in config.metrics
            },
        }
    return observed, comparisons


def _interpret(summary: dict[str, Any], metric: str) -> dict[str, Any]:
    result = dict(summary)
    result["ci_crosses_zero"] = bool(result["ci_lower"] <= 0 <= result["ci_upper"])
    result["set_transformer_favorable_direction"] = "positive" if metric in {"auprc", "auroc"} else "negative"
    if result["ci_crosses_zero"]:
        result["conclusion"] = "descriptive/inconclusive"
    elif (metric in {"auprc", "auroc"} and result["ci_lower"] > 0) or (metric == "brier" and result["ci_upper"] < 0):
        result["conclusion"] = "evidence_favorable_to_set_transformer"
    else:
        result["conclusion"] = "evidence_favorable_to_gated_attention"
    return result


def _build_summary(replicates: pd.DataFrame, config: SetTransformerBootstrapConfig, observed: dict[str, Any], observed_comparisons: dict[str, Any]) -> dict[str, Any]:
    key = "set_transformer_vs_gated_attention"
    result: dict[str, Any] = {"comparison": key, "left": "set_transformer", "right": "gated_attention", "delta_definition": "set_transformer - gated_attention", "observed": {"gated_attention": observed["gated_attention"]["mean_across_seeds"], "set_transformer": observed["set_transformer"]["mean_across_seeds"], "delta_set_minus_gated": observed_comparisons[key]["mean_delta_across_seeds"]}, "per_seed": {}, "campaign": {}}
    for seed in config.seeds:
        result["per_seed"][str(seed)] = {}
        for metric in config.metrics:
            values = replicates.loc[(replicates["seed"] == seed) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float)
            result["per_seed"][str(seed)][metric] = _interpret(_summary(values, config.confidence_level, values > 0 if metric != "brier" else values < 0), metric)
    for metric in config.metrics:
        values = replicates.loc[(replicates["seed"] == -1) & (replicates["metric"] == metric) & replicates["valid"], "delta"].dropna().to_numpy(dtype=float)
        result["campaign"][metric] = _interpret(_summary(values, config.confidence_level, values > 0 if metric != "brier" else values < 0), metric)
    return result


def _report(signature: str, config: SetTransformerBootstrapConfig, summary: dict[str, Any], valid: int, invalid: int, seconds: float) -> str:
    observed = summary["observed"]
    lines = [
        "# Gated Attention × Set Transformer paired grouped bootstrap",
        "",
        f"- Analysis signature: `{signature}`",
        f"- Réplicas solicitadas: {config.n_replicates}; válidas: {valid}; inválidas: {invalid}",
        "- Unidade: `hearing_id`, com reposição e multiplicidade preservada.",
        "- A mesma amostra bootstrap foi aplicada aos dois modelos e às três seeds.",
        "- Delta: `Set Transformer - Gated Attention`.",
        "- Para Brier, delta negativo favorece Set Transformer porque menor Brier é melhor.",
        "",
        "## Resultado inferencial da campanha",
        "",
        "| Métrica | Gated | Set Transformer | Delta Set − Gated | IC 95% | Cruza zero? | Conclusão |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for metric in config.metrics:
        item = summary["campaign"][metric]
        lines.append(f"| {metric} | {observed['gated_attention'][metric]:.6f} | {observed['set_transformer'][metric]:.6f} | {observed['delta_set_minus_gated'][metric]:+.6f} | [{item['ci_lower']:+.6f}, {item['ci_upper']:+.6f}] | {'sim' if item['ci_crosses_zero'] else 'não'} | {item['conclusion']} |")
    lines += [
        "",
        "A conclusão principal depende de AUPRC. Não há linguagem de superioridade quando o IC cruza zero.",
        "",
        f"Tempo de execução: {seconds:.3f} s.",
        "Nenhum modelo, tokenizer, threshold, ensemble ou treino foi usado.",
    ]
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


def _generate_replicates(joined: pd.DataFrame, config: SetTransformerBootstrapConfig) -> pd.DataFrame:
    labels = joined["label"].to_numpy(dtype=int)
    groups = sorted(joined["hearing_id"].astype(str).unique().tolist())
    group_values = joined["hearing_id"].astype(str).to_numpy()
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    scores = {name: {seed: joined[f"{name}_seed_{seed}"].to_numpy(dtype=float) for seed in config.seeds} for name in config.campaigns}
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    key = "set_transformer_vs_gated_attention"
    for replicate_id in range(config.n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([group_indices[position] for position in sampled_positions])
        sampled_labels = labels[indices]
        valid = np.unique(sampled_labels).size == 2
        common_record = {"replicate_id": replicate_id, "bootstrap_seed": config.seed, "valid": bool(valid), "invalid_reason": None if valid else "single_class_resample", "n_rows": int(len(indices)), "n_sampled_groups": len(groups), "n_unique_groups": int(len(set(sampled_positions.tolist()))), "sampled_groups_hash": _canonical_hash(sampled_positions.tolist()), "n_positive": int(sampled_labels.sum()), "n_negative": int((sampled_labels == 0).sum()), "comparison": key}
        mean_values: dict[str, list[tuple[float, float, float]]] = {metric: [] for metric in config.metrics}
        for seed in config.seeds:
            for metric in config.metrics:
                if valid:
                    left_value = _metric_value(metric, sampled_labels, scores["set_transformer"][seed][indices])
                    right_value = _metric_value(metric, sampled_labels, scores["gated_attention"][seed][indices])
                    delta = float(left_value - right_value)
                    mean_values[metric].append((left_value, right_value, delta))
                else:
                    left_value = right_value = delta = None
                records.append({**common_record, "left": "set_transformer", "right": "gated_attention", "seed": seed, "seed_label": f"seed_{seed}", "metric": metric, "left_value": left_value, "right_value": right_value, "delta": delta})
        for metric in config.metrics:
            values = mean_values[metric]
            records.append({**common_record, "left": "set_transformer", "right": "gated_attention", "seed": -1, "seed_label": "mean_seed_delta", "metric": metric, "left_value": float(np.mean([item[0] for item in values])) if valid else None, "right_value": float(np.mean([item[1] for item in values])) if valid else None, "delta": float(np.mean([item[2] for item in values])) if valid else None})
    return pd.DataFrame.from_records(records)


def run_set_transformer_bootstrap(config: SetTransformerBootstrapConfig, *, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
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
    labels = joined["label"].to_numpy(dtype=int)
    signature, signature_payload = _input_signature(config, input_hashes, joined, publichearing_signature)
    observed, observed_comparisons = _observed_metrics(labels, joined, config)
    output_dir = (config.output_root / signature).resolve()
    common = {
        "n_examples": len(joined),
        "n_hearings": int(joined["hearing_id"].nunique()),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "seeds": list(config.seeds),
        "n_bootstrap": config.n_replicates,
        "group_key": "hearing_id",
        "analysis_signature": signature,
        "output_dir": str(output_dir),
        "pairing": pairing,
        "input_hashes": input_hashes,
        "campaign_signatures": {name: campaign.signature for name, campaign in config.campaigns.items()},
        "publichearing_dataset_signature": publichearing_signature,
    }
    if validate_only:
        return {"status": "valid", "analysis_executed": False, "bootstrap_executed": False, "models_loaded": False, "cuda_initialized": False, **common}
    if output_dir.exists():
        if resume:
            return _verify_completed(output_dir, signature)
        raise FileExistsError(f"Saída já existe; use --resume: {output_dir}")

    groups = sorted(joined["hearing_id"].astype(str).unique().tolist())
    replicate_frame = _generate_replicates(joined, config)
    summary = _build_summary(replicate_frame, config, observed, observed_comparisons)
    reference = replicate_frame[(replicate_frame["seed"] == config.seeds[0]) & (replicate_frame["metric"] == config.metrics[0])]
    valid_count = int(reference["valid"].sum())
    invalid_count = config.n_replicates - valid_count
    invalid_reasons = {str(key): int(value) for key, value in reference.loc[~reference["valid"], "invalid_reason"].value_counts().items()}
    input_hashes_after = {key: sha256_file(Path(input_paths[name][str(seed)])) for name in input_paths for seed in input_paths[name] for key in [f"{name}_seed_{seed}_predictions.parquet"]}
    if input_hashes_after != input_hashes:
        raise RuntimeError("Uma das predições de entrada foi alterada durante a análise")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        replicate_frame.to_parquet(stage / "bootstrap_replicates.parquet", index=False)
        write_json(stage / "observed_metrics.json", {"campaigns": observed, "comparisons": observed_comparisons})
        write_json(stage / "comparison_summary.json", summary)
        write_json(stage / "resolved_config.json", config.to_dict())
        integrity = {"inputs": input_hashes, "inputs_reloaded_hashes": input_hashes_after, "input_paths": input_paths, "pairing": pairing, "publichearing_dataset_signature": publichearing_signature, "ragtruth_dataset_signature": EXPECTED_DATASET_SIGNATURE, "ragtruth_split_signature": EXPECTED_SPLIT_SIGNATURE, "n_examples": len(joined), "n_hearings": len(groups), "n_bootstrap_requested": config.n_replicates, "n_bootstrap_valid": valid_count, "n_bootstrap_invalid": invalid_count, "invalid_reasons": invalid_reasons, "paired": True, "grouped": True, "group_multiplicity_preserved": True, "same_indices_across_methods": True, "same_indices_across_seeds": True, "models_loaded": False, "tokenizers_loaded": False, "cuda_initialized": False, "delta_convention": "set_transformer - gated_attention"}
        write_json(stage / "integrity_audit.json", integrity)
        _atomic_text(stage / "run_log.jsonl", json.dumps({"event": "completed", "seconds": time.time() - started, "bootstrap_executed": True, "models_loaded": False, "cuda_initialized": False, "bootstrap_seed": config.seed}, ensure_ascii=False) + "\n")
        _atomic_text(stage / "report.md", _report(signature, config, summary, valid_count, invalid_count, time.time() - started))
        manifest = {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "status": "completed", "signature": signature, "signature_payload": signature_payload, "inputs": {"campaign_signatures": {name: campaign.signature for name, campaign in config.campaigns.items()}, "paths": input_paths, "hashes": input_hashes}, "counts": {"examples": len(joined), "hearings": len(groups), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "bootstrap_requested": config.n_replicates, "bootstrap_valid": valid_count, "bootstrap_invalid": invalid_count}, "metrics": list(config.metrics), "delta_convention": "set_transformer - gated_attention; negative brier favors set_transformer", "artifacts": {}}
        manifest["artifacts"] = _file_hashes(stage)
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
