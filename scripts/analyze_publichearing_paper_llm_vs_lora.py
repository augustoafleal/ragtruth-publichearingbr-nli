#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


REGIMES = ("best_f1", "fpr10")
METRICS = ("Precision", "Recall", "F1", "FPR", "MCC", "BalancedAccuracy")
EXPECTED_PAPER_PAIRS = 12
EXPECTED_CURRENT_EXAMPLES = 4235
EXPECTED_CURRENT_POSITIVES = 501


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _example_id(hearing_id: str, person_index: int, opinion_index: int) -> str:
    return f"{hearing_id}:{person_index}:{opinion_index}"


def load_paper_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

    all_rows: list[dict[str, Any]] = []
    pair_keys: set[str] = set()
    for record in _jsonl(path):
        hearing_id = str(record["id"])
        metadata = record["metadados_extraidos"]
        for person_index, person in enumerate(metadata.get("envolvidos", [])):
            for opinion_index, opinion_entry in enumerate(person.get("opinioes", [])):
                chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                claim = str(opinion_entry.get("opiniao", ""))
                verification = opinion_entry.get("verificacao_alucinacao", {})
                pairs = {
                    key: value
                    for key, value in verification.items()
                    if key.startswith("prompt_") and isinstance(value, dict)
                }
                pair_keys.update(pairs)
                reasons: list[str] = []
                if len(chunks) != 4:
                    reasons.append("chunk_count_not_4")
                if not hearing_id:
                    reasons.append("empty_hearing_id")
                if not claim.strip():
                    reasons.append("empty_claim")
                if any(not chunk.strip() for chunk in chunks):
                    reasons.append("empty_chunk")
                all_rows.append(
                    {
                        "example_id": _example_id(hearing_id, person_index, opinion_index),
                        "hearing_id": hearing_id,
                        "person_index": person_index,
                        "opinion_index": opinion_index,
                        "label": int(bool(verification.get("verificacao_manual"))),
                        "chunk_count": len(chunks),
                        "claim_nonempty": bool(claim.strip()),
                        "chunks_all_nonempty": bool(chunks) and all(chunk.strip() for chunk in chunks),
                        "modelable_current_zero_shot": not reasons,
                        "exclusion_reasons": ";".join(reasons),
                        "llm_pairs_available": len(pairs),
                        "verification": verification,
                    }
                )

    all_frame = pd.DataFrame(all_rows)
    if not all_frame["example_id"].is_unique:
        raise ValueError("IDs duplicados no PublicHearingBR_NLI.jsonl")
    modelable = all_frame.loc[all_frame["modelable_current_zero_shot"]].copy().reset_index(drop=True)
    return all_frame, modelable, sorted(pair_keys)


def alignment_frame(all_frame: pd.DataFrame, current_ids: set[str]) -> pd.DataFrame:
    frame = all_frame.drop(columns=["verification"]).copy()
    frame["in_current_lora_zero_shot_predictions"] = frame["example_id"].astype(str).isin(current_ids)
    frame["alignment_status"] = np.where(
        frame["modelable_current_zero_shot"] & frame["in_current_lora_zero_shot_predictions"],
        "included",
        "excluded",
    )
    return frame.sort_values("example_id", key=lambda values: values.map(lambda value: tuple(int(part) for part in str(value).split(":")))).reset_index(drop=True)


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "N": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "MCC": float(matthews_corrcoef(labels.astype(int), predictions.astype(int))),
        "BalancedAccuracy": (recall + specificity) / 2.0,
    }


def _paper_pair_parts(pair_key: str) -> tuple[str, str]:
    match = re.fullmatch(r"(prompt_[^_]+)_(.+)", pair_key)
    if not match:
        raise ValueError(f"Nome de par modelo × prompt inesperado: {pair_key}")
    return match.group(2), match.group(1)


def paper_metric_rows(modelable: pd.DataFrame, pair_keys: list[str]) -> list[dict[str, Any]]:
    labels = modelable["label"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for pair_key in pair_keys:
        predictions: list[bool] = []
        for verification in modelable["verification"]:
            decision = verification.get(pair_key)
            if not isinstance(decision, dict) or not isinstance(decision.get("alucinacao"), bool):
                raise ValueError(f"Decisão ausente/inválida para {pair_key}")
            predictions.append(bool(decision["alucinacao"]))
        model_name, prompt_name = _paper_pair_parts(pair_key)
        rows.append(
            {
                "evaluation_unit": "paper_llm",
                "system": pair_key,
                "paper_pair_original": pair_key,
                "paper_model": model_name,
                "prompt": prompt_name,
                "lora_seed": pd.NA,
                "operating_point": "fixed_binary",
                "threshold": np.nan,
                "threshold_source": "authors/paper stored decision",
                "model_name": model_name,
                "prediction_source": "stored in PublicHearingBR_NLI.jsonl",
                **binary_metrics(labels, np.asarray(predictions, dtype=bool)),
            }
        )
    return rows


def _seed_from_path(path: Path) -> int:
    matches = re.findall(r"seed_(\d+)", str(path))
    if not matches:
        raise ValueError(f"Não foi possível inferir a seed LoRA de {path}")
    return int(matches[-1])


def lora_metric_rows(modelable: pd.DataFrame, prediction_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = modelable["label"].to_numpy(bool)
    expected_ids = modelable["example_id"].astype(str).tolist()
    expected_hearings = modelable["hearing_id"].astype(str).tolist()
    expected_labels = labels.astype(int).tolist()
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    seen_seeds: set[int] = set()
    for path in prediction_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Predição LoRA não encontrada: {path}")
        frame = pd.read_parquet(path)
        seed = _seed_from_path(path)
        if seed in seen_seeds:
            raise ValueError(f"Mais de uma previsão LoRA para a seed {seed}")
        seen_seeds.add(seed)
        required = {
            "example_id",
            "hearing_id",
            "label",
            "prediction_ragtruth_best_f1",
            "prediction_ragtruth_fpr10",
            "ragtruth_best_f1_threshold",
            "ragtruth_fpr10_threshold",
            "model_run_signature",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Predição LoRA sem colunas {sorted(missing)}: {path}")
        if frame["example_id"].astype(str).tolist() != expected_ids:
            raise ValueError(f"IDs LoRA fora de ordem ou incompatíveis com o subconjunto atual: {path}")
        if frame["hearing_id"].astype(str).tolist() != expected_hearings or frame["label"].astype(int).tolist() != expected_labels:
            raise ValueError(f"Labels/hearing_id LoRA incompatíveis com o subconjunto atual: {path}")
        run_manifest_path = path.parents[2] / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.is_file() else {}
        model_name = str(run_manifest.get("config", {}).get("model_id", "LoRA checkpoint model"))
        seed_provenance = {
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "run_manifest": str(run_manifest_path.resolve()),
            "run_signature": str(frame["model_run_signature"].iloc[0]),
            "model_name": model_name,
        }
        provenance[str(seed)] = seed_provenance
        for regime, prediction_column, threshold_column in (
            ("best_f1", "prediction_ragtruth_best_f1", "ragtruth_best_f1_threshold"),
            ("fpr10", "prediction_ragtruth_fpr10", "ragtruth_fpr10_threshold"),
        ):
            thresholds = frame[threshold_column].astype(float).unique()
            if len(thresholds) != 1:
                raise ValueError(f"Threshold {threshold_column} não é único em {path}")
            rows.append(
                {
                    "evaluation_unit": "lora",
                    "system": f"LoRA seed_{seed}",
                    "paper_pair_original": pd.NA,
                    "paper_model": pd.NA,
                    "prompt": pd.NA,
                    "lora_seed": seed,
                    "operating_point": regime,
                    "threshold": float(thresholds[0]),
                    "threshold_source": "RAGTruth validation (frozen)",
                    "model_name": model_name,
                    "prediction_source": str(path.resolve()),
                    **binary_metrics(labels, frame[prediction_column].astype(bool).to_numpy()),
                }
            )
    if seen_seeds != {0, 1, 2}:
        raise ValueError(f"Esperavam-se as três seeds confirmatórias {{0, 1, 2}}, obtidas {sorted(seen_seeds)}")
    return rows, provenance


def write_plot(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    llm = metrics.loc[metrics["evaluation_unit"].eq("paper_llm")]
    lora = metrics.loc[metrics["evaluation_unit"].eq("lora")]
    ax.scatter(llm["Recall"], llm["FPR"], s=58, color="#4472c4", alpha=0.85, label="Pares modelo × prompt (paper)")
    for regime, marker, color in (("best_f1", "*", "#c00000"), ("fpr10", "D", "#ed7d31")):
        selected = lora.loc[lora["operating_point"].eq(regime)]
        ax.scatter(selected["Recall"], selected["FPR"], s=145 if marker == "*" else 76, marker=marker, color=color, edgecolor="black", linewidth=0.45, label=f"LoRA · {regime}", zorder=4)
        for _, row in selected.iterrows():
            ax.annotate(f"s{int(row['lora_seed'])}", (row["Recall"], row["FPR"]), xytext=(5, -10), textcoords="offset points", fontsize=8, color=color)
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=0.9, label="FPR = 0,10")
    ax.set_xlabel("Recall")
    ax.set_ylabel("FPR")
    ax.set_title("PublicHearingBR: FPR × Recall no subconjunto modelável (N=4.235)\n12 pares paper sem rótulos individuais; nomes completos na tabela")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    alignment: pd.DataFrame,
    paper_path: Path,
    prediction_paths: list[Path],
    provenance: dict[str, Any],
) -> None:
    included = alignment.loc[alignment["alignment_status"].eq("included")]
    excluded = alignment.loc[alignment["alignment_status"].eq("excluded")]
    lines = [
        "# Comparação: julgamentos LLM armazenados × operating points LoRA",
        "",
        "Esta análise é pós-hoc e determinística: não carregou modelos, não treinou e não chamou nenhum LLM.",
        "Todas as métricas abaixo foram recalculadas no mesmo subconjunto modelável usado pela avaliação zero-shot atual.",
        "",
        "## Alinhamento por ID",
        "",
        f"- Arquivo do paper: `{paper_path.resolve()}`.",
        f"- Opiniões no arquivo do paper: **{len(alignment):,}**.",
        f"- IDs incluídos no subconjunto atual: **{len(included):,}** ({int(included['label'].sum()):,} positivos, {len(included) - int(included['label'].sum()):,} negativos).",
        f"- IDs fora do subconjunto atual: **{len(excluded):,}**.",
        f"- Audiências incluídas: **{included['hearing_id'].nunique():,}**.",
        "",
        "Os resultados publicados/descritos para 4.238 opiniões não foram usados como métricas de comparação. A tabela seguinte contém apenas métricas recalculadas para N=4.235.",
        "",
        "| ID fora | Motivo | chunks | label manual |",
        "|---|---|---:|---:|",
    ]
    for _, row in excluded.iterrows():
        lines.append(f"| `{row['example_id']}` | `{row['exclusion_reasons']}` | {int(row['chunk_count'])} | {int(row['label'])} |")
    lines += [
        "",
        "A auditoria completa, incluindo todos os IDs que entraram e ficaram fora, está em `alignment_by_id.csv`.",
        "",
        "## Métricas recalculadas no subconjunto atual",
        "",
        "`label=1` significa alucinação segundo `verificacao_manual`. Para os julgamentos do paper, `alucinacao=True` foi usado como predição positiva.",
        "",
        "| Unidade | Sistema | Operating point | Precision | Recall | F1 | FPR | MCC | Balanced Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append("| " + " | ".join(_fmt(row[column]) for column in ("evaluation_unit", "system", "operating_point", *METRICS)) + " |")
    ranked = metrics.sort_values(["F1", "MCC", "FPR"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
    lines += [
        "",
        "## Mesma tabela ordenada por melhor desempenho",
        "",
        "Ordenação principal por **F1 decrescente**, com desempates por **MCC decrescente** e depois **FPR crescente**.",
        "",
        "| Rank | Unidade | Sistema | Operating point | Precision | Recall | F1 | FPR | MCC | Balanced Accuracy |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append("| " + " | ".join([str(rank)] + [_fmt(row[column]) for column in ("evaluation_unit", "system", "operating_point", *METRICS)]) + " |")
    lines += [
        "",
        "### LoRA: média ± desvio-padrão entre as três seeds",
        "",
    ]
    for regime in REGIMES:
        selected = metrics.loc[(metrics["evaluation_unit"].eq("lora")) & (metrics["operating_point"].eq(regime))]
        lines.append(f"- **{regime}**: " + ", ".join(f"{metric}={selected[metric].mean():.6f} ± {selected[metric].std(ddof=1):.6f}" for metric in METRICS) + ".")
    lines += [
        "",
        "## Fontes LoRA",
        "",
        "Os pontos LoRA foram lidos dos parquets confirmatórios já existentes. Seus thresholds continuam sendo os thresholds congelados selecionados no RAGTruth validation.",
        "",
    ]
    for seed in sorted(provenance, key=int):
        lines.append(f"- seed_{seed}: `{provenance[seed]['path']}`, SHA-256 `{provenance[seed]['sha256']}`.")
    lines += [
        "",
        "## Limitações do alinhamento",
        "",
        "- Três das 4.238 opiniões do arquivo do paper não são modeláveis pelo contrato atual: `141:1:0` tem número de chunks diferente de 4, enquanto `99:5:0` e `99:5:1` contêm chunk vazio.",
        "- Os 12 pares modelo × prompt têm decisão booleana armazenada para todos os 4.235 IDs incluídos. Não houve decisão faltante no subconjunto comparado.",
        "- Não foi calculada uma decisão LoRA nova nem um ensemble: os seis pontos LoRA são os resultados existentes das seeds 0, 1 e 2, separados por operating point.",
        "",
        "Arquivos gerados: `metrics_consolidated.csv`, `metrics_consolidated.parquet`, `metrics_consolidated_ranked.csv`, `metrics_consolidated_ranked.parquet`, `alignment_by_id.csv`, `fpr_x_recall.png` e `manifest.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    paper_path = args.paper_path.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    prediction_paths = [path.expanduser().resolve() for path in args.lora_prediction]
    if not paper_path.is_file():
        raise FileNotFoundError(f"PublicHearingBR_NLI.jsonl não encontrado: {paper_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frame, modelable, pair_keys = load_paper_rows(paper_path)
    if len(pair_keys) != EXPECTED_PAPER_PAIRS:
        raise ValueError(f"Esperavam-se {EXPECTED_PAPER_PAIRS} pares modelo × prompt, obtidos {len(pair_keys)}")
    if len(modelable) != EXPECTED_CURRENT_EXAMPLES or int(modelable["label"].sum()) != EXPECTED_CURRENT_POSITIVES:
        raise ValueError(f"Subconjunto atual inesperado: N={len(modelable)}, positivos={int(modelable['label'].sum())}")
    current_ids = set(modelable["example_id"].astype(str))
    alignment = alignment_frame(all_frame, current_ids)
    if not bool((alignment.loc[alignment["alignment_status"].eq("included"), "example_id"].astype(str).isin(current_ids)).all()):
        raise ValueError("Falha na auditoria de IDs incluídos")

    paper_rows = paper_metric_rows(modelable, pair_keys)
    lora_rows, provenance = lora_metric_rows(modelable, prediction_paths)
    metrics = pd.DataFrame(paper_rows + lora_rows)
    metrics.insert(0, "dataset", "PublicHearingBR")
    metrics.insert(1, "subset", "current_zero_shot_modelable")
    metrics.insert(2, "subset_N", len(modelable))
    metrics.insert(3, "subset_positives", int(modelable["label"].sum()))
    metrics.insert(4, "subset_negatives", int(len(modelable) - modelable["label"].sum()))
    metrics_path = output_dir / "metrics_consolidated.csv"
    metrics_parquet_path = output_dir / "metrics_consolidated.parquet"
    ranked_metrics = metrics.sort_values(["F1", "MCC", "FPR"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
    ranked_metrics.insert(0, "rank_by_F1", np.arange(1, len(ranked_metrics) + 1))
    ranked_metrics_path = output_dir / "metrics_consolidated_ranked.csv"
    ranked_metrics_parquet_path = output_dir / "metrics_consolidated_ranked.parquet"
    alignment_path = output_dir / "alignment_by_id.csv"
    plot_path = output_dir / "fpr_x_recall.png"
    report_path = output_dir / "summary.md"
    metrics.to_csv(metrics_path, index=False)
    metrics.to_parquet(metrics_parquet_path, index=False)
    ranked_metrics.to_csv(ranked_metrics_path, index=False)
    ranked_metrics.to_parquet(ranked_metrics_parquet_path, index=False)
    alignment.to_csv(alignment_path, index=False)
    write_plot(metrics, plot_path)
    write_report(report_path, metrics, alignment, paper_path, prediction_paths, provenance)

    manifest = {
        "status": "completed",
        "analysis": "stored_paper_llm_judgments_vs_frozen_lora_operating_points",
        "llm_calls": 0,
        "training_or_model_inference": False,
        "paper_dataset": {"path": str(paper_path), "sha256": sha256_file(paper_path), "opinions": len(all_frame), "manual_positives": int(all_frame["label"].sum())},
        "current_modelable_subset": {"examples": len(modelable), "positives": int(modelable["label"].sum()), "negatives": int((modelable["label"] == 0).sum()), "hearings": int(modelable["hearing_id"].nunique()), "included_ids": int((alignment["alignment_status"] == "included").sum()), "excluded_ids": int((alignment["alignment_status"] == "excluded").sum())},
        "paper_pairs": pair_keys,
        "lora_predictions": provenance,
        "files": {path.name: sha256_file(path) for path in (metrics_path, metrics_parquet_path, ranked_metrics_path, ranked_metrics_parquet_path, alignment_path, plot_path, report_path)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-path", type=Path, default=Path("/home/guto/Development/python/ideias-em-rede/data/raw/PublicHearingBR_NLI.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/publichearing_paper_llm_vs_lora_operating_points"))
    parser.add_argument("--lora-prediction", type=Path, action="append", dest="lora_prediction", default=None, help="Existing LoRA predictions.parquet; repeat three times for seeds 0, 1, 2.")
    args = parser.parse_args()
    if args.lora_prediction is None:
        base = Path("runs/ragtruth_confirmatory/4e12933c51136624")
        args.lora_prediction = [next(base.glob(f"seed_{seed}/publichearing_zero_shot/*/predictions.parquet")) for seed in (0, 1, 2)]
    manifest = run(args)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve()), "examples": manifest["current_modelable_subset"]["examples"], "excluded_ids": manifest["current_modelable_subset"]["excluded_ids"], "paper_pairs": len(manifest["paper_pairs"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
