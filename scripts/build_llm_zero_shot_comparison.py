#!/usr/bin/env python3
"""Build a common-population comparison of stored paper LLM decisions and zero-shot models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


EXPECTED_N = 4235
EXPECTED_POSITIVES = 501
EXPECTED_HEARINGS = 206
PROMPTS = ("prompt_1", "prompt_2", "prompt_3")
COMPACT_PROMPTS = {"prompt_1", "prompt_3"}
LLM_MODEL_ORDER = ("gpt-4o", "gpt-4o-mini", "deepseek", "sabia-3.1")
LLM_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
    "deepseek": "DeepSeek (stored key: deepseek-chat)",
    "sabia-3.1": "Sabiá-3.1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def example_id(hearing_id: str, person_index: int, opinion_index: int) -> str:
    return f"{hearing_id}:{person_index}:{opinion_index}"


def load_llm_source(path: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    pair_keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            hearing_id = str(record["id"]).strip()
            metadata = record.get("metadados_extraidos") or {}
            for person_index, person in enumerate(metadata.get("envolvidos") or []):
                for opinion_index, opinion_entry in enumerate(person.get("opinioes") or []):
                    chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                    verification = opinion_entry.get("verificacao_alucinacao") or {}
                    pair_values = {
                        key: value
                        for key, value in verification.items()
                        if key.startswith("prompt_") and isinstance(value, dict)
                    }
                    pair_keys.update(pair_values)
                    rows.append(
                        {
                            "example_id": example_id(hearing_id, person_index, opinion_index),
                            "hearing_id": hearing_id,
                            "label": int(bool(verification.get("verificacao_manual"))),
                            "modelable": bool(
                                hearing_id
                                and str(opinion_entry.get("opiniao", "")).strip()
                                and len(chunks) == 4
                                and all(chunk.strip() for chunk in chunks)
                            ),
                            "verification": verification,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["example_id"].is_unique:
        raise ValueError("LLM source has no unique example identifiers.")
    modelable = frame.loc[frame["modelable"]].reset_index(drop=True)
    if len(modelable) != EXPECTED_N or int(modelable["label"].sum()) != EXPECTED_POSITIVES:
        raise ValueError(
            f"Unexpected common population: N={len(modelable)}, positives={int(modelable['label'].sum())}"
        )
    if modelable["hearing_id"].nunique() != EXPECTED_HEARINGS:
        raise ValueError("Unexpected number of hearings in common population.")
    return modelable, sorted(pair_keys)


def parse_pair_key(pair_key: str) -> tuple[str, str]:
    match = re.fullmatch(r"prompt_([123])_(.+)", pair_key)
    if not match:
        raise ValueError(f"Unexpected LLM pair key: {pair_key}")
    prompt = f"prompt_{match.group(1)}"
    model_token = match.group(2)
    if model_token.startswith("gpt-4o-mini-"):
        model = "gpt-4o-mini"
    elif model_token.startswith("gpt-4o-"):
        model = "gpt-4o"
    elif model_token == "deepseek-chat":
        model = "deepseek"
    elif model_token.startswith("sabia-3.1-"):
        model = "sabia-3.1"
    else:
        raise ValueError(f"Unrecognized LLM model key: {pair_key}")
    return model, prompt


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    tn = int(np.sum(~labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "MCC": float(matthews_corrcoef(labels.astype(int), predictions.astype(int))),
    }


def build_llm_rows(modelable: pd.DataFrame, pair_keys: list[str], source: Path) -> list[dict[str, Any]]:
    expected = {f"prompt_{prompt}_{suffix}" for prompt in ("1", "2", "3") for suffix in (
        "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "deepseek-chat", "sabia-3.1-2025-05-08"
    )}
    if set(pair_keys) != expected:
        raise ValueError(f"Expected exactly 12 paper model/prompt keys, found {pair_keys}")
    labels = modelable["label"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for pair_key in pair_keys:
        model, prompt = parse_pair_key(pair_key)
        predictions: list[bool] = []
        for verification in modelable["verification"]:
            decision = verification.get(pair_key)
            if not isinstance(decision, dict) or not isinstance(decision.get("alucinacao"), bool):
                raise ValueError(f"Missing or non-boolean LLM prediction: {pair_key}")
            predictions.append(decision["alucinacao"])
        rows.append(
            {
                "method": LLM_LABELS[model],
                "prompt_or_criterion": prompt.replace("prompt_", "Prompt "),
                "model_key": pair_key,
                "evaluation_protocol": "paper stored binary judgment",
                "source_file": str(source.resolve()),
                "threshold_source": "not applicable",
                "evaluation_population": EXPECTED_N,
                "positive_labels": EXPECTED_POSITIVES,
                **binary_metrics(labels, np.asarray(predictions, dtype=bool)),
            }
        )
    return rows


def load_zero_shot_rows(path: Path) -> list[dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen zero-shot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("signature_payload", {}).get("config", {}).get("protocol", {})
    if manifest.get("signature") != "fc7bbae7bd5d0c99" or manifest.get("status") != "completed":
        raise ValueError("The supplied zero-shot summary is not the completed official run.")
    if protocol.get("thresholds") != "frozen per condition and seed from validation only":
        raise ValueError("The official zero-shot protocol does not document frozen validation thresholds.")
    frame = pd.read_csv(path)
    required = {
        "contrast_id", "metric", "regime", "point_a", "point_b", "population_examples", "population_positives",
    }
    if not required <= set(frame.columns):
        raise ValueError(f"Frozen summary lacks columns: {sorted(required - set(frame.columns))}")
    selected = frame.loc[
        frame["contrast_id"].eq("en_set_vs_attention_ph_pt_thresholded")
        & frame["regime"].isin(["best_f1", "fpr10"])
        & frame["metric"].isin(["precision", "recall", "f1", "fpr", "mcc"])
    ].copy()
    if len(selected) != 10:
        raise ValueError(f"Expected 10 frozen EN to PH PT thresholded rows, found {len(selected)}")
    if not (selected["population_examples"].eq(EXPECTED_N).all() and selected["population_positives"].eq(EXPECTED_POSITIVES).all()):
        raise ValueError("Frozen zero-shot summary has an incompatible population.")
    rows: list[dict[str, Any]] = []
    for regime in ("best_f1", "fpr10"):
        for column, method in (("point_a", "Gated Attention MIL"), ("point_b", "Set Transformer")):
            selected_regime = selected.loc[selected["regime"].eq(regime)].set_index("metric")
            if set(["precision", "recall", "f1", "fpr", "mcc"]) != set(selected_regime.index):
                raise ValueError(f"Incomplete frozen metrics for {method}/{regime}")
            rows.append(
                {
                    "method": method,
                    "prompt_or_criterion": "Max. F1 threshold" if regime == "best_f1" else "Recall under FPR <= 10% threshold",
                    "model_key": method.lower().replace(" ", "_"),
                    "evaluation_protocol": "official frozen zero-shot point estimate",
                    "source_file": str(path.resolve()),
                    "threshold_source": "RAGTruth validation only (frozen; not recomputed)",
                    "evaluation_population": EXPECTED_N,
                    "positive_labels": EXPECTED_POSITIVES,
                    "Precision": float(selected_regime.loc["precision", column]),
                    "Recall": float(selected_regime.loc["recall", column]),
                    "F1": float(selected_regime.loc["f1", column]),
                    "FPR": float(selected_regime.loc["fpr", column]),
                    "MCC": float(selected_regime.loc["mcc", column]),
                }
            )
    return rows


def write_markdown(frame: pd.DataFrame, path: Path, title: str) -> None:
    columns = ["method", "prompt_or_criterion", "Precision", "Recall", "F1", "FPR", "MCC", "evaluation_population"]
    display = frame[columns].rename(columns={"method": "Method", "prompt_or_criterion": "Prompt / criterion", "evaluation_population": "N"})
    for column in ["Precision", "Recall", "F1", "FPR", "MCC"]:
        display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    headers = [str(column) for column in display.columns]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compact_latex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stored PublicHearingBR paper LLM judgments and frozen zero-shot results on the common modelable population ($N=4{,}235$; 501 positives).}",
        r"\label{tab:llm-zero-shot-common-population}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Prompt / criterion & Precision & Recall & F1 & FPR & MCC \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        method = str(row.method).replace(" (stored key: deepseek-chat)", r" (key: \texttt{deepseek-chat})")
        criterion = str(row.prompt_or_criterion).replace("<=", r"$\leq$")
        lines.append(f"{method} & {criterion} & {row.Precision:.4f} & {row.Recall:.4f} & {row.F1:.4f} & {row.FPR:.4f} & {row.MCC:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    llm_source = args.llm_source.resolve()
    summary_source = args.zero_shot_summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modelable, pair_keys = load_llm_source(llm_source)
    rows = build_llm_rows(modelable, pair_keys, llm_source) + load_zero_shot_rows(summary_source)
    full = pd.DataFrame(rows)
    full_csv = output_dir / "llm_vs_zero_shot_full.csv"
    full_md = output_dir / "llm_vs_zero_shot_full.md"
    compact = full.loc[
        (~full["evaluation_protocol"].eq("paper stored binary judgment"))
        | full["prompt_or_criterion"].isin(["Prompt 1", "Prompt 3"])
    ].copy()
    compact_csv = output_dir / "llm_vs_zero_shot_compact.csv"
    compact_tex = output_dir / "llm_vs_zero_shot_compact.tex"
    full.to_csv(full_csv, index=False)
    compact.to_csv(compact_csv, index=False)
    write_markdown(full, full_md, "PublicHearingBR paper LLM judgments vs. frozen zero-shot models")
    write_compact_latex(compact, compact_tex)
    manifest = {
        "status": "completed",
        "comparison": "stored PublicHearingBR paper LLM binary judgments vs frozen EN to PH PT zero-shot point estimates",
        "llm_calls": 0,
        "thresholds_recomputed": False,
        "bootstrap_created": False,
        "population": {"examples": EXPECTED_N, "positives": EXPECTED_POSITIVES, "hearings": EXPECTED_HEARINGS},
        "llm_source": {"path": str(llm_source), "sha256": sha256_file(llm_source), "pair_keys": pair_keys},
        "zero_shot_summary_source": {"path": str(summary_source), "sha256": sha256_file(summary_source)},
        "compact_selection": "Prompts 1 and 3 for every LLM; both criteria for both zero-shot architectures",
        "deepseek_naming": "Repository key is deepseek-chat; explicit DeepSeek-V3 mapping not confirmed.",
        "files": {path.name: sha256_file(path) for path in (full_csv, full_md, compact_csv, compact_tex)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-source",
        type=Path,
        default=Path("data/processed/publichearing_nli_pt_to_en_nllb/PublicHearingBR_NLI.jsonl"),
    )
    parser.add_argument(
        "--zero-shot-summary",
        type=Path,
        default=Path("runs/publichearing_final_paired_bootstrap/fc7bbae7bd5d0c99/final_bootstrap_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper/llm_zero_shot_comparison"))
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve()), "rows": 16}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
