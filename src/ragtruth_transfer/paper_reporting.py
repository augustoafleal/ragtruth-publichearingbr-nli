from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFFICIAL_RUN_ID = "fc7bbae7bd5d0c99"
SUPERSEDED_RUN_ID = "01a3d72bd8e55038"
EXPECTED = {
    "examples": 4235,
    "positives": 501,
    "hearings": 206,
    "replicates": 10000,
    "seed": 20260815,
    "ci_method": "percentile",
    "confidence_level": 0.95,
}
HIGHER_BETTER = {"auprc", "auroc", "f1", "mcc", "precision", "recall", "balanced_accuracy"}
LOWER_BETTER = {"brier", "fpr"}

CONDITIONS = pd.DataFrame([
    {"condition_id": "en_ph_pt", "training": "EN", "target": "PH PT", "attention_id": "en_attention_ph_pt", "set_id": "en_set_ph_pt", "label": "EN → PH PT", "short": "EN/PT"},
    {"condition_id": "pt_nllb_ph_pt", "training": "PT-NLLB", "target": "PH PT", "attention_id": "pt_nllb_attention_ph_pt", "set_id": "pt_nllb_set_ph_pt", "label": "PT-NLLB → PH PT", "short": "NLLB/PT"},
    {"condition_id": "pt_madlad_ph_pt", "training": "PT-MADLAD", "target": "PH PT", "attention_id": "pt_madlad_attention_ph_pt", "set_id": "pt_madlad_set_ph_pt", "label": "PT-MADLAD → PH PT", "short": "MADLAD/PT"},
    {"condition_id": "en_ph_en_nllb", "training": "EN", "target": "PH EN-NLLB", "attention_id": "en_attention_ph_en_nllb", "set_id": "en_set_ph_en_nllb", "label": "EN → PH EN-NLLB", "short": "EN/EN-NLLB"},
    {"condition_id": "en_ph_en_madlad", "training": "EN", "target": "PH EN-MADLAD", "attention_id": "en_attention_ph_en_madlad", "set_id": "en_set_ph_en_madlad", "label": "EN → PH EN-MADLAD", "short": "EN/EN-MADLAD"},
])
AGGREGATOR_CONTRASTS = {
    "en_ph_pt": "en_set_vs_attention_ph_pt",
    "pt_nllb_ph_pt": "pt_nllb_set_vs_attention_ph_pt",
    "pt_madlad_ph_pt": "pt_madlad_set_vs_attention_ph_pt",
    "en_ph_en_nllb": "en_set_vs_attention_ph_en_nllb",
    "en_ph_en_madlad": "en_set_vs_attention_ph_en_madlad",
}
TRANSLATION_SPECS = [
    ("Training-side translation", "PT-NLLB training - EN training", "pt_nllb_set_vs_en_ph_pt", "training_translation"),
    ("Training-side translation", "PT-MADLAD training - EN training", "pt_madlad_set_vs_en_ph_pt", "training_translation"),
    ("Target-side translation", "PH EN-NLLB target - PH PT target", "en_set_ph_en_nllb_vs_pt", "target_translation"),
    ("Target-side translation", "PH EN-MADLAD target - PH PT target", "en_set_ph_en_madlad_vs_pt", "target_translation"),
]
INTERACTION_SPECS = [("NLLB", "nllb_set_target_minus_training"), ("MADLAD", "madlad_set_target_minus_training")]


def build_source_registry(root: Path, official_run: Path | str | None = None, output_root: Path | None = None) -> dict[str, Path]:
    root = Path(root).resolve()
    expected_official = (root / "runs/publichearing_final_paired_bootstrap" / OFFICIAL_RUN_ID).resolve()
    official = expected_official if official_run is None else Path(official_run)
    if not official.is_absolute():
        official = root / official
    official = official.resolve()
    if official != expected_official:
        raise ValueError(
            f"Only the explicit official statistical run is allowed: {expected_official}; received {official}. "
            f"The superseded run {SUPERSEDED_RUN_ID} is forbidden."
        )
    out = (output_root or root / "results/paper").resolve()
    return {
        "OFFICIAL_BOOTSTRAP_RUN": official,
        "FINAL_SUMMARY": official / "final_bootstrap_summary.csv",
        "FINAL_REPORT": official / "final_bootstrap_report.md",
        "MANIFEST": official / "manifest.json",
        "RESOLVED_CONFIG": official / "resolved_config.json",
        "POPULATION_VALIDATION": official / "population_validation.json",
        "POOLING_RESULTS": (root / "runs/publichearing_pooling_paired_bootstrap/b92d4fd6a6a7e771").resolve(),
        "BASELINE_RESULTS": (root / "runs/publichearing_off_the_shelf_max_entailment/54d9c623f8685c39").resolve(),
        "OFF_THE_SHELF_THRESHOLDED": (root / "runs/ragtruth_off_the_shelf_threshold_transfer/3ceffc4a74b484fe/publichearing_metrics.json").resolve(),
        "SUPERVISED_RESULTS": (root / "results/publichearing_lora_attention_mil/741ed3152c2175e7/outputs/overall_oof_metrics.csv").resolve(),
        "FILTERED_AGGREGATE": (root / "runs/ragtruth_pt_nllb_filtered_confirmatory/63745412afdb52ac/aggregate/aggregate_metrics.csv").resolve(),
        "BERTIMBAU_AGGREGATE": (root / "runs/ragtruth_pt_nllb_bertimbau_confirmatory/fce6272e2e72726d/aggregate/aggregate_metrics.csv").resolve(),
        "PT_NLLB_THRESHOLDED": (root / "runs/publichearing_pt_nllb_attention_vs_set_thresholded_bootstrap/e0c75270065fc471/bootstrap_summary.json").resolve(),
        "OUTPUT_ROOT": out,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_sources(registry: dict[str, Path]) -> dict[str, Any]:
    required = [
        "OFFICIAL_BOOTSTRAP_RUN", "FINAL_SUMMARY", "FINAL_REPORT", "MANIFEST",
        "RESOLVED_CONFIG", "POPULATION_VALIDATION", "POOLING_RESULTS",
        "BASELINE_RESULTS", "OFF_THE_SHELF_THRESHOLDED", "SUPERVISED_RESULTS", "FILTERED_AGGREGATE",
        "BERTIMBAU_AGGREGATE", "PT_NLLB_THRESHOLDED",
    ]
    missing = []
    for key in required:
        path = registry[key]
        if key in {"POOLING_RESULTS", "BASELINE_RESULTS"}:
            if not path.is_dir():
                missing.append(f"{key}: {path}")
        elif not path.exists():
            missing.append(f"{key}: {path}")
    if missing:
        raise FileNotFoundError("Required frozen reporting input(s) missing:\n" + "\n".join(missing))
    official = registry["OFFICIAL_BOOTSTRAP_RUN"]
    if official.name != OFFICIAL_RUN_ID:
        raise AssertionError(f"Incorrect official run resolved: {official}")
    if SUPERSEDED_RUN_ID in str(official):
        raise AssertionError("The superseded run was selected.")
    manifest = read_json(registry["MANIFEST"])
    config = read_json(registry["RESOLVED_CONFIG"])
    population = read_json(registry["POPULATION_VALIDATION"])
    summary = pd.read_csv(registry["FINAL_SUMMARY"])
    if manifest.get("signature") != OFFICIAL_RUN_ID or manifest.get("status") != "completed":
        raise AssertionError("Official manifest is not completed or has the wrong signature.")
    checks = {
        "examples": population.get("examples"),
        "positives": population.get("positives"),
        "hearings": population.get("hearings"),
        "replicates": set(summary["replicates"].unique()),
        "valid_replicates": set(summary["valid_replicates"].unique()),
        "invalid_replicates": set(summary["invalid_replicates"].unique()),
        "seed": set(summary["seed"].unique()),
        "population_examples": set(summary["population_examples"].unique()),
        "population_positives": set(summary["population_positives"].unique()),
        "population_hearings": set(summary["population_hearings"].unique()),
    }
    assert checks["examples"] == EXPECTED["examples"]
    assert checks["positives"] == EXPECTED["positives"]
    assert checks["hearings"] == EXPECTED["hearings"]
    assert checks["replicates"] == {EXPECTED["replicates"]}
    assert checks["valid_replicates"] == {EXPECTED["replicates"]}
    assert checks["invalid_replicates"] == {0}
    assert checks["seed"] == {EXPECTED["seed"]}
    assert checks["population_examples"] == {EXPECTED["examples"]}
    assert checks["population_positives"] == {EXPECTED["positives"]}
    assert checks["population_hearings"] == {EXPECTED["hearings"]}
    assert population["exact_example_alignment"]
    assert population["exact_label_alignment"]
    assert population["exact_hearing_alignment"]
    assert population["probabilities_valid"]
    bootstrap = config["bootstrap"]
    assert bootstrap["n_replicates"] == EXPECTED["replicates"]
    assert bootstrap["seed"] == EXPECTED["seed"]
    assert bootstrap["ci_method"] == EXPECTED["ci_method"]
    assert bootstrap["confidence_level"] == EXPECTED["confidence_level"]
    mismatches = orientation_mismatches(summary)
    assert not mismatches, f"Inconsistent significant_direction values: {mismatches}"
    return {"registry": registry, "manifest": manifest, "config": config, "population": population, "summary": summary, "orientation_mismatches": mismatches}


def orientation_mismatches(summary: pd.DataFrame) -> list[dict[str, Any]]:
    mismatches = []
    for idx, row in summary.iterrows():
        low, high = float(row["ci_low"]), float(row["ci_high"])
        family, metric = row["contrast_family"], row["metric"]
        if family in {"translation_interaction", "set_effect_interaction"}:
            expected = "NEGATIVE" if high < 0 else "POSITIVE" if low > 0 else "INCONCLUSIVE"
        elif metric in HIGHER_BETTER:
            expected = "FAVOR_B" if low > 0 else "FAVOR_A" if high < 0 else "INCONCLUSIVE"
        elif metric in LOWER_BETTER:
            expected = "FAVOR_A" if low > 0 else "FAVOR_B" if high < 0 else "INCONCLUSIVE"
        else:
            expected = "UNKNOWN"
        if row["significant_direction"] != expected:
            mismatches.append({"row": int(idx), "expected": expected, "actual": row["significant_direction"]})
    return mismatches


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "appendix": output_root / "appendix",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for stale_pdf in paths["figures"].glob("*.pdf"):
        stale_pdf.unlink()
    return paths


def collect_points(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    points: dict[str, dict[str, set[float]]] = {}
    source = summary.loc[summary["regime"].eq("threshold_free")].dropna(subset=["point_a", "point_b"])
    for _, row in source.iterrows():
        for cid, value in ((row["condition_a"], row["point_a"]), (row["condition_b"], row["point_b"])):
            points.setdefault(cid, {}).setdefault(row["metric"], set()).add(float(value))
    for cid, metrics in points.items():
        for metric, values in metrics.items():
            if len(values) != 1:
                raise AssertionError(f"Conflicting point estimates for {cid}/{metric}: {values}")
    return {cid: {metric: next(iter(values)) for metric, values in metrics.items()} for cid, metrics in points.items()}


def point(points: dict[str, dict[str, float]], condition_id: str, metric: str) -> float:
    if condition_id not in points or metric not in points[condition_id]:
        raise KeyError(f"Missing frozen point estimate: {condition_id}/{metric}")
    return points[condition_id][metric]


def contrast_row(summary: pd.DataFrame, contrast_id: str, metric: str, regime: str = "threshold_free", family: str | None = None) -> pd.Series:
    rows = summary.loc[
        summary["contrast_id"].eq(contrast_id)
        & summary["metric"].eq(metric)
        & summary["regime"].eq(regime)
    ]
    if family is not None:
        rows = rows.loc[rows["contrast_family"].eq(family)]
    if len(rows) != 1:
        raise AssertionError(f"Expected one row for {contrast_id}/{metric}/{regime}; found {len(rows)}")
    return rows.iloc[0]


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def fmt_signed(value: float) -> str:
    return f"{float(value):+.4f}"


def ci_text(row: pd.Series) -> str:
    return f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"


def presentation_verdict(row: pd.Series) -> str:
    return {"FAVOR_B": "Favors Set", "FAVOR_A": "Favors Attention", "INCONCLUSIVE": "Inconclusive"}.get(
        row["significant_direction"], row["significant_direction"]
    )


def save_table(frame: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, column_format: str) -> None:
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, escape=False, caption=caption, column_format=column_format))
    if csv_path.stat().st_size == 0 or tex_path.stat().st_size == 0:
        raise AssertionError(f"Empty table output: {csv_path} / {tex_path}")


def aggregate_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    rows = frame.loc[frame["dataset"].eq("publichearing_zero_shot") & frame["threshold_regime"].eq("threshold_free")]
    if len(rows) != 1:
        raise AssertionError(f"Expected one aggregate row in {path}; found {len(rows)}")
    row = rows.iloc[0]
    return {metric: float(row[metric]) for metric in ["AUPRC", "AUROC", "Brier"]}


def build_tables(ctx: dict[str, Any], paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    summary, points, registry = ctx["summary"], ctx["points"], ctx["registry"]
    table1 = pd.DataFrame([
        {"Training": row.training, "Target": row.target,
         "Attention AUPRC": point(points, row.attention_id, "auprc"), "Set AUPRC": point(points, row.set_id, "auprc"),
         "Attention AUROC": point(points, row.attention_id, "auroc"), "Set AUROC": point(points, row.set_id, "auroc")}
        for row in CONDITIONS.itertuples()
    ])
    assert len(table1) == 5
    assert (table1["Set AUPRC"] > table1["Attention AUPRC"]).all()
    table1_csv = table1.copy()
    for col in table1_csv.columns[2:]:
        table1_csv[col] = table1_csv[col].map(fmt)
    table1_tex = table1_csv.copy()
    for idx in table1.index:
        for acol, bcol in [("Attention AUPRC", "Set AUPRC"), ("Attention AUROC", "Set AUROC")]:
            a, b = table1.loc[idx, acol], table1.loc[idx, bcol]
            table1_tex.loc[idx, acol] = rf"\textbf{{{fmt(a)}}}" if a > b else fmt(a)
            table1_tex.loc[idx, bcol] = rf"\textbf{{{fmt(b)}}}" if b >= a else fmt(b)
    table1_csv.to_csv(paths["tables"] / "table_main_transfer.csv", index=False)
    (paths["tables"] / "table_main_transfer.tex").write_text(
        table1_tex.to_latex(index=False, escape=False, caption="Cross-lingual transfer and translation strategies on PublicHearingBR", column_format="llrrrr")
    )

    table2_rows = []
    for row in CONDITIONS.itertuples():
        r = contrast_row(summary, AGGREGATOR_CONTRASTS[row.condition_id], "auprc", family="aggregator")
        table2_rows.append({"Condition": row.label, "Attention AUPRC": float(r["point_a"]), "Set AUPRC": float(r["point_b"]),
                            "Delta Set-Attention": float(r["observed_delta"]), "95% CI": ci_text(r),
                            "Verdict": presentation_verdict(r), "_ci_low": float(r["ci_low"]), "_ci_high": float(r["ci_high"])})
    table2 = pd.DataFrame(table2_rows)
    assert len(table2) == 5
    assert (table2["Delta Set-Attention"] > 0).all()
    assert int((table2["_ci_low"] > 0).sum()) == 4
    assert int(((table2["_ci_low"] <= 0) & (table2["_ci_high"] >= 0)).sum()) == 1
    table2_export = table2.drop(columns=["_ci_low", "_ci_high"])
    for col in ["Attention AUPRC", "Set AUPRC", "Delta Set-Attention"]:
        table2_export[col] = table2_export[col].map(fmt_signed if col == "Delta Set-Attention" else fmt)
    save_table(table2_export, paths["tables"] / "table_set_vs_attention.csv", paths["tables"] / "table_set_vs_attention.tex", "Paired grouped bootstrap for Set Transformer vs gated attention", "lrrrrl")

    effects = []
    for group, label, cid, family in TRANSLATION_SPECS:
        r = contrast_row(summary, cid, "auprc", family=family)
        effects.append({"Group": group, "Contrast": label, "Observed delta": float(r["observed_delta"]), "CI low": float(r["ci_low"]), "CI high": float(r["ci_high"])})
    translation_effects = pd.DataFrame(effects)

    interaction_rows, interaction_full_rows = [], []
    for translator, cid in INTERACTION_SPECS:
        for metric in ["auprc", "auroc", "brier"]:
            r = contrast_row(summary, cid, metric, regime="interaction", family="translation_interaction")
            item = {"Translator": translator, "Metric": metric.upper(), "Interaction": float(r["observed_delta"]),
                    "95% CI": ci_text(r), "Interpretation": "Target-side more harmful" if r["significant_direction"] == "NEGATIVE" else "Inconclusive"}
            interaction_full_rows.append(item)
            if metric in {"auprc", "auroc"}:
                interaction_rows.append(item)
    table3 = pd.DataFrame(interaction_rows)
    table3_full = pd.DataFrame(interaction_full_rows)
    for frame, stem in [(table3, "table_translation_interaction"), (table3_full, "table_translation_interaction_full")]:
        out = frame.copy()
        out["Interaction"] = out["Interaction"].map(fmt_signed)
        save_table(out, paths["tables"] / f"{stem}.csv", paths["tables"] / f"{stem}.tex", "Paired interaction between training-side and target-side translation effects", "llrrl")

    baseline = read_json(registry["BASELINE_RESULTS"] / "metrics.json")
    supervised = pd.read_csv(registry["SUPERVISED_RESULTS"])
    ranking = supervised.loc[supervised["criterion"].eq("ranking")].iloc[0]
    table4 = pd.DataFrame([
        {"Method": "Off-the-shelf NLI", "Target supervision": "No", "AUPRC": float(baseline["AUPRC"]), "AUROC": float(baseline["AUROC"])},
        {"Method": "EN + Attention", "Target supervision": "No", "AUPRC": point(points, "en_attention_ph_pt", "auprc"), "AUROC": point(points, "en_attention_ph_pt", "auroc")},
        {"Method": "EN + Set", "Target supervision": "No", "AUPRC": point(points, "en_set_ph_pt", "auprc"), "AUROC": point(points, "en_set_ph_pt", "auroc")},
        {"Method": "Target-supervised Attention", "Target supervision": "Yes (out-of-fold)", "AUPRC": float(ranking["AUPRC"]), "AUROC": float(ranking["AUROC"])},
    ])
    table4_export = table4.copy()
    for col in ["AUPRC", "AUROC"]:
        table4_export[col] = table4_export[col].map(fmt)
    save_table(table4_export, paths["tables"] / "table_baselines.csv", paths["tables"] / "table_baselines.tex", "Contextual baselines on PublicHearingBR", "llrr")
    with (paths["tables"] / "table_baselines.tex").open("a") as handle:
        handle.write("\\par\\smallskip\\noindent\\textit{The supervised model follows an out-of-fold target-domain protocol and is included only as descriptive context, not as a paired statistical comparison.}\\n")

    pooling_observed = read_json(registry["POOLING_RESULTS"] / "observed_metrics.json")["campaigns"]
    pooling = pd.DataFrame([
        {"Pooling": "Mean", "AUPRC": pooling_observed["mean"]["mean_across_seeds"]["auprc"], "AUROC": pooling_observed["mean"]["mean_across_seeds"]["auroc"], "Brier": pooling_observed["mean"]["mean_across_seeds"]["brier"]},
        {"Pooling": "Max", "AUPRC": pooling_observed["max"]["mean_across_seeds"]["auprc"], "AUROC": pooling_observed["max"]["mean_across_seeds"]["auroc"], "Brier": pooling_observed["max"]["mean_across_seeds"]["brier"]},
        {"Pooling": "Attention", "AUPRC": point(points, "en_attention_ph_pt", "auprc"), "AUROC": point(points, "en_attention_ph_pt", "auroc"), "Brier": point(points, "en_attention_ph_pt", "brier")},
        {"Pooling": "Set", "AUPRC": point(points, "en_set_ph_pt", "auprc"), "AUROC": point(points, "en_set_ph_pt", "auroc"), "Brier": point(points, "en_set_ph_pt", "brier")},
    ])
    pooling_export = pooling.copy()
    for col in pooling_export.columns[1:]:
        pooling_export[col] = pooling_export[col].map(fmt)
    save_table(pooling_export, paths["appendix"] / "table_pooling_ablations.csv", paths["appendix"] / "table_pooling_ablations.tex", "Pooling ablations for EN to PH PT", "lrrr")

    filtered, bertimbau = aggregate_metrics(registry["FILTERED_AGGREGATE"]), aggregate_metrics(registry["BERTIMBAU_AGGREGATE"])
    negative = pd.DataFrame([
        {"Method": "Off-the-shelf NLI", "AUPRC": baseline["AUPRC"], "AUROC": baseline["AUROC"], "Brier": baseline["Brier"]},
        {"Method": "EN + Attention", "AUPRC": point(points, "en_attention_ph_pt", "auprc"), "AUROC": point(points, "en_attention_ph_pt", "auroc"), "Brier": point(points, "en_attention_ph_pt", "brier")},
        {"Method": "PT-NLLB filtered + Attention", "AUPRC": filtered["AUPRC"], "AUROC": filtered["AUROC"], "Brier": filtered["Brier"]},
        {"Method": "PT-NLLB + BERTimbau + Attention", "AUPRC": bertimbau["AUPRC"], "AUROC": bertimbau["AUROC"], "Brier": bertimbau["Brier"]},
    ])
    negative_export = negative.copy()
    for col in negative_export.columns[1:]:
        negative_export[col] = negative_export[col].map(fmt)
    save_table(negative_export, paths["appendix"] / "table_negative_ablations.csv", paths["appendix"] / "table_negative_ablations.tex", "Negative ablations and contextual references", "lrrr")

    brier = pd.DataFrame([{"Condition": row.label, "Attention Brier": point(points, row.attention_id, "brier"), "Set Brier": point(points, row.set_id, "brier")} for row in CONDITIONS.itertuples()])
    brier_export = brier.copy()
    for col in brier_export.columns[1:]:
        brier_export[col] = brier_export[col].map(fmt)
    save_table(brier_export, paths["appendix"] / "table_brier_main_conditions.csv", paths["appendix"] / "table_brier_main_conditions.tex", "Brier scores for the five main conditions", "lrr")

    thresholded = thresholded_frame(summary, registry["PT_NLLB_THRESHOLDED"])
    thresholded["condition_id"] = thresholded["contrast_id"].str.replace("_thresholded", "", regex=False).map({v: k for k, v in AGGREGATOR_CONTRASTS.items()})
    thresholded["order"] = thresholded["condition_id"].map({cid: i for i, cid in enumerate(CONDITIONS["condition_id"])})
    assert len(thresholded) == 50
    thresholded["Condition"] = thresholded["condition_id"].map(dict(zip(CONDITIONS["condition_id"], CONDITIONS["label"])))
    thresholded_export = thresholded.sort_values(["order", "regime", "metric"])[["Condition", "regime", "metric", "point_a", "point_b", "observed_delta", "ci_low", "ci_high", "probability_favorable", "significant_direction"]].rename(columns={"regime": "Regime", "metric": "Metric", "point_a": "Attention", "point_b": "Set", "observed_delta": "Delta Set-Attention", "ci_low": "CI low", "ci_high": "CI high", "probability_favorable": "Probability favorable", "significant_direction": "Verdict"})
    thresholded_export["Verdict"] = thresholded_export["Verdict"].map({"FAVOR_A": "Favors Attention", "FAVOR_B": "Favors Set", "INCONCLUSIVE": "Inconclusive"})
    thresholded_display = thresholded_export.copy()
    for col in ["Attention", "Set", "Delta Set-Attention", "CI low", "CI high", "Probability favorable"]:
        thresholded_display[col] = thresholded_display[col].map(fmt_signed if col == "Delta Set-Attention" else fmt)
    save_table(thresholded_display, paths["appendix"] / "table_thresholded_results.csv", paths["appendix"] / "table_thresholded_results.tex", "Thresholded paired results for the five main conditions", "llrrrrrrrl")
    with (paths["appendix"] / "table_thresholded_results.tex").open("a") as handle:
        handle.write("\\par\\smallskip\\noindent\\textit{FPR is lower-is-better; the verdict uses the corrected significant direction from the frozen artifact.}\\n")

    full = summary.loc[summary["regime"].eq("threshold_free")]
    full_export = full[["contrast_family", "condition_a", "condition_b", "metric", "point_a", "point_b", "observed_delta", "ci_low", "ci_high", "probability_favorable", "significant_direction"]].rename(columns={"contrast_family": "Contrast family", "condition_a": "Condition A", "condition_b": "Condition B", "metric": "Metric", "point_a": "Point A", "point_b": "Point B", "observed_delta": "Delta", "ci_low": "CI low", "ci_high": "CI high", "probability_favorable": "Probability favorable", "significant_direction": "Verdict"})
    full_export["Verdict"] = full_export["Verdict"].map({"FAVOR_A": "Favors A", "FAVOR_B": "Favors B", "INCONCLUSIVE": "Inconclusive", "NEGATIVE": "Negative", "POSITIVE": "Positive"})
    full_display = full_export.copy()
    for col in ["Point A", "Point B", "Delta", "CI low", "CI high", "Probability favorable"]:
        full_display[col] = full_display[col].map(lambda x: "" if pd.isna(x) else fmt_signed(x) if col == "Delta" else fmt(x))
    save_table(full_display, paths["appendix"] / "table_full_paired_bootstrap.csv", paths["appendix"] / "table_full_paired_bootstrap.tex", "Full threshold-free paired bootstrap summary", "llllrrrrrl")

    set_effect = summary.loc[summary["contrast_family"].eq("set_effect_interaction") & summary["metric"].eq("auprc")].copy()
    set_effect["Translator"] = set_effect["contrast_id"].str.extract(r"^(nllb|madlad)")[0].str.upper()
    set_effect_export = set_effect[["Translator", "observed_delta", "ci_low", "ci_high", "significant_direction"]].rename(columns={"observed_delta": "AUPRC interaction", "ci_low": "CI low", "ci_high": "CI high", "significant_direction": "Verdict"})
    set_effect_export["Verdict"] = set_effect_export["Verdict"].map({"INCONCLUSIVE": "Inconclusive", "NEGATIVE": "Negative", "POSITIVE": "Positive"})
    set_effect_display = set_effect_export.copy()
    for col in ["AUPRC interaction", "CI low", "CI high"]:
        set_effect_display[col] = set_effect_display[col].map(fmt_signed)
    save_table(set_effect_display, paths["appendix"] / "table_set_effect_interactions.csv", paths["appendix"] / "table_set_effect_interactions.tex", "Set-effect interactions by translation side", "lrrrl")

    return {
        "table1": table1, "table2": table2, "translation_effects": translation_effects,
        "table3": table3, "table3_full": table3_full, "table4": table4,
        "pooling": pooling, "negative": negative, "brier": brier,
        "thresholded": thresholded, "thresholded_export": thresholded_export,
        "full_export": full_export, "set_effect": set_effect_export,
    }


def thresholded_frame(summary: pd.DataFrame, legacy_path: Path) -> pd.DataFrame:
    frame = summary.loc[
        summary["contrast_family"].eq("thresholded_set_attention") & summary["regime"].isin(["best_f1", "fpr10"])
    ].copy()
    legacy = read_json(legacy_path)
    rows = []
    for regime in ["best_f1", "fpr10"]:
        for metric, values in legacy[regime]["metrics"].items():
            if metric not in {"precision", "recall", "f1", "mcc", "fpr"}:
                continue
            boot = values["bootstrap"]
            if metric == "fpr":
                direction = "FAVOR_A" if boot["ci_lower"] > 0 else "FAVOR_B" if boot["ci_upper"] < 0 else "INCONCLUSIVE"
            else:
                direction = "FAVOR_B" if boot["ci_lower"] > 0 else "FAVOR_A" if boot["ci_upper"] < 0 else "INCONCLUSIVE"
            rows.append({
                "contrast_id": "pt_nllb_set_vs_attention_ph_pt_thresholded", "contrast_family": "thresholded_set_attention",
                "regime": regime, "metric": metric, "point_a": values["attention_mean"], "point_b": values["set_mean"],
                "observed_delta": values["observed_delta"], "ci_low": boot["ci_lower"], "ci_high": boot["ci_upper"],
                "probability_favorable": boot["favorable_probability"], "significant_direction": direction,
            })
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def export_figure(fig: plt.Figure, paths: dict[str, Path], stem: str) -> Path:
    png = paths["figures"] / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if png.stat().st_size == 0:
        raise AssertionError(f"Empty figure output: {png}")
    return png


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_metric_figure(labels: list[str], attention: list[float], set_values: list[float], ylabel: str, title: str, lower_is_better: bool = False, value_labels: bool = False, series_labels: tuple[str, str] = ("Attention", "Set Transformer")) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(labels))
    width = 0.36
    bars_a = ax.bar(x - width / 2, attention, width, label=series_labels[0], color="#4C78A8")
    bars_b = ax.bar(x + width / 2, set_values, width, label=series_labels[1], color="#F58518")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    if lower_is_better:
        ax.text(0.01, 0.98, "Lower is better", transform=ax.transAxes, va="top", color="0.35", fontsize=9)
    if value_labels:
        for bars in [bars_a, bars_b]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
    style_axes(ax)
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.04), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def forest_figure(labels: list[str], deltas: list[float], lows: list[float], highs: list[float], xlabel: str, title: str, lower_is_better: bool = False, groups: list[str] | None = None) -> plt.Figure:
    deltas, lows, highs = np.asarray(deltas), np.asarray(lows), np.asarray(highs)
    if np.any(lows > deltas) or np.any(deltas > highs):
        raise AssertionError("A forest point estimate lies outside its confidence interval.")
    fig, ax = plt.subplots(figsize=(7.4, max(3.8, 0.55 * len(labels) + 1.5)))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(deltas, y, xerr=[deltas - lows, highs - deltas], fmt="o", color="#4C78A8", ecolor="#4C78A8", elinewidth=1.5, capsize=3, markersize=5)
    ax.axvline(0, color="0.25", linewidth=1.0)
    if groups:
        for idx in range(1, len(groups)):
            if groups[idx] != groups[idx - 1]:
                ax.axhline(len(labels) - idx - 0.5, color="0.85", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    if lower_is_better:
        ax.text(0.01, 0.98, "Negative delta favors Set", transform=ax.transAxes, va="top", color="0.35", fontsize=9)
    ax.grid(axis="x", color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def precision_recall_figure(thresholded: pd.DataFrame, regime: str) -> plt.Figure:
    rows = thresholded.loc[thresholded["regime"].eq(regime) & thresholded["metric"].isin(["precision", "recall"])].copy()
    rows["condition_id"] = rows["contrast_id"].str.replace("_thresholded", "", regex=False).map({v: k for k, v in AGGREGATOR_CONTRASTS.items()})
    assert len(rows) == 10
    wide = rows.pivot(index="condition_id", columns="metric", values=["point_a", "point_b"])
    if regime == "best_f1":
        fig, (ax, inset) = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        inset = None
    colors = plt.get_cmap("tab10").colors
    for idx, row in enumerate(CONDITIONS.itertuples()):
        values = wide.loc[row.condition_id]
        ax.scatter(values[("point_a", "recall")], values[("point_a", "precision")], marker="o", s=50, color=colors[idx], label="Attention" if idx == 0 else None)
        ax.scatter(values[("point_b", "recall")], values[("point_b", "precision")], marker="D", s=45, color=colors[idx], label="Set Transformer" if idx == 0 else None)
        if not (regime == "best_f1" and idx < 3):
            ax.annotate(row.short, (values[("point_b", "recall")], values[("point_b", "precision")]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=colors[idx])
    for col in ["point_a", "point_b"]:
        for metric in ["precision", "recall"]:
            assert wide[(col, metric)].between(0, 1).all()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Operating points: {regime}", loc="left", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="lower left")
    if inset is not None:
        inset.set_xlim(0.43, 0.59)
        inset.set_ylim(0.48, 0.62)
        for idx, row in enumerate(CONDITIONS.itertuples()):
            values = wide.loc[row.condition_id]
            for column, marker in [("point_a", "o"), ("point_b", "D")]:
                recall = values[(column, "recall")]
                precision = values[(column, "precision")]
                if 0.44 <= recall <= 0.58 and 0.49 <= precision <= 0.61:
                    inset.scatter(recall, precision, marker=marker, s=28, color=colors[idx], zorder=3)
                    inset.annotate(row.short, (recall, precision), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=colors[idx])
        inset.set_title("Zoom: clustered operating points", fontsize=7, pad=2)
        inset.set_xlabel("Recall", fontsize=8)
        inset.set_ylabel("Precision", fontsize=8)
        inset.tick_params(labelsize=6)
        inset.grid(color="0.90", linewidth=0.6)
        inset.set_axisbelow(True)
        inset.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def tradeoff_figure(thresholded: pd.DataFrame, regime: str) -> plt.Figure:
    rows = thresholded.loc[thresholded["regime"].eq(regime) & thresholded["metric"].isin(["recall", "fpr"])].copy()
    rows["condition_id"] = rows["contrast_id"].str.replace("_thresholded", "", regex=False).map({v: k for k, v in AGGREGATOR_CONTRASTS.items()})
    wide = rows.pivot(index="condition_id", columns="metric", values="observed_delta").loc[CONDITIONS["condition_id"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = plt.get_cmap("tab10").colors
    for idx, row in enumerate(CONDITIONS.itertuples()):
        ax.scatter(wide.loc[row.condition_id, "fpr"], wide.loc[row.condition_id, "recall"], s=52, color=colors[idx])
        ax.annotate(row.short, (wide.loc[row.condition_id, "fpr"], wide.loc[row.condition_id, "recall"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.25", linewidth=1.0)
    ax.set_xlabel("ΔFPR (Set - Attention - positive is worse)")
    ax.set_ylabel("ΔRecall (Set - Attention)")
    ax.set_title(f"Recall-FPR trade-off: {regime}", loc="left", fontsize=10)
    ax.grid(color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def operational_tradeoff_figure(rows: list[dict[str, Any]], x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    offsets = {
        "Off-the-shelf NLI": (8, 8),
        "PublicHearingBR in-domain": (8, 8),
        "EN → PH PT + Gated Attention": (-118, 12),
        "EN → PH PT + Set Transformer": (8, -28),
    }
    for row in rows:
        ax.scatter(row[x_metric], row[y_metric], s=72, color=row["color"], marker=row["marker"], zorder=3)
        ax.annotate(
            f"{row['label']}\nF1={row['f1']:.3f}",
            (row[x_metric], row[y_metric]),
            xytext=offsets[row["label"]],
            textcoords="offset points",
            fontsize=8,
            color=row["color"],
            va="center",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.02, "Frozen best-F1 operating points", transform=ax.transAxes, ha="right", va="bottom", color="0.35", fontsize=8)
    fig.tight_layout()
    return fig


def operational_tradeoff_rows(ctx: dict[str, Any], thresholded: pd.DataFrame) -> list[dict[str, Any]]:
    regime = "best_f1"
    baseline = read_json(ctx["registry"]["OFF_THE_SHELF_THRESHOLDED"])["off_the_shelf"][regime]
    supervised = pd.read_csv(ctx["registry"]["SUPERVISED_RESULTS"])
    in_domain = supervised.loc[supervised["criterion"].eq("max_f1")].iloc[0]
    transfer = thresholded.loc[
        thresholded["contrast_id"].eq("en_set_vs_attention_ph_pt_thresholded") & thresholded["regime"].eq(regime)
    ].set_index("metric")
    required = {"precision", "recall", "f1", "fpr"}
    if not required <= set(transfer.index):
        raise AssertionError("Frozen EN to PH PT thresholded artifact lacks operational metrics.")
    rows = [
        {"label": "Off-the-shelf NLI", "precision": float(baseline["Precision"]), "recall": float(baseline["Recall"]), "f1": float(baseline["F1"]), "fpr": float(baseline["FPR"]), "color": "#E45756", "marker": "o"},
        {"label": "PublicHearingBR in-domain", "precision": float(in_domain["Precision"]), "recall": float(in_domain["Recall"]), "f1": float(in_domain["F1"]), "fpr": float(in_domain["FPR"]), "color": "#54A24B", "marker": "s"},
        {"label": "EN → PH PT + Gated Attention", "precision": float(transfer.loc["precision", "point_a"]), "recall": float(transfer.loc["recall", "point_a"]), "f1": float(transfer.loc["f1", "point_a"]), "fpr": float(transfer.loc["fpr", "point_a"]), "color": "#4C78A8", "marker": "o"},
        {"label": "EN → PH PT + Set Transformer", "precision": float(transfer.loc["precision", "point_b"]), "recall": float(transfer.loc["recall", "point_b"]), "f1": float(transfer.loc["f1", "point_b"]), "fpr": float(transfer.loc["fpr", "point_b"]), "color": "#F58518", "marker": "D"},
    ]
    for row in rows:
        if not all(0 <= row[metric] <= 1 for metric in ["precision", "recall", "f1", "fpr"]):
            raise AssertionError(f"Operational metric outside [0, 1] for {row['label']}.")
    return rows


def build_figures(ctx: dict[str, Any], tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> list[dict[str, str]]:
    points, summary, registry = ctx["points"], ctx["summary"], ctx["registry"]
    records: list[dict[str, str]] = []
    labels = CONDITIONS["label"].tolist()
    def register(stem: str, metric: str, question: str, destination: str, caption: str, source: Path | list[Path], fig: plt.Figure) -> None:
        png = export_figure(fig, paths, stem)
        sources = source if isinstance(source, list) else [source]
        source_refs = [str(path.relative_to(paths["root"].parent.parent)) if path.is_relative_to(paths["root"].parent.parent) else str(path) for path in sources]
        records.append({"figure_id": stem, "filename_png": str(png.relative_to(paths["root"])), "metric": metric, "question": question, "suggested_destination": destination, "short_caption": caption, "source_artifact": " + ".join(source_refs)})
    attention = lambda metric: [point(points, row.attention_id, metric) for row in CONDITIONS.itertuples()]
    set_values = lambda metric: [point(points, row.set_id, metric) for row in CONDITIONS.itertuples()]
    for metric, ylabel, stem, lower in [
        ("auprc", "AUPRC", "figure_main_auprc_comparison", False),
        ("auroc", "AUROC", "figure_main_auroc_comparison", False),
        ("brier", "Brier (lower is better)", "figure_main_brier_comparison", True),
    ]:
        register(stem, metric.upper(), "Complete five-condition metric comparison", "Main" if metric != "brier" else "Appendix", f"{ylabel} across the five main conditions", registry["FINAL_SUMMARY"], grouped_metric_figure(labels, attention(metric), set_values(metric), ylabel, stem, lower, value_labels=False))

    for metric, stem, xlabel, lower in [
        ("auprc", "figure_set_vs_attention_auprc_forest", "Delta Set - Attention (AUPRC)", False),
        ("auroc", "figure_set_vs_attention_auroc_forest", "Delta Set - Attention (AUROC)", False),
        ("brier", "figure_set_vs_attention_brier_forest", "Delta Set - Attention (Brier)", True),
    ]:
        rows = [contrast_row(summary, AGGREGATOR_CONTRASTS[row.condition_id], metric, family="aggregator") for row in CONDITIONS.itertuples()]
        register(stem, metric.upper(), "Set versus Attention paired effect", "Main" if metric == "auprc" else "Appendix", f"{metric.upper()} Set - Attention - negative delta favors Set" if lower else f"{metric.upper()} Set - Attention", registry["FINAL_SUMMARY"], forest_figure(labels, [r["observed_delta"] for r in rows], [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], xlabel, stem, lower))

    for metric, stem, xlabel in [
        ("auprc", "figure_translation_effects_auprc_forest", "AUPRC difference (Set Transformer)"),
        ("auroc", "figure_translation_effects_auroc_forest", "AUROC difference (Set Transformer)"),
    ]:
        rows = [contrast_row(summary, cid, metric, family=family) for _, _, cid, family in TRANSLATION_SPECS]
        groups = [group for group, _, _, _ in TRANSLATION_SPECS]
        register(stem, metric.upper(), "Training-side and target-side translation effects", "Main" if metric == "auprc" else "Appendix", f"{metric.upper()} translation-side effects under the frozen paired protocol", registry["FINAL_SUMMARY"], forest_figure([label for _, label, _, _ in TRANSLATION_SPECS], [r["observed_delta"] for r in rows], [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], xlabel, stem, groups=groups))

    interaction_rows = []
    for translator, cid in INTERACTION_SPECS:
        for metric in ["auprc", "auroc"]:
            interaction_rows.append((translator, metric.upper(), contrast_row(summary, cid, metric, regime="interaction", family="translation_interaction")))
    register("figure_translation_side_interaction_forest", "AUPRC/AUROC", "Translation-side interaction", "Main", "Negative values mean target-side translation is more harmful than training-side translation.", registry["FINAL_SUMMARY"], forest_figure([f"{t} {m}" for t, m, _ in interaction_rows], [float(r["observed_delta"]) for _, _, r in interaction_rows], [float(r["ci_low"]) for _, _, r in interaction_rows], [float(r["ci_high"]) for _, _, r in interaction_rows], "Interaction delta", "figure_translation_side_interaction_forest", groups=[t for t, _, _ in interaction_rows]))

    thresholded = tables["thresholded"]
    operational_rows = operational_tradeoff_rows(ctx, thresholded)
    operational_sources = [ctx["registry"]["OFF_THE_SHELF_THRESHOLDED"], ctx["registry"]["SUPERVISED_RESULTS"], ctx["registry"]["PT_NLLB_THRESHOLDED"]]
    register("figure_operational_tradeoff_en_to_phpt", "Precision/Recall/F1", "Operational comparison for EN to PublicHearingBR PT", "Main", "Frozen best-F1 Precision-Recall operating points for four reference systems.", operational_sources, operational_tradeoff_figure(operational_rows, "recall", "precision", "Operational trade-off: EN → PublicHearingBR PT", "Recall", "Precision"))
    register("figure_recall_fpr_en_to_phpt", "Recall/FPR", "Operational recall-FPR comparison for EN to PublicHearingBR PT", "Appendix", "Frozen best-F1 recall-FPR operating points for four reference systems.", operational_sources, operational_tradeoff_figure(operational_rows, "fpr", "recall", "Operational FPR-recall trade-off: EN → PublicHearingBR PT", "FPR", "Recall"))
    for regime in ["best_f1", "fpr10"]:
        register(f"figure_precision_recall_{regime}", "Precision/Recall", "Thresholded precision-recall operating points", "Main" if regime == "best_f1" else "Appendix", f"Precision-Recall operating points ({regime})", registry["FINAL_SUMMARY"], precision_recall_figure(thresholded, regime))
        register(f"figure_recall_fpr_tradeoff_{regime}", "Recall/FPR", "Descriptive thresholded recall-FPR trade-off", "Appendix", f"Recall-FPR trade-off ({regime}) - descriptive point deltas only", registry["FINAL_SUMMARY"], tradeoff_figure(thresholded, regime))
        for metric in ["f1", "recall", "fpr"]:
            rows = thresholded.loc[thresholded["regime"].eq(regime) & thresholded["metric"].eq(metric)]
            rows = rows.set_index("condition_id").loc[CONDITIONS["condition_id"]]
            register(f"figure_{metric}_{regime}_comparison", metric.upper(), f"Thresholded {metric} comparison", "Appendix", f"{metric.upper()} comparison ({regime})" + (" - lower is better" if metric == "fpr" else ""), registry["FINAL_SUMMARY"], grouped_metric_figure(labels, rows["point_a"].tolist(), rows["point_b"].tolist(), f"{metric.upper()} ({'lower is better' if metric == 'fpr' else 'higher is better'})", f"figure_{metric}_{regime}_comparison", metric == "fpr"))

    pooling = tables["pooling"]
    baseline_metrics = read_json(registry["BASELINE_RESULTS"] / "metrics.json")
    pooling_labels = ["NLI off-the-shelf", "Mean", "Max", "Gated Attention MIL", "Set Transformer"]
    pooling_auprc = [float(baseline_metrics["AUPRC"]), *pooling["AUPRC"].astype(float).tolist()]
    pooling_auroc = [float(baseline_metrics["AUROC"]), *pooling["AUROC"].astype(float).tolist()]
    register("figure_pooling_ablations", "AUPRC/AUROC", "Cross-lingual zero-shot transfer from RAGTruth EN to PublicHearingBR PT", "Appendix", "AUPRC and AUROC for NLI off-the-shelf and pooling methods under RAGTruth EN to PublicHearingBR PT zero-shot transfer.", [registry["BASELINE_RESULTS"] / "metrics.json", registry["POOLING_RESULTS"] / "observed_metrics.json"], grouped_metric_figure(pooling_labels, pooling_auprc, pooling_auroc, "Metric value", "RAGTruth EN → PublicHearingBR PT: zero-shot pooling comparison", False, value_labels=True, series_labels=("AUPRC", "AUROC")))
    baseline = tables["table4"]
    register("figure_contextual_baselines", "AUPRC/AUROC", "Contextual baselines on PublicHearingBR", "Appendix", "Supervised result is descriptive context only.", registry["BASELINE_RESULTS"] / "metrics.json", grouped_metric_figure(baseline["Method"].tolist(), baseline["AUPRC"].tolist(), baseline["AUROC"].tolist(), "Metric value", "Contextual baselines", False, series_labels=("AUPRC", "AUROC")))
    return records


def build_summary_and_claims(ctx: dict[str, Any], tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> pd.DataFrame:
    points, summary, output_root = ctx["points"], ctx["summary"], paths["root"]
    table1, table2, table4 = tables["table1"], tables["table2"], tables["table4"]
    best_row = table1.loc[table1["Set AUPRC"].idxmax()]
    target_nllb = contrast_row(summary, "en_set_ph_en_nllb_vs_pt", "auprc", family="target_translation")
    target_madlad = contrast_row(summary, "en_set_ph_en_madlad_vs_pt", "auprc", family="target_translation")
    training_nllb = contrast_row(summary, "pt_nllb_set_vs_en_ph_pt", "auprc", family="training_translation")
    training_madlad = contrast_row(summary, "pt_madlad_set_vs_en_ph_pt", "auprc", family="training_translation")
    interaction_nllb = contrast_row(summary, "nllb_set_target_minus_training", "auprc", regime="interaction", family="translation_interaction")
    interaction_madlad = contrast_row(summary, "madlad_set_target_minus_training", "auprc", regime="interaction", family="translation_interaction")
    higher_count = int((table2["Delta Set-Attention"] > 0).sum())
    favorable_count = int((table2["_ci_low"] > 0).sum())
    paper_summary = "\n".join([
        "# Paper-ready numerical summary", "",
        "## Best direct transfer",
        f"- EN + Set → PH PT: AUPRC {fmt(best_row['Set AUPRC'])}; AUROC {fmt(best_row['Set AUROC'])}.", "",
        "## Set effect",
        f"- Set has higher AUPRC in {higher_count}/5 main conditions.",
        f"- The paired 95% CI favors Set in {favorable_count}/5 conditions; {5 - favorable_count} is inconclusive.", "",
        "## Target translation effect — Set",
        f"- NLLB: delta {fmt_signed(target_nllb['observed_delta'])}, 95% CI {ci_text(target_nllb)}.",
        f"- MADLAD: delta {fmt_signed(target_madlad['observed_delta'])}, 95% CI {ci_text(target_madlad)}.", "",
        "## Training translation effect — Set",
        f"- NLLB: delta {fmt_signed(training_nllb['observed_delta'])}, 95% CI {ci_text(training_nllb)}.",
        f"- MADLAD: delta {fmt_signed(training_madlad['observed_delta'])}, 95% CI {ci_text(training_madlad)}.", "",
        "## Translation-side interaction",
        f"- NLLB AUPRC interaction: {fmt_signed(interaction_nllb['observed_delta'])}, 95% CI {ci_text(interaction_nllb)}; target-side more harmful.",
        f"- MADLAD AUPRC interaction: {fmt_signed(interaction_madlad['observed_delta'])}, 95% CI {ci_text(interaction_madlad)}; target-side more harmful.", "",
        "These are associations under the frozen paired protocol; no causal interpretation is implied.",
    ]) + "\n"
    (output_root / "paper_ready_summary.md").write_text(paper_summary)

    claims = pd.DataFrame([
        {"claim_id": "C1", "claim": "Direct EN->PT transfer works well", "evidence": f"EN + Set is the highest-AUPRC main condition ({fmt(best_row['Set AUPRC'])}).", "statistical_status": "SUPPORTED", "allowed_wording": "Direct EN→PT transfer is the strongest evaluated transfer strategy.", "destination": "Results"},
        {"claim_id": "C2", "claim": "Set has higher AUPRC across the main conditions", "evidence": f"Higher point estimate in {higher_count}/5; paired CI favors Set in {favorable_count}/5.", "statistical_status": "SUPPORTED", "allowed_wording": "Set has higher AUPRC in all five main conditions, with paired CIs favoring Set in four.", "destination": "Results"},
        {"claim_id": "C3", "claim": "Training translation does not provide consistent overall improvement over EN", "evidence": f"NLLB {fmt_signed(training_nllb['observed_delta'])} {ci_text(training_nllb)}; MADLAD {fmt_signed(training_madlad['observed_delta'])} {ci_text(training_madlad)}.", "statistical_status": "SUPPORTED", "allowed_wording": "Explicit training translation does not provide a consistent improvement over EN training.", "destination": "Discussion"},
        {"claim_id": "C4", "claim": "Target translation substantially degrades transfer", "evidence": f"Set target deltas: NLLB {fmt_signed(target_nllb['observed_delta'])} {ci_text(target_nllb)}; MADLAD {fmt_signed(target_madlad['observed_delta'])} {ci_text(target_madlad)}.", "statistical_status": "SUPPORTED", "allowed_wording": "Target translation produces large ranking degradation for both translators.", "destination": "Results"},
        {"claim_id": "C5", "claim": "Target-side translation is more harmful than training-side translation", "evidence": f"Set interactions: NLLB {fmt_signed(interaction_nllb['observed_delta'])} {ci_text(interaction_nllb)}; MADLAD {fmt_signed(interaction_madlad['observed_delta'])} {ci_text(interaction_madlad)}.", "statistical_status": "SUPPORTED", "allowed_wording": "Target-side translation is associated with larger degradation than training-side translation.", "destination": "Results"},
        {"claim_id": "C6", "claim": "EN+Set approaches target-supervised Attention descriptively", "evidence": f"EN + Set ({fmt(table4.loc[table4['Method'].eq('EN + Set'), 'AUPRC'].iloc[0])}/{fmt(table4.loc[table4['Method'].eq('EN + Set'), 'AUROC'].iloc[0])}) and target-supervised Attention ({fmt(table4.loc[table4['Method'].eq('Target-supervised Attention'), 'AUPRC'].iloc[0])}/{fmt(table4.loc[table4['Method'].eq('Target-supervised Attention'), 'AUROC'].iloc[0])}).", "statistical_status": "DESCRIPTIVE_ONLY", "allowed_wording": "EN+Set approaches the target-supervised Attention result descriptively; no paired equivalence or superiority claim is made.", "destination": "Results"},
    ])
    if claims.loc[claims["claim_id"].eq("C6"), "statistical_status"].iloc[0] != "DESCRIPTIVE_ONLY":
        raise AssertionError("Supervised comparison must remain descriptive only.")
    claims.to_csv(output_root / "claim_support.csv", index=False)
    return claims


def write_inventories(ctx: dict[str, Any], figure_records: list[dict[str, str]], paths: dict[str, Path]) -> pd.DataFrame:
    figure_inventory = pd.DataFrame(figure_records)
    figure_inventory.to_csv(paths["root"] / "figure_inventory.csv", index=False)
    table_pairs = [
        ("table_main_transfer", "Main transfer strategies", "Main"),
        ("table_set_vs_attention", "Set versus Attention AUPRC", "Main"),
        ("table_translation_interaction", "Translation-side interaction", "Main"),
        ("table_translation_interaction_full", "Translation-side interaction including Brier", "Appendix"),
        ("table_baselines", "Contextual baselines", "Appendix"),
        ("table_pooling_ablations", "Pooling ablations", "Appendix"),
        ("table_negative_ablations", "Negative ablations", "Appendix"),
        ("table_brier_main_conditions", "Calibration/Brier", "Appendix"),
        ("table_thresholded_results", "Thresholded comparisons", "Appendix"),
        ("table_full_paired_bootstrap", "Full paired bootstrap", "Appendix"),
        ("table_set_effect_interactions", "Set-effect interactions", "Appendix"),
    ]
    table_inventory = []
    for stem, question, destination in table_pairs:
        base = paths["tables"] if stem in {"table_main_transfer", "table_set_vs_attention", "table_translation_interaction", "table_translation_interaction_full", "table_baselines"} else paths["appendix"]
        table_inventory.append({"table_id": stem, "filename_csv": str((base / f"{stem}.csv").relative_to(paths["root"])), "filename_tex": str((base / f"{stem}.tex").relative_to(paths["root"])), "question": question, "suggested_destination": destination, "source_artifact": str(ctx["registry"]["FINAL_SUMMARY"].relative_to(ctx["root"]))})
    pd.DataFrame(table_inventory).to_csv(paths["root"] / "table_inventory.csv", index=False)
    return figure_inventory


def validate_outputs(ctx: dict[str, Any], tables: dict[str, pd.DataFrame], figure_records: list[dict[str, str]], paths: dict[str, Path]) -> None:
    summary = ctx["summary"]
    table2 = tables["table2"]
    assert (tables["table1"]["Set AUPRC"] > tables["table1"]["Attention AUPRC"]).all()
    assert len(table2) == 5 and int((table2["_ci_low"] > 0).sum()) == 4
    assert int(((table2["_ci_low"] <= 0) & (table2["_ci_high"] >= 0)).sum()) == 1
    for cid in ["en_set_ph_en_nllb_vs_pt", "en_set_ph_en_madlad_vs_pt"]:
        r = contrast_row(summary, cid, "auprc", family="target_translation")
        assert r["observed_delta"] < 0 and r["ci_high"] < 0
    r = contrast_row(summary, "pt_nllb_set_vs_en_ph_pt", "auprc", family="training_translation")
    assert r["observed_delta"] < 0 and r["ci_high"] < 0
    r = contrast_row(summary, "pt_madlad_set_vs_en_ph_pt", "auprc", family="training_translation")
    assert r["observed_delta"] < 0 and r["ci_low"] < 0 < r["ci_high"]
    for cid in ["nllb_set_target_minus_training", "madlad_set_target_minus_training"]:
        r = contrast_row(summary, cid, "auprc", regime="interaction", family="translation_interaction")
        assert r["observed_delta"] < 0 and r["ci_high"] < 0
    expected = [
        paths["tables"] / "table_main_transfer.csv", paths["tables"] / "table_main_transfer.tex",
        paths["tables"] / "table_set_vs_attention.csv", paths["tables"] / "table_set_vs_attention.tex",
        paths["tables"] / "table_translation_interaction.csv", paths["tables"] / "table_translation_interaction.tex",
        paths["tables"] / "table_translation_interaction_full.csv", paths["tables"] / "table_translation_interaction_full.tex",
        paths["tables"] / "table_baselines.csv", paths["tables"] / "table_baselines.tex",
        paths["appendix"] / "table_pooling_ablations.csv", paths["appendix"] / "table_pooling_ablations.tex",
        paths["appendix"] / "table_negative_ablations.csv", paths["appendix"] / "table_negative_ablations.tex",
        paths["appendix"] / "table_brier_main_conditions.csv", paths["appendix"] / "table_brier_main_conditions.tex",
        paths["appendix"] / "table_thresholded_results.csv", paths["appendix"] / "table_thresholded_results.tex",
        paths["appendix"] / "table_full_paired_bootstrap.csv", paths["appendix"] / "table_full_paired_bootstrap.tex",
        paths["appendix"] / "table_set_effect_interactions.csv", paths["appendix"] / "table_set_effect_interactions.tex",
        paths["root"] / "paper_ready_summary.md", paths["root"] / "claim_support.csv",
        paths["root"] / "figure_inventory.csv", paths["root"] / "table_inventory.csv",
    ]
    expected += [Path(record["filename_png"]) if Path(record["filename_png"]).is_absolute() else paths["root"] / record["filename_png"] for record in figure_records]
    for path in expected:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty reporting output: {path}")
    for path in paths["root"].rglob("*.csv"):
        if pd.read_csv(path).empty:
            raise AssertionError(f"Empty CSV: {path}")
    if list(paths["figures"].glob("*.pdf")):
        raise AssertionError("PDF outputs are forbidden; reporting is PNG-only.")


def generate_paper_results(root: Path | None = None, output_root: Path | None = None, official_run: Path | str | None = None) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    registry = build_source_registry(root, official_run=official_run, output_root=output_root)
    ctx = validate_sources(registry)
    ctx["root"] = root
    ctx["points"] = collect_points(ctx["summary"])
    paths = make_output_dirs(registry["OUTPUT_ROOT"])
    tables = build_tables(ctx, paths)
    figure_records = build_figures(ctx, tables, paths)
    claims = build_summary_and_claims(ctx, tables, paths)
    figure_inventory = write_inventories(ctx, figure_records, paths)
    validate_outputs(ctx, tables, figure_records, paths)
    print(f"Reporting complete: {len(figure_inventory)} figures, {len(claims)} claims, official run {OFFICIAL_RUN_ID}")
    print(f"Outputs: {paths['root']}")
    print("Scientific computation: none; predictions: not used; bootstrap: not executed; thresholds: not recomputed.")
    return {"context": ctx, "paths": paths, "tables": tables, "figure_inventory": figure_inventory, "claims": claims}
