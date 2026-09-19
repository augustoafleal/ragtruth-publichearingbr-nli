from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ragtruth_transfer.paper_reporting import (
    OFFICIAL_RUN_ID,
    SUPERSEDED_RUN_ID,
    build_source_registry,
    build_tables,
    collect_points,
    export_figure,
    forest_figure,
    generate_paper_results,
    make_output_dirs,
    operational_tradeoff_rows,
    precision_recall_figure,
    grouped_metric_figure,
    llm_comparison_rows,
    llm_f1_fpr_figure,
    llm_metric_delta_frame,
    llm_metric_delta_heatmap,
    thresholded_frame,
    validate_sources,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.frozen_artifacts
def test_official_provenance_and_superseded_rejection():
    ctx = validate_sources(build_source_registry(ROOT))
    assert ctx["registry"]["OFFICIAL_BOOTSTRAP_RUN"].name == OFFICIAL_RUN_ID
    set_config = ctx["registry"]["SUPERVISED_SET_RUN_CONFIG"]
    assert set_config.is_file()
    assert not ctx["orientation_mismatches"]
    with pytest.raises(ValueError, match=SUPERSEDED_RUN_ID):
        build_source_registry(ROOT, official_run=ROOT / "runs/publichearing_final_paired_bootstrap" / SUPERSEDED_RUN_ID)


@pytest.mark.frozen_artifacts
def test_main_tables_thresholded_invariants_and_forest_smoke(tmp_path):
    ctx = validate_sources(build_source_registry(ROOT))
    ctx["root"] = ROOT
    ctx["points"] = collect_points(ctx["summary"])
    paths = make_output_dirs(tmp_path)
    stale_pdf = paths["figures"] / "stale.pdf"
    stale_pdf.write_bytes(b"stale")
    make_output_dirs(tmp_path)
    assert not stale_pdf.exists()
    tables = build_tables(ctx, paths)
    assert len(tables["table1"]) == 5
    assert len(tables["table2"]) == 5
    assert set(tables["table4"]["Method"]) >= {
        "Gated Attention MIL - zero-shot",
        "Set Transformer - zero-shot",
        "Gated Attention MIL - in-domain OOF",
        "Set Transformer - in-domain OOF",
    }
    assert not {"CI low", "CI high", "95% CI"} & set(tables["table4"].columns)
    set_metrics = pd.read_csv(ctx["registry"]["SUPERVISED_SET_RESULTS"])
    assert set(set_metrics["criterion"]) == {"ranking", "max_f1", "fpr_operational"}
    ranking_set = set_metrics.loc[set_metrics["criterion"].eq("ranking")].iloc[0]
    assert ranking_set["AUPRC"] == pytest.approx(0.618284)
    assert ranking_set["AUROC"] == pytest.approx(0.880518)
    assert ranking_set["Brier"] == pytest.approx(0.076327)
    set_max_f1 = set_metrics.loc[set_metrics["criterion"].eq("max_f1")].iloc[0]
    assert set_max_f1["Precision"] == pytest.approx(0.592357)
    assert set_max_f1["Recall"] == pytest.approx(0.556886)
    assert set_max_f1["F1"] == pytest.approx(0.574074)
    assert set_max_f1["FPR"] == pytest.approx(0.051419, abs=1e-6)
    assert (tables["table2"]["Delta Set-Attention"] > 0).all()
    assert int((tables["table2"]["_ci_low"] > 0).sum()) == 4
    assert int(((tables["table2"]["_ci_low"] <= 0) & (tables["table2"]["_ci_high"] >= 0)).sum()) == 1

    thresholded = thresholded_frame(ctx["summary"], ctx["registry"]["PT_NLLB_THRESHOLDED"])
    assert len(thresholded) == 50
    for metric in ["precision", "recall"]:
        assert thresholded.loc[thresholded["metric"].eq(metric), ["point_a", "point_b"]].apply(lambda col: col.between(0, 1).all()).all()
    assert thresholded.loc[thresholded["metric"].eq("fpr"), "significant_direction"].notna().all()
    fpr10_en_pt = thresholded.loc[
        thresholded["contrast_label"].eq("EN Set minus Attention at PH PT")
        & thresholded["regime"].eq("fpr10")
        & thresholded["metric"].eq("fpr")
    ].iloc[0]
    assert fpr10_en_pt["observed_delta"] > 0
    assert fpr10_en_pt["significant_direction"] == "FAVOR_A"

    pr_fig = precision_recall_figure(thresholded, "best_f1")
    assert len(pr_fig.axes) == 2
    plt.close(pr_fig)

    operational = operational_tradeoff_rows(ctx, thresholded)
    assert [row["label"] for row in operational] == [
        "Off-the-shelf NLI",
        "PublicHearingBR in-domain",
        "EN → PH PT + Gated Attention",
        "EN → PH PT + Set Transformer",
    ]
    assert all(row["f1"] is not None for row in operational)

    pooling_fig = grouped_metric_figure(["NLI", "Mean", "Max", "Attention", "Set"], [0.3375, 0.5735, 0.5704, 0.5949, 0.6035], [0.7680, 0.8661, 0.8840, 0.8788, 0.8862], "Metric value", "Pooling", value_labels=True, series_labels=("AUPRC", "AUROC"))
    assert len(pooling_fig.axes[0].patches) == 10
    plt.close(pooling_fig)

    fig = forest_figure(["test"], [0.1], [0.0], [0.2], "delta", "test")
    png = export_figure(fig, paths, "forest_smoke")
    assert png.stat().st_size > 0
    assert not list(paths["figures"].glob("*.pdf"))


@pytest.mark.frozen_artifacts
def test_full_reporting_registry_and_repeatability(tmp_path):
    first = generate_paper_results(root=ROOT, output_root=tmp_path)
    expected_stems = {
        "figure_main_auprc_comparison",
        "figure_main_auroc_comparison",
        "figure_set_vs_attention_auprc_forest",
        "figure_translation_effects_auprc_forest",
        "figure_translation_side_interaction_forest",
        "figure_precision_recall_best_f1",
        "figure_precision_recall_fpr10",
        "figure_f1_best_f1_comparison",
        "figure_f1_fpr10_comparison",
        "figure_recall_best_f1_comparison",
        "figure_recall_fpr10_comparison",
        "figure_fpr_best_f1_comparison",
        "figure_fpr_fpr10_comparison",
        "figure_recall_fpr_tradeoff_best_f1",
        "figure_recall_fpr_tradeoff_fpr10",
        "figure_pooling_ablations",
        "figure_contextual_baselines",
        "figure_operational_tradeoff_en_to_phpt",
        "figure_recall_fpr_en_to_phpt",
        "figure_llm_f1_vs_fpr",
        "figure_llm_metric_deltas_heatmap",
    }
    assert expected_stems <= set(first["figure_inventory"]["figure_id"])
    contextual = first["figure_inventory"].loc[first["figure_inventory"]["figure_id"].eq("figure_contextual_baselines"), "source_artifact"].iloc[0]
    assert "results/publichearing_lora_set_transformer_mil/80eb95b5698f4a75/outputs/overall_oof_metrics.csv" in contextual
    assert len(first["figure_inventory"]) == 25
    assert len(pd.read_csv(tmp_path / "figure_inventory.csv")) == 25
    assert "filename_pdf" not in pd.read_csv(tmp_path / "figure_inventory.csv").columns
    assert not list((tmp_path / "figures").glob("*.pdf"))
    assert len(pd.read_csv(tmp_path / "table_inventory.csv")) == 11

    tracked = [
        path for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".tex", ".md"}
    ]
    before = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    generate_paper_results(root=ROOT, output_root=tmp_path)
    after = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    assert before == after


@pytest.mark.frozen_artifacts
def test_llm_comparison_figures_use_frozen_rows_and_delta_conventions():
    ctx = validate_sources(build_source_registry(ROOT))
    rows = llm_comparison_rows(ctx)
    assert len(rows) == 14
    assert len(rows.loc[rows["evaluation_protocol"].eq("paper stored binary judgment")]) == 12
    reference = rows.loc[
        rows["method"].eq("Set Transformer") & rows["prompt_or_criterion"].eq("Max. F1 threshold")
    ]
    assert len(reference) == 1
    assert reference.iloc[0]["evaluation_protocol"] == "official frozen zero-shot point estimate"
    assert set(rows["evaluation_population"]) == {4235}
    assert set(rows["positive_labels"]) == {501}
    assert not {"AUPRC", "AUROC", "Brier"} & set(rows.columns)

    delta = llm_metric_delta_frame(rows)
    assert len(delta) == 12
    ref = reference.iloc[0]
    source = rows.loc[rows["evaluation_protocol"].eq("paper stored binary judgment")].set_index("model_key")
    delta = delta.set_index("model_key")
    for model_key, row in source.iterrows():
        for metric in ["Precision", "Recall", "F1", "MCC"]:
            assert delta.loc[model_key, metric] == pytest.approx(float(ref[metric]) - float(row[metric]))
        assert delta.loc[model_key, "FPR"] == pytest.approx(float(row["FPR"]) - float(ref["FPR"]))
    assert {"Precision", "Recall", "F1", "FPR", "MCC"} <= set(delta.columns)

    scatter = llm_f1_fpr_figure(rows)
    assert len(scatter.axes[0].collections) == 14
    for _, row in rows.iterrows():
        assert any(
            point.get_offsets()[0, 0] == pytest.approx(float(row["FPR"]))
            and point.get_offsets()[0, 1] == pytest.approx(float(row["F1"]))
            for point in scatter.axes[0].collections
            if len(point.get_offsets())
        )
    plt.close(scatter)

    heatmap = llm_metric_delta_heatmap(rows)
    assert len(heatmap.axes[0].images) == 1
    assert heatmap.axes[0].images[0].get_array().shape == (12, 5)
    plt.close(heatmap)
