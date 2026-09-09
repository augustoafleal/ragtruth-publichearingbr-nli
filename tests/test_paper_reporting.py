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
    thresholded_frame,
    validate_sources,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.frozen_artifacts
def test_official_provenance_and_superseded_rejection():
    ctx = validate_sources(build_source_registry(ROOT))
    assert ctx["registry"]["OFFICIAL_BOOTSTRAP_RUN"].name == OFFICIAL_RUN_ID
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
    }
    assert expected_stems <= set(first["figure_inventory"]["figure_id"])
    assert len(first["figure_inventory"]) == 23
    assert len(pd.read_csv(tmp_path / "figure_inventory.csv")) == 23
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
