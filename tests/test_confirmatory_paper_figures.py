import csv
import math
import shutil
from pathlib import Path

import pytest

from scripts.regenerate_confirmatory_paper_figures import (
    regenerate_confirmatory_figures,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_FIGURES = (
    "figure_translation_effects_auprc_forest",
    "figure_main_auprc_comparison",
    "figure_set_vs_attention_auprc_forest",
)
EXPECTED_VALUES = {
    "main": {
        "en_ph_pt": (0.5949, 0.6035),
        "pt_nllb_ph_pt": (0.5370, 0.5658),
        "en_ph_en_nllb": (0.3900, 0.4062),
    },
    "translation": {
        "pt_nllb_set_vs_en_ph_pt": (-0.0377, -0.0587, -0.0178),
        "en_set_ph_en_nllb_vs_pt": (-0.1973, -0.2270, -0.1633),
    },
    "set_vs_attention": {
        "en_ph_pt": (0.0086, -0.0046, 0.0231),
        "pt_nllb_ph_pt": (0.0289, 0.0130, 0.0442),
        "en_ph_en_nllb": (0.0162, 0.0003, 0.0335),
    },
}


@pytest.mark.frozen_artifacts
def test_confirmatory_figures_exclude_madlad_and_preserve_other_inventory(tmp_path):
    output_root = tmp_path / "paper"
    (output_root / "figures").mkdir(parents=True)
    source_inventory = ROOT / "results/paper/figure_inventory.csv"
    shutil.copy(source_inventory, output_root / "figure_inventory.csv")
    with source_inventory.open(newline="", encoding="utf-8") as handle:
        before_inventory = {row["figure_id"]: row for row in csv.DictReader(handle)}
    result = regenerate_confirmatory_figures(root=ROOT, output_root=output_root)
    assert result["changed_pngs"] == sorted(f"{stem}.png" for stem in TARGET_FIGURES)
    assert [record["figure_id"] for record in result["records"]] == [
        "figure_main_auprc_comparison",
        "figure_set_vs_attention_auprc_forest",
        "figure_translation_effects_auprc_forest",
    ]
    assert result["records"][0]["conditions"] == ["EN → PH PT", "PT-NLLB → PH PT", "EN → PH EN-NLLB"]
    assert result["records"][1]["conditions"] == ["EN → PH PT", "PT-NLLB → PH PT", "EN → PH EN-NLLB"]
    assert result["records"][2]["conditions"] == ["PT-NLLB training - EN training", "PH EN-NLLB target - PH PT target"]
    assert all(not record["madlad_present"] for record in result["records"])
    assert (output_root / "figures/figure_translation_effects_auprc_forest.png").stat().st_size > 0
    for section, expected in EXPECTED_VALUES.items():
        for key, expected_values in expected.items():
            actual_values = result["values"][section][key]
            assert all(math.isclose(float(actual), float(reference), rel_tol=0.0, abs_tol=5e-5) for actual, reference in zip(actual_values, expected_values))
    assert set(path.name for path in (output_root / "figures").glob("*.png")) == {f"{stem}.png" for stem in TARGET_FIGURES}
    with (output_root / "figure_inventory.csv").open(newline="", encoding="utf-8") as handle:
        inventory = {row["figure_id"]: row for row in csv.DictReader(handle)}
    for figure_id, row in before_inventory.items():
        if figure_id not in TARGET_FIGURES:
            assert inventory[figure_id] == row
    assert "three NLLB confirmatory conditions" in inventory["figure_main_auprc_comparison"]["short_caption"]
    assert "three NLLB confirmatory conditions" in inventory["figure_set_vs_attention_auprc_forest"]["short_caption"]
    assert "NLLB translation-side effects" in inventory["figure_translation_effects_auprc_forest"]["short_caption"]
    assert all("MADLAD" not in " ".join(row.values()) for row in inventory.values() if row["figure_id"] in TARGET_FIGURES)
