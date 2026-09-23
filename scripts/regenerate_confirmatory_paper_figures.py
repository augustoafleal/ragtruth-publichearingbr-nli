from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from ragtruth_transfer.paper_reporting import (
    AGGREGATOR_CONTRASTS,
    CONDITIONS,
    OFFICIAL_RUN_ID,
    TRANSLATION_SPECS,
    build_source_registry,
    collect_points,
    contrast_row,
    export_figure,
    forest_figure,
    grouped_metric_figure,
    point,
    validate_sources,
)


TARGET_FIGURES = (
    "figure_translation_effects_auprc_forest",
    "figure_main_auprc_comparison",
    "figure_set_vs_attention_auprc_forest",
)
CONFIRMATORY_CONDITION_IDS = ("en_ph_pt", "pt_nllb_ph_pt", "en_ph_en_nllb")
CONFIRMATORY_TRANSLATION_SPECS = tuple(spec for spec in TRANSLATION_SPECS if "NLLB" in spec[1])
EXPECTED = {
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
TOLERANCE = 5e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_pngs(path: Path) -> dict[str, str]:
    return {item.name: sha256(item) for item in path.glob("*.png")}


def confirmatory_conditions() -> Any:
    return CONDITIONS.loc[CONDITIONS["condition_id"].isin(CONFIRMATORY_CONDITION_IDS)].copy()


def validate_expected_values(ctx: dict[str, Any]) -> dict[str, Any]:
    points = ctx["points"]
    summary = ctx["summary"]
    conditions = confirmatory_conditions()
    if tuple(conditions["condition_id"]) != CONFIRMATORY_CONDITION_IDS:
        raise ValueError(f"Unexpected confirmatory condition order: {tuple(conditions['condition_id'])}")
    actual_main = {
        row.condition_id: (point(points, row.attention_id, "auprc"), point(points, row.set_id, "auprc"))
        for row in conditions.itertuples()
    }
    actual_translation = {}
    for _, _, contrast_id, family in CONFIRMATORY_TRANSLATION_SPECS:
        row = contrast_row(summary, contrast_id, "auprc", family=family)
        actual_translation[contrast_id] = (float(row["observed_delta"]), float(row["ci_low"]), float(row["ci_high"]))
    actual_set_vs_attention = {}
    for row in conditions.itertuples():
        result = contrast_row(summary, AGGREGATOR_CONTRASTS[row.condition_id], "auprc", family="aggregator")
        actual_set_vs_attention[row.condition_id] = (float(result["observed_delta"]), float(result["ci_low"]), float(result["ci_high"]))
    checks = [("main", EXPECTED["main"], actual_main), ("translation", EXPECTED["translation"], actual_translation), ("set_vs_attention", EXPECTED["set_vs_attention"], actual_set_vs_attention)]
    for section, expected, actual in checks:
        if set(expected) != set(actual):
            raise ValueError(f"Frozen artifact keys differ for {section}: expected={sorted(expected)}, actual={sorted(actual)}")
        for key in expected:
            if len(expected[key]) != len(actual[key]) or not all(math.isclose(float(a), float(b), rel_tol=TOLERANCE, abs_tol=TOLERANCE) for a, b in zip(expected[key], actual[key])):
                raise ValueError(f"Frozen artifact value mismatch for {section}/{key}: expected={expected[key]}, actual={actual[key]}")
    return {"main": actual_main, "translation": actual_translation, "set_vs_attention": actual_set_vs_attention}


def render_confirmatory_figures(ctx: dict[str, Any], figures_dir: Path) -> list[dict[str, Any]]:
    points = ctx["points"]
    summary = ctx["summary"]
    conditions = confirmatory_conditions()
    labels = conditions["label"].tolist()
    records = []
    main_attention = [point(points, row.attention_id, "auprc") for row in conditions.itertuples()]
    main_set = [point(points, row.set_id, "auprc") for row in conditions.itertuples()]
    main_stem = "figure_main_auprc_comparison"
    export_figure(grouped_metric_figure(labels, main_attention, main_set, "AUPRC", main_stem), {"figures": figures_dir}, main_stem)
    records.append({"figure_id": main_stem, "conditions": labels, "madlad_present": False})

    set_rows = [contrast_row(summary, AGGREGATOR_CONTRASTS[row.condition_id], "auprc", family="aggregator") for row in conditions.itertuples()]
    set_stem = "figure_set_vs_attention_auprc_forest"
    export_figure(forest_figure(labels, [float(row["observed_delta"]) for row in set_rows], [float(row["ci_low"]) for row in set_rows], [float(row["ci_high"]) for row in set_rows], "Delta Set - Attention (AUPRC)", set_stem), {"figures": figures_dir}, set_stem)
    records.append({"figure_id": set_stem, "conditions": labels, "madlad_present": False})

    translation_rows = [contrast_row(summary, contrast_id, "auprc", family=family) for _, _, contrast_id, family in CONFIRMATORY_TRANSLATION_SPECS]
    translation_labels = [label for _, label, _, _ in CONFIRMATORY_TRANSLATION_SPECS]
    translation_groups = [group for group, _, _, _ in CONFIRMATORY_TRANSLATION_SPECS]
    translation_stem = "figure_translation_effects_auprc_forest"
    export_figure(forest_figure(translation_labels, [float(row["observed_delta"]) for row in translation_rows], [float(row["ci_low"]) for row in translation_rows], [float(row["ci_high"]) for row in translation_rows], "AUPRC difference (Set Transformer)", translation_stem, groups=translation_groups, height=1.6), {"figures": figures_dir}, translation_stem)
    records.append({"figure_id": translation_stem, "conditions": translation_labels, "madlad_present": False})
    return records


def update_inventory(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) if rows else []
    updates = {
        "figure_main_auprc_comparison": ("Three-condition confirmatory metric comparison", "AUPRC across the three NLLB confirmatory conditions"),
        "figure_set_vs_attention_auprc_forest": ("Set versus Attention paired effect for three confirmatory conditions", "AUPRC Set - Attention across the three NLLB confirmatory conditions"),
        "figure_translation_effects_auprc_forest": ("NLLB training-side and target-side translation effects", "AUPRC NLLB translation-side effects under the frozen paired protocol"),
    }
    seen = set()
    for row in rows:
        if row["figure_id"] in updates:
            row["question"], row["short_caption"] = updates[row["figure_id"]]
            seen.add(row["figure_id"])
    if seen != set(updates):
        raise ValueError(f"Figure inventory is missing target rows: {sorted(set(updates) - seen)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def regenerate_confirmatory_figures(root: Path | None = None, output_root: Path | None = None) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    output_root = (output_root or root / "results/paper").resolve()
    registry = build_source_registry(root, official_run=root / "runs/publichearing_final_paired_bootstrap" / OFFICIAL_RUN_ID, output_root=output_root)
    ctx = validate_sources(registry)
    ctx["root"] = root
    ctx["points"] = collect_points(ctx["summary"])
    values = validate_expected_values(ctx)
    figures_dir = output_root / "figures"
    inventory_path = output_root / "figure_inventory.csv"
    if not figures_dir.is_dir() or not inventory_path.is_file():
        raise FileNotFoundError("Existing paper figure directory and inventory are required")
    before_pngs = snapshot_pngs(figures_dir)
    before_inventory = inventory_path.read_text(encoding="utf-8")
    records = render_confirmatory_figures(ctx, figures_dir)
    update_inventory(inventory_path)
    after_pngs = snapshot_pngs(figures_dir)
    changed_pngs = {name for name in set(before_pngs) | set(after_pngs) if before_pngs.get(name) != after_pngs.get(name)}
    if changed_pngs != {f"{stem}.png" for stem in TARGET_FIGURES}:
        raise AssertionError(f"Unexpected PNG changes: {sorted(changed_pngs)}")
    if set(after_pngs) - set(before_pngs) != {f"{stem}.png" for stem in TARGET_FIGURES if f"{stem}.png" not in before_pngs}:
        raise AssertionError("Unexpected PNG creation outside target figures")
    if not inventory_path.read_text(encoding="utf-8"):
        raise AssertionError("Figure inventory became empty")
    return {"records": records, "values": values, "changed_pngs": sorted(changed_pngs), "inventory_changed": before_inventory != inventory_path.read_text(encoding="utf-8")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    result = regenerate_confirmatory_figures(root=args.root, output_root=args.output_root)
    print(f"Regenerated: {', '.join(result['changed_pngs'])}")
    print("Bootstrap, metrics, tables, claims, manifests, and other figures were not regenerated.")


if __name__ == "__main__":
    main()
