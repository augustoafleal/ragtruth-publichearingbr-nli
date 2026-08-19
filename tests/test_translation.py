from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragtruth_transfer.translation import (
    TranslationConfig,
    TranslationSettings,
    translate_ragtruth,
)


class FakeTranslator:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) >= self.fail_on_call:
            raise RuntimeError("falha simulada")
        return [f"PT:{text}" for text in texts]


def _config(
    tmp_path: Path,
    batch_size: int = 2,
    splits: tuple[str, ...] = ("train", "test"),
    sample_fraction: float | None = None,
    sample_seed: int = 42,
    max_examples_per_split: int | None = None,
) -> TranslationConfig:
    return TranslationConfig(
        translation=TranslationSettings(
            translator="nllb",
            model_name="fake-model",
            source_language="eng_Latn",
            target_language="por_Latn",
            batch_size=batch_size,
            device="cpu",
        ),
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        splits=splits,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        max_examples_per_split=max_examples_per_split,
    )


def _write_input(config: TranslationConfig) -> dict:
    config.input_dir.mkdir()
    rows = {
        "train": [
            {
                "example_id": "e2",
                "source_id": "s2",
                "label": True,
                "claim": "same claim",
                "evidence": ["valid evidence", "masked text", "another evidence"],
                "evidence_mask": [True, False, True],
                "metadata": {"keep": [1, "x"]},
            },
            {
                "example_id": "e1",
                "source_id": "s1",
                "label": False,
                "claim": "second claim",
                "evidence": ["same claim", "untouched"],
                "evidence_mask": [True, False],
                "metadata": {"keep": [2, "y"]},
            },
        ],
        "test": [
            {
                "example_id": "e0",
                "source_id": "s0",
                "label": False,
                "claim": "same claim",
                "evidence": ["valid evidence", "untouched"],
                "evidence_mask": [True, False],
                "metadata": {"keep": [3, "z"]},
            }
        ],
    }
    for split, split_rows in rows.items():
        (config.input_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows), encoding="utf-8"
        )
    return rows


def _write_many_input(config: TranslationConfig, rows_per_split: int = 8) -> None:
    config.input_dir.mkdir()
    for split in config.splits:
        rows = []
        for index in range(rows_per_split):
            rows.append(
                {
                    "example_id": f"{split}-{index}",
                    "source_id": f"source-{split}-{index}",
                    "label": bool(index % 2),
                    "claim": f"claim {split} {index}",
                    "evidence": [f"evidence {split} {index}", "masked slot"],
                    "evidence_mask": [True, False],
                    "metadata": {"index": index},
                }
            )
        (config.input_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_yaml_selects_nllb_and_madlad_defaults() -> None:
    nllb = TranslationConfig.from_yaml(Path("configs/ragtruth_translate_nllb.yaml"))
    madlad = TranslationConfig.from_yaml(Path("configs/ragtruth_translate_madlad.yaml"))
    nllb_smoke = TranslationConfig.from_yaml(Path("configs/ragtruth_translate_nllb_smoke.yaml"))
    madlad_smoke = TranslationConfig.from_yaml(Path("configs/ragtruth_translate_madlad_smoke.yaml"))
    assert nllb.translation.translator == "nllb"
    assert nllb.translation.target_language == "por_Latn"
    assert nllb.sample_fraction == 1.0 and nllb.sample_seed == 42
    assert madlad.translation.translator == "madlad"
    assert madlad.translation.target_language == "pt"
    assert madlad.sample_fraction == 1.0 and madlad.sample_seed == 42
    assert nllb_smoke.max_examples_per_split == 3
    assert nllb_smoke.output_dir.name.endswith("_smoke")
    assert madlad_smoke.max_examples_per_split == 3


def test_unknown_translator_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Tradutor desconhecido"):
        TranslationSettings("unknown", "model", "en", "pt")


@pytest.mark.parametrize("fraction", [0.01, 0.25, 1.0])
def test_sample_fraction_validates(fraction: float, tmp_path: Path) -> None:
    config = _config(tmp_path, sample_fraction=fraction)
    assert config.effective_sample_fraction == fraction


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, 2.0])
def test_sample_fraction_out_of_range_fails(fraction: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_fraction"):
        _config(tmp_path, sample_fraction=fraction)


def test_sampling_and_smoke_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        _config(tmp_path, sample_fraction=0.25, max_examples_per_split=3)


def test_translation_preserves_structure_masks_order_and_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = _write_input(config)
    fake = FakeTranslator()
    manifest = translate_ragtruth(config, translator=fake)

    assert sum(len(call) for call in fake.calls) == 4
    assert sum(len(call) for call in fake.calls) == len({text for rows in original.values() for row in rows for text in [row["claim"], *[value for value, valid in zip(row["evidence"], row["evidence_mask"]) if valid]]})
    assert manifest["backend"] == "nllb"
    assert manifest["mode"] == "full"
    assert manifest["is_full_dataset"] is True
    assert manifest["sample_fraction"] == 1.0
    assert manifest["counts"]["examples"] == 3
    assert manifest["counts"]["unique_texts"] == 4
    assert manifest["splits"] == ["train", "test"]

    output_train = [json.loads(line) for line in (config.output_dir / "train.jsonl").read_text().splitlines()]
    output_test = [json.loads(line) for line in (config.output_dir / "test.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in output_train + output_test] == ["e2", "e1", "e0"]
    assert output_train[0]["claim"] == "PT:same claim"
    assert output_train[0]["evidence"] == ["PT:valid evidence", "masked text", "PT:another evidence"]
    assert output_train[0]["evidence_mask"] == [True, False, True]
    assert output_train[0]["metadata"] == original["train"][0]["metadata"]
    assert output_train[1]["evidence"][0] == "PT:same claim"
    assert output_test[0]["evidence"][1] == "untouched"
    assert json.loads((config.output_dir / "manifest.json").read_text())["status"] == "completed"


def test_fraction_sampling_is_per_split_deterministic_and_ordered(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        batch_size=4,
        splits=("train", "validation", "test"),
        sample_fraction=0.25,
        sample_seed=42,
    )
    _write_many_input(config)
    translate_ragtruth(config, translator=FakeTranslator())

    def output_ids(output_dir: Path) -> dict[str, list[str]]:
        return {
            split: [json.loads(line)["example_id"] for line in (output_dir / f"{split}.jsonl").read_text().splitlines()]
            for split in config.splits
        }

    first_ids = output_ids(config.output_dir)
    assert {split: len(ids) for split, ids in first_ids.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    for ids in first_ids.values():
        assert [int(value.rsplit("-", 1)[1]) for value in ids] == sorted(
            int(value.rsplit("-", 1)[1]) for value in ids
        )
    assert json.loads((config.output_dir / "manifest.json").read_text())["mode"] == "sample"

    same_config = _config(
        tmp_path / "same",
        batch_size=4,
        splits=config.splits,
        sample_fraction=0.25,
        sample_seed=42,
    )
    same_config.input_dir.mkdir(parents=True)
    for path in config.input_dir.glob("*.jsonl"):
        (same_config.input_dir / path.name).write_bytes(path.read_bytes())
    translate_ragtruth(same_config, translator=FakeTranslator())
    assert output_ids(same_config.output_dir) == first_ids

    changed_config = _config(
        tmp_path / "changed",
        batch_size=4,
        splits=config.splits,
        sample_fraction=0.25,
        sample_seed=7,
    )
    changed_config.input_dir.mkdir(parents=True)
    for path in config.input_dir.glob("*.jsonl"):
        (changed_config.input_dir / path.name).write_bytes(path.read_bytes())
    translate_ragtruth(changed_config, translator=FakeTranslator())
    assert output_ids(changed_config.output_dir) != first_ids


def test_smoke_limits_each_split_and_manifest_marks_it(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        splits=("train", "validation", "test"),
        max_examples_per_split=3,
    )
    _write_many_input(config)
    manifest = translate_ragtruth(config, translator=FakeTranslator())
    assert manifest["mode"] == "smoke"
    assert manifest["is_full_dataset"] is False
    assert manifest["max_examples_per_split"] == 3
    assert manifest["counts"]["examples_by_split"] == {"train": 3, "validation": 3, "test": 3}


def test_cache_is_reused_and_incompatible_cache_fails(tmp_path: Path) -> None:
    config = _config(tmp_path, batch_size=2)
    _write_input(config)
    first = FakeTranslator()
    translate_ragtruth(config, translator=first)
    second = FakeTranslator()
    translate_ragtruth(config, translator=second)
    assert second.calls == []
    resumed = FakeTranslator()
    translate_ragtruth(config, translator=resumed, resume=True)
    assert resumed.calls == []

    changed = TranslationConfig(
        translation=TranslationSettings("nllb", "different-model", "eng_Latn", "por_Latn", device="cpu"),
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        splits=config.splits,
    )
    (config.output_dir / "manifest.json").unlink()
    with pytest.raises(ValueError, match="Cache de tradução incompatível"):
        translate_ragtruth(changed, translator=FakeTranslator())


def test_failed_batch_does_not_write_output_or_cache_original_text(tmp_path: Path) -> None:
    config = _config(tmp_path, batch_size=2)
    _write_input(config)
    with pytest.raises(RuntimeError, match="falha simulada"):
        translate_ragtruth(config, translator=FakeTranslator(fail_on_call=2))
    assert not (config.output_dir / "manifest.json").exists()
    assert not (config.output_dir / "train.jsonl").exists()

    resumed = FakeTranslator()
    manifest = translate_ragtruth(config, translator=resumed)
    assert manifest["status"] == "completed"
    assert len(resumed.calls) == 1
    output = (config.output_dir / "train.jsonl").read_text()
    assert '"claim": "same claim"' not in output
    assert "PT:same claim" in output
