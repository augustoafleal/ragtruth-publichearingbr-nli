import json
from argparse import Namespace

import pytest

from ragtruth_transfer.support_verification import run_materialize


def _row(example_id, source_id, label):
    return {
        "example_id": example_id,
        "source_id": source_id,
        "claim": "claim",
        "label": label,
        "evidence": ["evidence", "", "", ""],
        "evidence_mask": [True, False, False, False],
        "task_type": "Summary",
    }


def test_strict_is_subset_and_preserves_positive(tmp_path):
    broad = tmp_path / "broad"
    broad.mkdir()
    rows = {
        "train": [_row("a:0", "a", True), _row("a:1", "a", False)],
        "validation": [_row("b:0", "b", False)],
        "test": [_row("c:0", "c", False)],
    }
    for split, split_rows in rows.items():
        (broad / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in split_rows) + "\n", encoding="utf-8")
    scores = tmp_path / "scores.jsonl"
    records = []
    for row, individual, concat in [
        (rows["train"][1], 0.1, 0.1),
        (rows["validation"][0], 0.1, 0.9),
        (rows["test"][0], 0.9, 0.1),
    ]:
        records.append({
            "example_id": row["example_id"], "source_id": row["source_id"], "split": next(split for split, values in rows.items() if row in values),
            "individual_max_score": individual, "concat_score": concat, "concat_truncated": False,
            "valid_evidence_count": 1, "verifier_model_id": "mock", "verifier_revision": "rev",
            "tokenizer_fingerprint": "fp", "max_length": 512,
        })
    scores.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output = tmp_path / "strict"
    run_materialize(Namespace(broad_dir=broad, scores_path=scores, output_dir=output, individual_threshold=0.8, concat_threshold=0.8, allow_nontruncated_concat=False, overwrite=False))
    original_ids = {row["example_id"] for values in rows.values() for row in values}
    strict_rows = [json.loads(line) for split in ("train", "validation", "test") for line in (output / f"{split}.jsonl").read_text().splitlines()]
    assert {row["example_id"] for row in strict_rows} <= original_ids
    assert [row["example_id"] for row in strict_rows if row["label"]] == ["a:0"]
    assert all(row["example_id"] != "b:0" for row in strict_rows)
    assert all(row["label"] in (True, False) for row in strict_rows)


def test_materialize_rejects_source_overlap_across_splits(tmp_path):
    broad = tmp_path / "broad"
    broad.mkdir()
    for split in ("train", "validation", "test"):
        row = _row(f"{split}:0", "shared", False)
        (broad / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    scores = tmp_path / "scores.jsonl"
    records = []
    for split in ("train", "validation", "test"):
        records.append({"example_id": f"{split}:0", "source_id": "shared", "individual_max_score": 0.1, "concat_score": 0.1, "concat_truncated": False})
    scores.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_id"):
        run_materialize(Namespace(broad_dir=broad, scores_path=scores, output_dir=tmp_path / "strict", individual_threshold=0.5, concat_threshold=0.5, allow_nontruncated_concat=False, overwrite=False))
