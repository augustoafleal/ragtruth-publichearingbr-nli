import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import ragtruth_transfer.support_verification as support_verification
from ragtruth_transfer.support_verification import (
    AUDIT_FIELDS,
    EntailmentVerifier,
    _select_negative_rows,
    run_audit,
    run_materialize,
    score_row,
)


class MockTokenizer:
    name_or_path = "mock-tokenizer"
    init_kwargs = {"mock": True}
    special_tokens_map = {}

    def __call__(self, premises, claims=None, truncation=False, max_length=None, padding=False, return_tensors=None, add_special_tokens=True):
        if isinstance(premises, str):
            premises = [premises]
        if isinstance(claims, str):
            claims = [claims] * len(premises)
        claims = claims or [""] * len(premises)
        encoded = []
        for premise, claim in zip(premises, claims):
            values = [1] * (len(str(premise).split()) + len(str(claim).split()) + 3)
            if truncation == "only_first" and max_length is not None:
                values = values[:max_length]
            encoded.append(values)
        if return_tensors == "pt":
            width = max(len(values) for values in encoded)
            return {"input_ids": torch.tensor([values + [0] * (width - len(values)) for values in encoded])}
        return {"input_ids": encoded[0] if len(encoded) == 1 else encoded}


class MockVerifier:
    model_id = "mock-model"
    revision = "mock-rev"
    tokenizer_id = "mock-tokenizer"
    tokenizer_revision = "mock-rev"
    tokenizer = MockTokenizer()

    def __init__(self):
        self.calls = []

    def score_pairs(self, premises, claims, batch_size):
        self.calls.append((list(premises), list(claims), batch_size))
        return [min(0.99, 0.1 + 0.1 * len(str(premise).split())) for premise in premises]


def _row(example_id: str, split: str, label: bool, source_id: str | None = None) -> dict:
    return {
        "example_id": example_id,
        "source_id": source_id or example_id,
        "response_id": example_id.split(":")[0],
        "claim": "claim text",
        "label": label,
        "evidence": ["one two", "three four", "", ""],
        "evidence_mask": [True, True, False, False],
        "task_type": "QA",
        "_broad_split": split,
    }


def _write_broad(root: Path, rows_by_split: dict[str, list[dict]]) -> None:
    for split, rows in rows_by_split.items():
        path = root / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps({k: v for k, v in row.items() if k != "_broad_split"}) for row in rows) + "\n", encoding="utf-8")


def _score_record(row: dict, individual: float, concat: float, truncated: bool = False) -> dict:
    return {
        "split": row["_broad_split"],
        "example_id": row["example_id"],
        "source_id": row["source_id"],
        "label": row["label"],
        "claim_sha256": "x",
        "evidence_sha256": "y",
        "evidence_mask": row["evidence_mask"],
        "valid_evidence_count": 2,
        "individual_scores": [individual, individual - 0.1, None, None],
        "individual_max_score": individual,
        "best_evidence_index": 0,
        "concat_score": concat,
        "concat_tokens_before": 5,
        "concat_tokens_after": 3 if truncated else 5,
        "concat_truncated": truncated,
        "verifier_model_id": "mock-model",
        "verifier_revision": "mock-rev",
        "tokenizer_fingerprint": "mock-tokenizer-fp",
        "max_length": 512,
        "execution_fingerprint": {"fingerprint_sha256": "mock"},
    }


def test_score_row_preserves_mask_and_records_both_modes():
    row = _row("1:0", "train", False)
    verifier = MockVerifier()
    record = score_row(row, verifier, 2, 3, "\n[SEP]\n", {"tokenizer_fingerprint": "mock-fp"})
    assert record["example_id"] == "1:0"
    assert record["evidence_mask"] == [True, True, False, False]
    assert record["valid_evidence_count"] == 2
    assert record["individual_scores"][2] is None
    assert record["individual_max_score"] is not None
    assert record["concat_score"] is not None
    assert record["concat_truncated"] is True
    assert verifier.calls[0][0] == ["one two", "three four"]
    assert verifier.calls[0][1] == ["claim text", "claim text"]


def test_smoke_limit_keeps_negative_order_and_limits_each_split():
    rows = [
        _row("train:0", "train", False), _row("train:1", "train", True), _row("train:2", "train", False),
        _row("validation:0", "validation", False), _row("validation:1", "validation", False),
    ]
    selected = _select_negative_rows(rows, 1)
    assert [row["example_id"] for row in selected] == ["train:0", "validation:0"]


def test_entailment_mapping_must_be_unambiguous():
    verifier = EntailmentVerifier.__new__(EntailmentVerifier)

    class Config:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    class Model:
        config = Config()

    verifier.model = Model()
    assert verifier._find_entailment_index() == 1
    verifier.model.config.id2label = {0: "entailment", 1: "ENTAILMENT"}
    with pytest.raises(ValueError, match="ambíguo"):
        verifier._find_entailment_index()


def test_audit_sample_has_required_fields_and_excludes_test_by_default(tmp_path):
    broad = tmp_path / "broad"
    broad.mkdir()
    rows = {"train": [_row("1:0", "train", False)], "validation": [_row("2:0", "validation", True)], "test": [_row("3:0", "test", False)]}
    _write_broad(broad, rows)
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(json.dumps(_score_record(row, 0.5, 0.4)) for values in rows.values() for row in values if not row["label"]) + "\n", encoding="utf-8")
    output = tmp_path / "audit.csv"
    result = run_audit(Namespace(broad_dir=broad, scores_path=scores, output_path=output, sample_size=10, seed=42, include_test=False))
    assert result["sample_size"] == 1
    with output.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert set(records[0]) == set(AUDIT_FIELDS)
    assert {record["split"] for record in records} == {"train"}
    assert all(record["manual_decision"] == "" for record in records)


def test_materialize_preserves_schema_order_and_labels(tmp_path):
    broad = tmp_path / "broad"
    broad.mkdir()
    rows = {"train": [_row("1:0", "train", True), _row("1:1", "train", False)], "validation": [_row("2:0", "validation", False)], "test": [_row("3:0", "test", False)]}
    _write_broad(broad, rows)
    scores = tmp_path / "scores.jsonl"
    score_rows = [
        _score_record(rows["train"][1], 0.8, 0.0),
        _score_record(rows["validation"][0], 0.1, 0.9),
        _score_record(rows["test"][0], 0.1, 0.1),
    ]
    scores.write_text("\n".join(json.dumps(record) for record in score_rows) + "\n", encoding="utf-8")
    output = tmp_path / "strict"
    run_materialize(Namespace(broad_dir=broad, scores_path=scores, output_dir=output, individual_threshold=0.7, concat_threshold=0.7, allow_nontruncated_concat=True, overwrite=False))
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in train] == ["1:0", "1:1"]
    assert [row["example_id"] for row in validation] == ["2:0"]
    assert all("_broad_split" not in row for row in train + validation)
    assert all(row["label"] is False for row in validation)
    assert set(train[0]) == set(json.loads((broad / "train.jsonl").read_text().splitlines()[0]))


def test_resume_rejects_incompatible_fingerprint(tmp_path, monkeypatch):
    broad = tmp_path / "broad"
    broad.mkdir()
    rows = {split: [_row(f"{index}:0", split, False)] for index, split in enumerate(("train", "validation", "test"))}
    _write_broad(broad, rows)
    output = tmp_path / "work"
    output.mkdir()
    partial = _score_record(rows["train"][0], 0.5, 0.5)
    partial["execution_fingerprint"] = {"fingerprint_sha256": "incompatible"}
    (output / "support_verification.jsonl.partial").write_text(json.dumps(partial) + "\n", encoding="utf-8")
    monkeypatch.setattr(support_verification.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: MockTokenizer())
    args = Namespace(
        broad_dir=broad, output_dir=output, verifier_model="mock-model", verifier_revision="mock-rev",
        tokenizer_id=None, tokenizer_revision=None, batch_size=2, max_length=512,
        separator="\n\n[SEP]\n\n", device="cpu", resume=True, overwrite=False,
    )
    with pytest.raises(ValueError, match="Fingerprint incompatível"):
        support_verification.run_score(args)
