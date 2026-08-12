import json

import pandas as pd
import pytest

from ragtruth_transfer.publichearing.training import _artifact_hashes, _complete_seed_dir


def test_partial_seed_directory_is_not_accepted(tmp_path):
    run = tmp_path / "seed"; run.mkdir()
    (run / "SEED_RUN_COMPLETE.json").write_text(json.dumps({"seed_signature": "same"}), encoding="utf-8")
    assert not _complete_seed_dir(run, "same")


def test_complete_seed_requires_matching_signature(tmp_path):
    run = tmp_path / "seed"; run.mkdir()
    for name in ("validation_predictions.csv", "test_predictions.csv", "training_history.csv", "metadata.json"):
        (run / name).write_text("x", encoding="utf-8")
    (run / "head.pt").write_bytes(b"x")
    (run / "SEED_RUN_COMPLETE.json").write_text(json.dumps({"seed_signature": "old"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatível"):
        _complete_seed_dir(run, "new")


def test_complete_seed_rejects_hash_mismatch(tmp_path):
    run = tmp_path / "seed"; run.mkdir()
    for name in ("validation_predictions.csv", "test_predictions.csv"):
        (run / name).write_text("row_index,label,probability\n0,0,0.1\n", encoding="utf-8")
    (run / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    (run / "metadata.json").write_text("{}", encoding="utf-8")
    import torch
    torch.save({"head.weight": torch.zeros(1)}, run / "head.pt")
    (run / "SEED_RUN_COMPLETE.json").write_text(json.dumps({"seed_signature": "sig", "files": _artifact_hashes(run)}), encoding="utf-8")
    (run / "validation_predictions.csv").write_text("corrupted", encoding="utf-8")
    assert not _complete_seed_dir(run, "sig")
