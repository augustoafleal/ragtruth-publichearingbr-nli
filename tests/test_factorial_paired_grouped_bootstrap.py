import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.factorial_paired_grouped_bootstrap import (
    CONDITIONS,
    run_factorial_bootstrap,
    validate_factorial_frames,
)


def _frames():
    base = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "hearing_id": ["h1", "h1", "h2", "h2"],
            "label": [1, 0, 1, 0],
        }
    )
    frames = {}
    for condition_index, condition in enumerate(CONDITIONS):
        frames[condition] = {}
        for seed in (0, 1, 2):
            frame = base.copy()
            frame["probability"] = np.array([0.9, 0.2, 0.7, 0.1]) + condition_index * 0.01 + seed * 0.001
            frames[condition][seed] = frame
    return frames


def test_validate_factorial_frames_checks_exact_population_and_alignment():
    audit = validate_factorial_frames(_frames(), expected_rows=4, expected_positives=2, expected_hearings=2)

    assert audit["exact_example_alignment"] is True
    assert audit["exact_hearing_alignment"] is True
    assert audit["exact_label_alignment"] is True
    assert audit["prevalence"] == 0.5

    broken = _frames()
    broken["pt_set"][2] = broken["pt_set"][2].assign(example_id=["a", "b", "c", "other"])
    with pytest.raises(ValueError, match="example_id mismatch"):
        validate_factorial_frames(broken, expected_rows=4, expected_positives=2, expected_hearings=2)


def test_factorial_bootstrap_is_deterministic_and_uses_requested_metric():
    kwargs = {
        "expected_rows": 4,
        "expected_positives": 2,
        "expected_hearings": 2,
        "n_replicates": 7,
        "seed": 19,
        "metrics": ("auprc",),
    }
    first = run_factorial_bootstrap(_frames(), **kwargs)
    second = run_factorial_bootstrap(_frames(), **kwargs)

    pd.testing.assert_frame_equal(first["replicates"], second["replicates"])
    assert first["protocol"]["metrics"] == ["auprc"]
    assert first["protocol"]["n_replicates"] == 7
    assert set(first["replicates"]["metric"]) == {"auprc"}
