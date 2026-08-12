import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.publichearing.config import PublicHearingConfig
from ragtruth_transfer.publichearing.splits import assert_no_group_leakage, make_splits


def _frame():
    rows = []
    for hearing in range(20):
        for opinion in range(2):
            rows.append({"sample_id": f"{hearing}:{opinion}", "hearing_id": str(hearing), "label": int((hearing + opinion) % 2)})
    return pd.DataFrame(rows)


def test_grouped_splits_cover_each_row_once_without_hearing_leakage():
    frame = _frame()
    config = PublicHearingConfig(mode="smoke", outer_splits=2, inner_splits=2, max_epochs=1, seeds=(101,))
    folds = make_splits(frame, config)
    assert sorted(np.concatenate([fold.test for fold in folds]).tolist()) == list(range(len(frame)))
    for fold in folds:
        assert_no_group_leakage(frame.hearing_id.to_numpy(), fold.train, fold.validation, fold.test)


def test_group_leakage_is_rejected():
    with pytest.raises(RuntimeError, match="Leakage"):
        assert_no_group_leakage(np.array(["a", "a", "b"]), np.array([0]), np.array([1]), np.array([2]))
