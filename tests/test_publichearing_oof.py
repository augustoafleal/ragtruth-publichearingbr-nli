import pandas as pd
import pytest

from ragtruth_transfer.publichearing.campaign import validate_oof_coverage


def test_oof_coverage_requires_one_prediction_per_modelable_sample():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"]})
    valid = pd.DataFrame({"sample_id": ["a", "b", "c"]})
    validate_oof_coverage(valid, frame)
    with pytest.raises(RuntimeError, match="Cobertura OOF"):
        validate_oof_coverage(pd.DataFrame({"sample_id": ["a", "a", "c"]}), frame)
