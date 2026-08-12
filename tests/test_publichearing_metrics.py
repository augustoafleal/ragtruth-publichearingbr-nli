import numpy as np
import pandas as pd

from ragtruth_transfer.publichearing.metrics import grouped_bootstrap, select_operating_threshold


def test_operating_thresholds_use_validation_rules():
    labels = np.array([0, 0, 0, 1, 1], dtype=bool)
    probabilities = np.array([.1, .2, .3, .7, .9])
    threshold, metrics = select_operating_threshold(labels, probabilities, "max_f1", .10)
    assert metrics["F1"] == 1.0 and .2 < threshold <= .7
    threshold, metrics = select_operating_threshold(labels, probabilities, "fpr_operational", .10)
    assert metrics["FPR"] <= .10 and metrics["Recall"] == 1.0


def test_cluster_bootstrap_preserves_group_sampling_and_metrics():
    frame = pd.DataFrame({"hearing_id": ["a", "a", "b", "b"], "label": [0, 1, 0, 1], "probability": [.1, .9, .2, .8], "prediction_max_f1": [0, 1, 0, 1], "prediction_fpr_operational": [0, 1, 0, 1]})
    samples, summary = grouped_bootstrap(frame, repetitions=8, seed=4)
    assert samples.repetition.nunique() <= 8
    assert {"AUPRC", "AUROC", "Brier"} <= set(summary.metric)
