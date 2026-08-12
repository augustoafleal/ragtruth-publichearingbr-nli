import numpy as np

from ragtruth_transfer.metrics import binary_metrics, select_threshold


def test_select_threshold_f1():
    y = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    threshold, metrics, feasible = select_threshold(y, scores, "f1")
    assert feasible
    assert metrics["F1"] == 1.0
    assert 0.2 < threshold <= 0.7


def test_select_threshold_fpr10():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.7, 0.9])
    threshold, metrics, feasible = select_threshold(y, scores, "fpr10")
    assert feasible
    assert metrics["FPR"] <= 0.10
    assert binary_metrics(y, scores >= threshold)["FPR"] <= 0.10
