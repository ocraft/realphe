import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from realphe.pipe.metrics import confidence_interval, map_at_k, pr_roc_auc
from realphe.pipe.shared import SharedArray


def test_mean_average_precision_matches_sklearn():
    y_true = np.array([[0, 0, 1, 1]])
    y_pred = np.array([[0.1, 0.4, 0.35, 0.8]])

    assert np.isclose(
        map_at_k(y_true, y_pred, k=4),
        average_precision_score(y_true, y_pred, average='samples'),
    )


def test_pr_roc_auc_matches_sklearn():
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ], dtype=np.int8)
    y_pred = np.array([
        [0.7, 0.4],
        [0.8, 0.1],
        [0.3, 0.5],
    ], dtype=np.float32)

    y_true_shm = SharedArray(y_true)
    y_pred_shm = SharedArray(y_pred)
    try:
        pr_auc_val, roc_auc_val = pr_roc_auc(y_true_shm, y_pred_shm, n_jobs=1)
    finally:
        y_true_shm.unlink()
        y_pred_shm.unlink()

    assert np.allclose(pr_auc_val, average_precision_score(y_true, y_pred, average=None))
    assert np.allclose(roc_auc_val, roc_auc_score(y_true, y_pred, average=None))


def test_confidence_interval_zero_for_single_score():
    assert confidence_interval([0.5]) == 0.0


def test_confidence_interval_positive_for_multiple_scores():
    assert confidence_interval([0.4, 0.5, 0.6]) > 0.0
