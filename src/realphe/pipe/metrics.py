from collections import Counter
from functools import wraps
import gc
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, cast

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import scipy
from sklearn.metrics import accuracy_score, auc, average_precision_score, roc_auc_score

from .shared import SharedArray


Metric = Callable[[np.ndarray, np.ndarray], float]
SampleIdx = Optional[NDArray[np.intp]]


def div_zero(x1: np.ndarray, x2: np.ndarray) -> NDArray[np.float32]:
    return np.divide(x1, x2, out=np.zeros_like(x1, dtype=np.float32), where=x2 != 0)


def threshold(func, value: float = 0.5):
    @wraps(func)
    def wrapper(y_true, y_pred):
        return func(y_true, y_pred > value)
    return wrapper


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred))


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_pred))


def map_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Optional[int] = None,
    break_ties: bool = False,
    return_mean: bool = True
) -> float:
    # Calculate the precision within the top predictions (from 1 to k), where precision
    # is the proportion of correct hits in the set. Then, compute the weighted average
    # based on the increase in recall, which reflects how much recall improves with each
    # additional prediction. Recall represents the proportion of relevant items that have
    # actually been identified out of the total relevant items.

    if k is None:
        k = y_true.shape[1]

    # breaking ties randomly
    if break_ties:
        idx = np.argsort(
            y_pred + (np.random.default_rng().random(
                size=y_pred.shape,
                dtype=np.float32
            ) * 10**(-10)),
            axis=1
        ).astype(np.int32)[:, ::-1]
    else:
        n_batch = (y_pred.shape[0] // 20000) + 1
        idx_list = []
        for y_pred_batch in np.array_split(y_pred, n_batch):
            idx_list.append(np.argsort(y_pred_batch, axis=1).astype(np.int32)[:, ::-1])
        idx = np.concatenate(idx_list, dtype=np.int32)
        del idx_list

    # get the top k predictions for each row, where each position in rel_k array means a hit or miss
    # on this rank, e.g [1,0,1,0] means that 1st and 3th highest predictions were correct
    rel_k = np.zeros(shape=(y_true.shape[0], k), dtype=np.int8)
    for i in range(y_true.shape[0]):
        rel_k[i] = y_true[i][idx[i][:k]]

    # calculate precision at each position up to k for each row, i.e. how many hit we have for k
    precision_at_k = (
        np.cumsum(rel_k, axis=1, dtype=np.float32) /
        np.tile(np.arange(1, k+1, dtype=np.int16), (y_true.shape[0], 1))
    )

    # Calculate the average P@k from 1 to k, weighted by the increase in recall
    # (which is equal to rel(k) / sum(y_true), where rel(k) = 0 means no increase
    # and rel(k) = 1 means an increase, indicating that the next label was correctly predicted).

    # Use np.minimum(k, np.sum(y_true, axis=1))) for calculating recall only for k relevant
    # prediction, not the wholeset (as in tensorflow_ranking).
    mapk = (
        np.sum(precision_at_k * rel_k, axis=1) /
        np.maximum(1, np.sum(y_true, axis=1, dtype=np.int16))
    )
    if return_mean:
        result = float(np.mean(mapk))
        del mapk
    else:
        result = mapk
    del idx, rel_k, precision_at_k
    return result


def pr_roc_auc(
    y_true: SharedArray,
    y_pred: SharedArray,
    n_jobs: int = -1
):

    def _compute_label_pr_roc_auc(i: int):
        # Sort scores and corresponding true labels

        y_true_sorted = y_true.read()[:, i][
            np.argsort(-y_pred.read()[:, i])
        ]

        # Compute cumulative sums for TP, FP
        tps = np.cumsum(y_true_sorted)  # True Positives
        fps = np.cumsum(1.0 - y_true_sorted)  # False Positives
        total_positives = tps[-1]

        # Precision-Recall components
        precision = np.hstack((1, div_zero(tps, tps + fps)))
        recall = np.hstack((0, div_zero(tps, total_positives)))
        del tps

        # ROC components
        fpr = div_zero(fps, len(y_true_sorted) - total_positives)
        del fps, total_positives
        tpr = recall  # Recall is equivalent to TPR for ROC

        # Calculate AUCs
        sl = slice(None, None, -1)  # reverse the outputs so recall is decreasing
        # This implementation is not interpolated and is different
        # from computing the area under the precision-recall curve with the
        # trapezoidal rule, which uses linear interpolation and can be too
        # optimistic.

        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve.
        # Due to numerical error, we can get `-0.0` and we therefore clip it.
        pr_auc_val = max(0.0, -np.sum(np.diff(recall[sl]) * np.array(precision[sl])[:-1],
                                      dtype=np.float32))

        roc_auc_val = auc(np.hstack((0, fpr)), tpr)
        del fpr, tpr, recall, precision

        return pr_auc_val, roc_auc_val

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_label_pr_roc_auc)(i)
        for i in range(y_true.shape[-1])
    )

    pr_aucs, roc_aucs = zip(*results)
    return pr_aucs, roc_aucs


def confidence_interval(scores: List[float], level: float = 0.95) -> float:
    rounds = len(scores)
    if rounds == 1:
        return 0.0
    t_value = scipy.stats.t.ppf((1 + level) / 2.0, df=rounds - 1)

    sd = np.std(scores, ddof=1)
    se = sd / np.sqrt(rounds)

    ci_length = t_value * se

    return ci_length


def bootstrap(
    y_true: SharedArray,
    y_pred: SharedArray,
    statistic: Callable[[np.ndarray, np.ndarray], float | np.ndarray],
    level: float = 0.95,
    n_jobs: int = -1,
    sample_size: float = 1.0,
    n_resamples: int = 200
) -> Tuple[np.float32, np.float32, np.float32]:

    idx = np.arange(y_true.shape[0])

    def _stat(i):
        rng = np.random.RandomState(seed=i)
        pred_idx = rng.choice(idx, size=int(idx.shape[0] * sample_size), replace=True)
        return statistic(y_true.read()[pred_idx], y_pred.read()[pred_idx])

    results = cast(List, Parallel(n_jobs=n_jobs)(
        delayed(_stat)(i)
        for i in range(n_resamples)
    ))

    level_range = ((1.0 - level) / 2.0) * 100
    return (np.percentile(results, level_range, axis=0),
            np.mean(results, axis=0),
            np.percentile(results, 100 - level_range, axis=0))
