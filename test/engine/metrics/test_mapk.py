import numpy as np
import pytest
from sklearn.metrics import average_precision_score
import torch

from realphe.engine.metrics import MeanAveragePrecisionAtK
from realphe.pipe.metrics import map_at_k


def test_perfect_prediction():
    # given
    metric = MeanAveragePrecisionAtK(k=3)

    y_pred = torch.tensor([
        [0.9, 0.8, 0.1],
        [0.1, 0.9, 0.8],
    ])

    y_true = torch.tensor([
        [1, 1, 0],
        [0, 1, 1],
    ])

    # when
    metric.update(y_pred, y_true)
    result = metric.compute()

    # then
    assert torch.isclose(result, torch.tensor(1.0))


def test_worst_prediction():
    # given
    metric = MeanAveragePrecisionAtK(k=2)

    y_pred = torch.tensor([
        [0.1, 0.8, 0.9],
    ])

    y_true = torch.tensor([
        [1, 0, 0],
    ])

    # when
    metric.update(y_pred, y_true)
    result = metric.compute()

    # then
    assert torch.isclose(result, torch.tensor(0.0))


def test_partial_prediction():
    # given
    metric = MeanAveragePrecisionAtK(k=3)

    y_pred = torch.tensor([
        [0.9, 0.8, 0.7],
    ])

    y_true = torch.tensor([
        [1, 0, 1],
    ])

    # when
    metric.update(y_pred, y_true)
    result = metric.compute()

    # then

    # AP:
    # rank1 -> relevant => 1/1
    # rank2 -> irrelevant
    # rank3 -> relevant => 2/3
    #
    # AP = (1 + 2/3) / 2 = 5/6

    expected = torch.tensor(5.0 / 6.0)

    assert torch.isclose(result, expected, atol=1e-6)


def test_ignores_samples_without_positive_labels():
    # given
    metric = MeanAveragePrecisionAtK(k=3)

    y_pred = torch.tensor([
        [0.9, 0.8, 0.1],
        [0.1, 0.2, 0.3],
    ])

    y_true = torch.tensor([
        [1, 1, 0],
        [0, 0, 0],
    ])

    # when
    metric.update(y_pred, y_true)
    result = metric.compute()

    # then
    assert torch.isclose(result, torch.tensor(1.0))


@pytest.mark.parametrize("batch_size,num_labels", [
    (1, 5),
    (4, 10),
    (8, 32),
])
def test_matches_sklearn_when_k_equals_num_labels(
    batch_size: int,
    num_labels: int
):
    # given
    torch.manual_seed(42)

    y_pred = torch.rand(batch_size, num_labels)
    y_true = torch.randint(0, 2, (batch_size, num_labels))

    # - ensure at least one positive per sample
    empty_rows = y_true.sum(dim=1) == 0

    for idx in torch.where(empty_rows)[0]:
        y_true[idx, 0] = 1

    metric = MeanAveragePrecisionAtK(k=num_labels)

    # when
    metric.update(y_pred, y_true)
    torch_result = metric.compute().item()
    sklearn_result = average_precision_score(y_true, y_pred, average='samples')

    # then
    assert abs(torch_result - sklearn_result) < 1e-6


def test_multiple_updates_accumulate_correctly():
    # given
    metric = MeanAveragePrecisionAtK(k=4)

    y_pred_1 = torch.tensor([
        [0.9, 0.8, 0.1, 0.2],
    ])

    y_true_1 = torch.tensor([
        [1, 1, 0, 0],
    ])

    y_pred_2 = torch.tensor([
        [0.1, 0.2, 0.9, 0.8],
    ])

    y_true_2 = torch.tensor([
        [1, 0, 1, 0],
    ])

    # when
    metric.update(y_pred_1, y_true_1)
    metric.update(y_pred_2, y_true_2)
    result = metric.compute()

    # then
    expected_1 = 1.0

    # ranking: [2,3,1,0]
    # relevance: [1,0,0,1]
    # AP = (1/1 + 2/4) / 2 = 0.75
    expected_2 = 0.75
    expected = torch.tensor((expected_1 + expected_2) / 2)

    assert torch.isclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("batch_size,num_labels,k", [
    (1, 5, 5),
    (4, 10, 5),
    (4, 10, 10),
    (8, 32, 10),
    (8, 32, 32),
])
def test_matches_numpy_map_at_k(
    batch_size: int,
    num_labels: int,
    k: int
):
    # given
    torch.manual_seed(1234)

    y_pred = torch.rand(batch_size, num_labels)
    y_true = torch.randint(
        0,
        2,
        (batch_size, num_labels),
    )

    # ensure at least one positive label per sample
    empty_rows = y_true.sum(dim=1) == 0

    for idx in torch.where(empty_rows)[0]:
        y_true[idx, 0] = 1

    metric = MeanAveragePrecisionAtK(k=k)

    # when
    metric.update(y_pred, y_true)
    torch_result = metric.compute().item()

    numpy_result = map_at_k(
        y_true=y_true.numpy(),
        y_pred=y_pred.numpy(),
        k=k,
        break_ties=False,
        return_mean=True
    )

    # then
    assert np.isclose(torch_result, numpy_result, atol=1e-6)


def test_matches_numpy_map_at_k_multiple_updates():
    # given
    torch.manual_seed(5678)

    batch_size = 16
    num_labels = 20
    k = 10

    y_pred = torch.rand(batch_size, num_labels)

    y_true = torch.randint(
        0,
        2,
        (batch_size, num_labels),
    )

    # ensure valid rows
    empty_rows = y_true.sum(dim=1) == 0

    for idx in torch.where(empty_rows)[0]:
        y_true[idx, 0] = 1

    metric = MeanAveragePrecisionAtK(k=k)

    # when

    # split into several updates
    metric.update(y_pred[:5], y_true[:5])
    metric.update(y_pred[5:11], y_true[5:11])
    metric.update(y_pred[11:], y_true[11:])

    torch_result = metric.compute().item()

    numpy_result = map_at_k(
        y_true=y_true.numpy(),
        y_pred=y_pred.numpy(),
        k=k,
        break_ties=False,
        return_mean=True
    )

    # then
    assert np.isclose(torch_result, numpy_result, atol=1e-6)
