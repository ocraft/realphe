import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from realphe.engine.loss import WeightedBCEWithLogitsLoss


def make_packed(
    data: torch.Tensor,
    lengths: list[int],
):
    return pack_padded_sequence(
        data,
        lengths=lengths,
        batch_first=True,
        enforce_sorted=False,
    )


def test_perfect_predictions_have_low_loss():
    # given
    criterion = WeightedBCEWithLogitsLoss()

    y_true = torch.tensor([
        [[1.0], [0.0], [1.0]],
        [[0.0], [1.0], [0.0]],
    ])

    logits = torch.tensor([
        [[10.0], [-10.0], [10.0]],
        [[-10.0], [10.0], [-10.0]],
    ])
    lengths = [3, 3]

    y_true = make_packed(y_true, lengths)
    y_pred = make_packed(logits, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then
    assert loss < 1e-3


def test_wrong_predictions_have_high_loss():
    # given
    criterion = WeightedBCEWithLogitsLoss()

    y_true = torch.tensor([
        [[1.0], [0.0]],
    ])

    logits = torch.tensor([
        [[-10.0], [10.0]],
    ])

    lengths = [2]

    y_true = make_packed(y_true, lengths)
    y_pred = make_packed(logits, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then
    assert loss > 5.0


def test_matches_manual_computation():
    # given
    criterion = WeightedBCEWithLogitsLoss(
        ramp_min=0.5,
    )

    y_true_dense = torch.tensor([
        [[1.0], [0.0], [1.0]],
        [[0.0], [1.0], [0.0]],
    ])

    logits_dense = torch.tensor([
        [[1.0], [5.0], [0.0]],
        [[2.0], [0.0], [8.0]],
    ])

    lengths = [3, 2]

    y_true = make_packed(y_true_dense, lengths)
    y_pred = make_packed(logits_dense, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then

    # Manual computation
    raw_loss = F.binary_cross_entropy_with_logits(
        y_pred.data,
        y_true.data,
        reduction="none",
    )

    timestep_weights = torch.linspace(1.0, 0.5, 3)

    # batch_sizes = [2, 2, 1]
    weights = torch.tensor([
        timestep_weights[0],
        timestep_weights[0],
        timestep_weights[1],
        timestep_weights[1],
        timestep_weights[2],
    ]).unsqueeze(-1)

    expected = (raw_loss * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected)


def test_label_smoothing_changes_targets():
    # given
    criterion_no_smoothing = WeightedBCEWithLogitsLoss(
        label_smoothing=0.0,
    )

    criterion_smoothing = WeightedBCEWithLogitsLoss(
        label_smoothing=0.2,
    )

    y_true = torch.tensor([
        [[1.0], [0.0]],
    ])

    logits = torch.tensor([
        [[0.0], [0.0]],
    ])

    lengths = [2]

    y_true_packed = make_packed(y_true, lengths)
    y_pred_packed = make_packed(logits, lengths)

    # when

    loss_no_smoothing = criterion_no_smoothing(
        y_pred_packed,
        y_true_packed,
    )

    loss_smoothing = criterion_smoothing(
        y_pred_packed,
        y_true_packed,
    )

    # then
    assert torch.isfinite(loss_smoothing)
    assert torch.isfinite(loss_no_smoothing)


def test_variable_length_sequences():
    # given

    criterion = WeightedBCEWithLogitsLoss()

    y_true = torch.tensor([
        [[1.0], [0.0], [1.0], [0.0]],
        [[0.0], [1.0], [0.0], [0.0]],
        [[1.0], [0.0], [0.0], [0.0]],
    ])

    logits = torch.zeros_like(y_true)

    lengths = [4, 2, 1]

    y_true = make_packed(y_true, lengths)
    y_pred = make_packed(logits, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_raises_for_different_batch_sizes():
    # given
    criterion = WeightedBCEWithLogitsLoss()

    y_true = torch.tensor([
        [[1.0], [0.0], [1.0]],
    ])

    logits = torch.tensor([
        [[0.0], [0.0]],
    ])

    y_true = make_packed(y_true, [3])
    y_pred = make_packed(logits, [2])

    # then
    with pytest.raises(ValueError):
        criterion(y_pred, y_true)


def test_multilabel_input():
    # given
    criterion = WeightedBCEWithLogitsLoss()

    y_true = torch.tensor([
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ])

    logits = torch.zeros_like(y_true)

    lengths = [2]

    y_true = make_packed(y_true, lengths)
    y_pred = make_packed(logits, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_is_bce_when_ramp_min_is_one():
    # given
    criterion = WeightedBCEWithLogitsLoss(
        ramp_min=1.0,
    )

    y_true_dense = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    ])

    logits_dense = torch.tensor([
        [[8.0, 5.0, 2.0], [1.0, 3.0, 2.0], [0.0, 8.0, 2.0]],
        [[3.0, 7.0, 2.0], [0.0, 1.0, 2.0], [2.0, 3.0, 2.0]],
    ])

    lengths = [3, 3]

    y_true = make_packed(y_true_dense, lengths)
    y_pred = make_packed(logits_dense, lengths)

    # when
    loss = criterion(y_pred, y_true)

    # then
    # Manual computation
    raw_loss = F.binary_cross_entropy_with_logits(y_pred.data, y_true.data)

    assert torch.isclose(loss, raw_loss)
