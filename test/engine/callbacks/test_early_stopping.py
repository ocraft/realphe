import pytest
import torch
from torch import nn

from realphe.engine import History
from realphe.engine.callbacks.early_stopping import EarlyStopping


@pytest.fixture(name='model')
def test_model():
    torch.manual_seed(0)
    return nn.Linear(4, 1)


def clone_state_dict(model: nn.Module):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def assert_state_dict_equal(a, b):
    assert a.keys() == b.keys()

    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_restores_best_weights(model):
    # given
    callback = EarlyStopping(
        monitor='val_metric',
        mode='max',
        patience=10,
        restore_best_weights=True,
    )
    history = History({}, 1, 1)

    #
    # epoch 1 -> best
    #
    history.append('val_metric', 0.5)

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)

    best_state = clone_state_dict(model)

    callback.on_epoch_end(model, history)

    #
    # epoch 2 -> worse
    #
    history.append('val_metric', 0.4)

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(2.0)

    callback.on_epoch_end(model, history)

    #
    # epoch 3 -> worse again
    #
    history.append('val_metric', 0.3)

    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(3.0)

    callback.on_epoch_end(model, history)

    # when
    callback.on_train_end(model, history)

    # then
    assert_state_dict_equal(model.state_dict(), best_state)


def test_stops_training_after_patience_exceeded(model):
    # given
    callback = EarlyStopping(
        monitor='val_metric',
        mode='max',
        patience=2,
    )

    history = History({}, 1, 1)

    # when

    # best epoch
    history.append('val_metric', 1.0)
    callback.on_epoch_end(model, history)

    assert history.stop_training is False

    # no improvement 1
    history.append('val_metric', 0.9)
    callback.on_epoch_end(model, history)

    assert history.stop_training is False

    # no improvement 2
    history.append('val_metric', 0.8)
    callback.on_epoch_end(model, history)

    # then
    assert history.stop_training is True


def test_resets_patience_counter_after_improvement(model):
    # given
    callback = EarlyStopping(
        monitor='val_metric',
        mode='max',
        patience=3,
    )

    history = History({}, 1, 1)

    # when
    history.append('val_metric', 1.0)
    callback.on_epoch_end(model, history)

    history.append('val_metric', 0.9)
    callback.on_epoch_end(model, history)

    history.append('val_metric', 1.1)
    callback.on_epoch_end(model, history)

    # then
    assert callback.patience_counter == 0
    assert history.stop_training is False


def test_min_mode(model):
    # given
    callback = EarlyStopping(
        monitor='val_metric',
        mode='min',
        patience=10,
    )

    history = History({}, 1, 1)

    # when
    history.append('val_metric', 10.0)
    callback.on_epoch_end(model, history)

    history.append('val_metric', 5.0)
    callback.on_epoch_end(model, history)

    # then
    assert callback.best_metric == 5.0
    assert callback.patience_counter == 0


def test_min_delta_for_max_mode(model):
    # given
    callback = EarlyStopping(
        monitor="val_metric",
        mode="max",
        min_delta=0.1,
        patience=10,
    )

    history = History({}, 1, 1)

    # initial best
    history.append('val_metric', 10.0)
    callback.on_epoch_end(model, history)

    # when

    # improvement too small
    history.append('val_metric', 10.05)
    callback.on_epoch_end(model, history)

    # then
    assert callback.best_metric == 10.0
    assert callback.patience_counter == 1


def test_min_delta_for_min_mode(model):
    # given
    callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.1,
        patience=10,
    )

    history = History({}, 1, 1)

    # initial best
    history.append('val_loss', 1.0)
    callback.on_epoch_end(model, history)

    # when

    # decrease too small
    history.append('val_loss', 0.95)
    callback.on_epoch_end(model, history)

    # then
    assert callback.best_metric == 1.0
    assert callback.patience_counter == 1


def test_does_not_restore_weights_when_disabled(model):
    # given
    callback = EarlyStopping(
        monitor='val_metric',
        mode='max',
        restore_best_weights=False,
    )

    history = History({}, 1, 1)

    history.append('val_metric', 1.0)

    with torch.no_grad():
        model.weight.fill_(1.0)

    callback.on_epoch_end(model, history)

    with torch.no_grad():
        model.weight.fill_(5.0)

    modified_state = clone_state_dict(model)

    # when
    callback.on_train_end(model, history)

    # then
    assert_state_dict_equal(model.state_dict(), modified_state)
