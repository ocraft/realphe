from typing import Callable, Dict, List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import torchmetrics

from realphe.engine.callbacks.log import Log

from . import Callback, History


def fit(
    # fit params
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    # compile params
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: Dict[str, torchmetrics.Metric],
    callbacks: List[Callback] = []
) -> Tuple[nn.Module, Dict[str, List]]:

    callbacks = [
        *callbacks,
        Log(epochs)
    ]

    history = History(metrics, len(train_loader), len(val_loader))

    for metric in metrics.values():
        metric.to(device)

    # on train start
    for callback in callbacks:
        callback.on_train_start(model, history)

    for epoch in range(epochs):
        # on epoch start
        for callback in callbacks:
            callback.on_epoch_start(model, history)

        # train step
        for callback in callbacks:
            callback.on_train_step_start(model, history)

        train_step(model, train_loader, criterion, optimizer, metrics, history, device)

        for callback in callbacks:
            callback.on_train_step_end(model, history)

        # eval step
        eval_step(model, val_loader, criterion, metrics, history, device)

        # on epoch end
        for callback in callbacks:
            callback.on_epoch_end(model, history)

        if history.stop_training:
            break

    # on train end
    for callback in callbacks:
        callback.on_train_end(model, history)

    return model, history.history


def train_step(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, torchmetrics.Metric],
    history: History,
    device: torch.device
):
    model.train()
    total_loss = 0.0
    for metric in metrics.values():
        metric.reset()

    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        update_metrics(Y_pred, Y_batch, metrics)

    history.update('train', total_loss, metrics)


@torch.no_grad
def update_metrics(
    Y_pred: PackedSequence | torch.Tensor,
    Y_batch: PackedSequence | torch.Tensor,
    metrics: Dict[str, torchmetrics.Metric],
):

    if isinstance(Y_pred, PackedSequence) and isinstance(Y_batch, PackedSequence):
        Y_hat, lengths_pred = torch.nn.utils.rnn.pad_packed_sequence(Y_pred, batch_first=True)
        Y_true, lengths_batch = torch.nn.utils.rnn.pad_packed_sequence(Y_batch, batch_first=True)
        for metric, value in metrics.items():
            if metric.startswith('ls_'):
                value.update(
                    Y_hat[torch.arange(Y_hat.size(0)), lengths_pred-1],
                    Y_true[torch.arange(Y_true.size(0)), lengths_batch-1],
                )
            elif metric.startswith('fs_'):
                value.update(
                    Y_hat[torch.arange(Y_hat.size(0)), 0],
                    Y_true[torch.arange(Y_true.size(0)), 0],
                )
            else:
                value.update(Y_pred.data, Y_batch.data)
    elif isinstance(Y_pred, torch.Tensor) and isinstance(Y_batch, torch.Tensor):
        for metric, value in metrics.items():
            value.update(Y_pred, Y_batch)
    else:
        raise ValueError('The inputs to calculate the metrics must be of the same type: '
                         '`Tensor` or `PackedSequence`.')


def eval_step(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: Dict[str, torchmetrics.Metric],
    history: History,
    device: torch.device
):
    model.eval()
    total_loss = 0.0
    for metric in metrics.values():
        metric.reset()

    with torch.inference_mode():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch.float())
            total_loss += loss.item()

            update_metrics(Y_pred, Y_batch, metrics)

    history.update('eval', total_loss, metrics)
