import torch
from torch import nn

from .. import Callback, History


class ReduceLROnPlateau(Callback):
    def __init__(self, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, monitor: str) -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, model: nn.Module, history: History):
        history.append('train_lr', self.scheduler.get_last_lr()[-1])
        self.scheduler.step(history[self.monitor][-1])
