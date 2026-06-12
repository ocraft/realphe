import torch
from torch import nn

from .. import Callback, History


class StepLR(Callback):
    def __init__(self, scheduler: torch.optim.lr_scheduler.StepLR) -> None:
        self.scheduler = scheduler

    def on_epoch_end(self, model: nn.Module, history: History):
        history.append('train_lr', self.scheduler.get_last_lr()[-1])
        self.scheduler.step()
