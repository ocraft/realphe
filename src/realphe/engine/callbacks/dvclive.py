from dvclive.live import Live
from torch import nn

from .. import Callback, History


class DVCLive(Callback):
    def __init__(self, live: Live) -> None:
        self.live = live

    def on_epoch_end(self, model: nn.Module, history: History):
        for metric, values in history:
            self.live.log_metric(metric.replace('_', '/'), values[-1])
        self.live.next_step()

    def on_train_end(self, model: nn.Module, history: History):
        self.live.end()
