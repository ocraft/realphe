import time

from torch import nn

from realphe.env import format_duration, get_log

from .. import Callback, History


log = get_log(__name__)


class Log(Callback):
    def __init__(
        self,
        n_epochs: int,
    ) -> None:
        self.time_start_epoch = 0.0
        self.time_end_epoch = 0.0
        self.time_start_train_step = 0.0
        self.time_end_train_step = 0.0
        self.n_epochs = n_epochs
        self.epoch = 0

    def on_epoch_start(self, model: nn.Module, history: History):
        self.time_start_epoch = time.perf_counter()

    def on_epoch_end(self, model: nn.Module, history: History):
        self.time_end_epoch = time.perf_counter()

        log_str = 'Epoch %4d/%4d [%s %s/step]'
        to_log = []
        for metric, values in history:
            if metric != 'train_lr':
                log_str += f' - {metric}: %.4f'
            else:
                log_str += f' - {metric}: %.6f'
            to_log.append(values[-1])

        log.info(
            log_str,
            self.epoch + 1,
            self.n_epochs,
            format_duration(self.time_end_epoch - self.time_start_epoch),
            format_duration((self.time_end_train_step -
                            self.time_start_train_step) / history.n_train_samples),
            *to_log
        )

        self.epoch += 1

    def on_train_step_start(self, model: nn.Module, history: History):
        self.time_start_train_step = time.perf_counter()

    def on_train_step_end(self, model: nn.Module, history: History):
        self.time_end_train_step = time.perf_counter()
