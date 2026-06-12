from typing import Dict

from torch import nn
import torchmetrics

from realphe.env import get_log
log = get_log(__name__)


class History:
    def __init__(
        self,
        metrics: Dict[str, torchmetrics.Metric],
        n_train_samples: int,
        n_eval_samples: int,
    ) -> None:
        self.history = {}
        self.history['train_loss'] = []
        for metric in metrics:
            self.history['train_' + metric] = []
        self.history['eval_loss'] = []
        for metric in metrics:
            self.history['eval_' + metric] = []

        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        self.stop_training = False

    def append(self, key: str, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    def update(self, step: str, total_loss: float, metrics: Dict[str, torchmetrics.Metric]):
        self.append(
            f'{step}_loss',
            total_loss / (self.n_train_samples if step == 'train' else self.n_eval_samples)
        )
        for metric, value in metrics.items():
            self.append(f'{step}_{metric}', value.compute().item())

    def __getitem__(self, key):
        return self.history[key]

    def __iter__(self):
        yield from self.history.items()


class Callback:
    def on_train_start(self, model: nn.Module, history: History):
        ...

    def on_train_end(self, model: nn.Module, history: History):
        ...

    def on_train_step_start(self, model: nn.Module, history: History):
        ...

    def on_train_step_end(self, model: nn.Module, history: History):
        ...

    def on_epoch_start(self, model: nn.Module, history: History):
        ...

    def on_epoch_end(self, model: nn.Module, history: History):
        ...
