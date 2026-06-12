import io

import torch
from torch import nn

from realphe.env import get_log

from .. import Callback, History


log = get_log(__name__)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = 'eval_loss',
        mode: str = 'min',
        patience: int = 0,
        min_delta: float = 0.0,
        restore_best_weights: bool = False,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta

        self.patience_counter = 0

        if mode == 'max':
            self.best_metric = float('-inf')
            self.compare_fn = (lambda current, best: current >= (best + self.min_delta))
        else:
            self.best_metric = float('inf')
            self.compare_fn = (lambda current, best: current <= (best - self.min_delta))

        self.best_state = io.BytesIO() if restore_best_weights else None

    def on_epoch_end(self, model: nn.Module, history: History):
        current_metric = history[self.monitor][-1]

        if self.compare_fn(current_metric, self.best_metric):
            log.info('New best! - %.4f [%s]', current_metric, self.monitor)
            self.best_metric = current_metric
            self.patience_counter = 0

            if self.best_state is not None:
                self.best_state.seek(0)
                self.best_state.truncate(0)

                torch.save(model.state_dict(), self.best_state)

        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            log.info('Early stopping! %d >= %d', self.patience_counter, self.patience)
            history.stop_training = True

    def on_train_end(self, model: nn.Module, history: History):
        if self.best_state is not None:
            self.best_state.seek(0)

            state_dict = torch.load(
                self.best_state,
                map_location=model.device if hasattr(model, 'device') else None,
            )

            model.load_state_dict(state_dict)

            self.best_state.close()
            del self.best_state
