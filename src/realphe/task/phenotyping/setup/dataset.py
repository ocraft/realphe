import math
import multiprocessing as mp
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence

from realphe.env import get_log


log = get_log(__name__)


class PhenotypingDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        signals_df: pd.DataFrame,
        target_df: pd.DataFrame,
        shuffle: bool = True,
        reshuffle_each_iteration: bool = True,
        seed: Optional[int] = None
    ) -> None:

        super().__init__()

        self.shuffle = shuffle
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.seed = seed
        self.epoch_ = 0

        # IMPORTANT
        # samples must be contiguous in dataframe
        x_df = signals_df
        y_df = target_df

        # one contiguous ndarray
        self.x_values = x_df.to_numpy(dtype=np.float32, copy=False)
        self.y_values = y_df.to_numpy(dtype=np.int8, copy=False)

        # integer sample ids from first MultiIndex level
        sample_codes = cast(pd.MultiIndex, x_df.index).codes[x_df.index.names.index('sample')]
        boundaries = np.flatnonzero(np.diff(sample_codes)) + 1

        self.offsets = np.concatenate(([0], boundaries))
        self.ends = np.concatenate((boundaries, [len(sample_codes)]))
        self.n_samples = len(self.offsets)
        self.n_features = len(signals_df.columns)
        self.n_targets = len(target_df.columns)

    def _get_worker_setup(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return 0, self.n_samples, self.seed

        if mp.get_start_method() != 'fork':
            log.warning(
                "The multiprocessing start method is not set to 'fork'. "
                "This may result in NumPy arrays being duplicated in memory."
            )

        per_worker = math.ceil(self.n_samples / worker_info.num_workers)
        start = worker_info.id * per_worker
        end = min(start + per_worker, self.n_samples)

        return start, end, worker_info.seed

    def __len__(self):
        return self.n_samples

    def __iter__(self):

        start, end, seed = self._get_worker_setup()
        indices = list(range(start, end))

        if self.shuffle:
            rng = np.random.default_rng(
                seed + self.epoch_
                if self.reshuffle_each_iteration and seed is not None
                else seed
            )

            rng.shuffle(indices)

        self.epoch_ += 1

        for idx in indices:
            yield (
                torch.from_numpy(self.x_values[self.offsets[idx]:self.ends[idx]]),
                torch.from_numpy(self.y_values[idx])
            )

    @property
    def element_spec(self):
        return (
            (self.n_samples, int(np.max(self.ends - self.offsets)), self.n_features),
            (self.n_samples, self.n_targets),
        )


def packed_sample_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[PackedSequence, torch.Tensor]:

    batch_x = [sample[0] for sample in batch]
    batch_y = [sample[1] for sample in batch]

    return (
        torch.nn.utils.rnn.pack_sequence(
            batch_x,
            enforce_sorted=False
        ),
        torch.stack(batch_y)
    )


def packed_ts_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[PackedSequence, PackedSequence]:

    batch_x = [sample[0] for sample in batch]
    lengths = [len(sample[0]) for sample in batch]
    batch_y = [sample[1].repeat(lengths[i]).reshape(lengths[i], -1)
               for i, sample in enumerate(batch)]

    return (
        torch.nn.utils.rnn.pack_sequence(
            batch_x,
            enforce_sorted=False
        ),
        torch.nn.utils.rnn.pack_sequence(
            batch_y,
            enforce_sorted=False
        ),
    )
