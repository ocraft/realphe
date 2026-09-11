import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.utils.data import DataLoader

from realphe.task.phenotyping.setup.dataset import (
    PhenotypingDataset,
    packed_sample_collate_fn,
    packed_ts_collate_fn,
)


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self):

        # intentionally unsorted lengths
        self.samples = [
            (
                torch.full((3, 2), 10.0),   # sample 0
                torch.tensor([0], dtype=torch.int8)
            ),
            (
                torch.full((5, 2), 20.0),   # sample 1
                torch.tensor([1], dtype=torch.int8)
            ),
            (
                torch.full((2, 2), 30.0),   # sample 2
                torch.tensor([2], dtype=torch.int8)
            ),
            (
                torch.full((4, 2), 40.0),   # sample 3
                torch.tensor([3], dtype=torch.int8)
            ),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@pytest.fixture(name='sample_data')
def fixture_sample_data():

    rng = np.random.default_rng(42)

    n_features = 4
    lengths = [3, 5, 2]

    x_parts = []
    y_parts = []

    sample_ids = []
    step_ids = []

    for sample_id, length in enumerate(lengths):

        x = rng.normal(size=(length, n_features)).astype(np.float32)
        x_parts.append(x)

        sample_ids.extend([sample_id] * length)
        step_ids.extend(range(length))

        y_parts.append(np.array([sample_id % 2], dtype=np.int8))

    x_values = np.concatenate(x_parts, axis=0)

    x_index = pd.MultiIndex.from_arrays(
        [sample_ids, step_ids],
        names=["sample", "step"]
    )

    x_df = pd.DataFrame(
        x_values,
        index=x_index,
        columns=[f"f{i}" for i in range(n_features)]
    )

    y_index = pd.Index(
        range(len(lengths)),
        name="sample"
    )

    y_df = pd.DataFrame(
        np.stack(y_parts),
        index=y_index,
        columns=["target"]
    )

    return x_df, y_df, lengths


@pytest.fixture(name='packed_ts_batch')
def fixture_packed_ts_batch():

    dataset = DummyDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=packed_ts_collate_fn,
        persistent_workers=False,
    )

    return next(iter(loader))


@pytest.fixture(name='unpacked_ts_batch')
def fixture_unpacked_ts_batch(packed_ts_batch):

    packed_x, packed_y = packed_ts_batch

    unpacked_x, x_lengths = pad_packed_sequence(
        packed_x,
        batch_first=True
    )

    unpacked_y, y_lengths = pad_packed_sequence(
        packed_y,
        batch_first=True
    )

    return unpacked_x, x_lengths, unpacked_y, y_lengths


def test_dataset_length_preprocessing(sample_data):
    # given
    x_df, y_df, lengths = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=False)

    # then
    expected_offsets = np.array([0, 3, 8])
    expected_ends = np.array([3, 8, 10])

    assert dataset.n_samples == len(lengths)
    np.testing.assert_array_equal(
        dataset.offsets,
        expected_offsets
    )
    np.testing.assert_array_equal(
        dataset.ends,
        expected_ends
    )


def test_iteration_returns_correct_shapes(sample_data):
    # given
    x_df, y_df, lengths = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=False)

    # when
    samples = list(iter(dataset))

    # then
    assert len(samples) == 3

    for i, (x, y) in enumerate(samples):

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        assert x.shape[0] == lengths[i]
        assert x.shape[1] == 4

        assert y.shape == (1,)

        assert x.dtype == torch.float32
        assert y.dtype == torch.int8


def test_shuffle_changes_order(sample_data):
    # given
    x_df, y_df, _ = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=True, reshuffle_each_iteration=True, seed=42)

    # when
    order1 = [int(y.item()) for _, y in iter(dataset)]
    order2 = [int(y.item()) for _, y in iter(dataset)]

    # then
    assert order1 != order2


def test_sample_collate_function(sample_data):
    # given
    x_df, y_df, _ = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=packed_sample_collate_fn,
        num_workers=0
    )

    # when
    packed_x, y = next(iter(loader))
    unpacked_x, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_x, batch_first=True)

    # then
    assert isinstance(
        packed_x,
        torch.nn.utils.rnn.PackedSequence
    )
    assert y.shape == (2, 1)
    assert y.dtype == torch.int8
    assert unpacked_x.shape == (2, 5, 4)
    assert lengths.tolist() == [3, 5]


def test_all_samples_seen_without_shuffle(sample_data):
    # given
    x_df, y_df, _ = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=False)

    # when
    seen = []
    for _, y in dataset:
        seen.append(int(y.item()))

    # then
    assert seen == [0, 1, 0]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_multiworker(sample_data, num_workers):
    # given
    x_df, y_df, lengths = sample_data
    dataset = PhenotypingDataset(x_df, y_df, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=packed_sample_collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    # when
    total = 0
    for _, y in loader:
        total += y.shape[0]

    # then
    assert total == len(lengths)


def test_packed_ts_collate_returns_packed_sequences(packed_ts_batch):
    # given
    packed_x, packed_y = packed_ts_batch

    # then
    assert isinstance(packed_x, PackedSequence)
    assert isinstance(packed_y, PackedSequence)


def test_packed_ts_collate_preserves_order(unpacked_ts_batch):
    # given
    unpacked_x, _, _, _ = unpacked_ts_batch

    # when
    recovered_ids = [
        int(unpacked_x[i, 0, 0].item())
        for i in range(unpacked_x.shape[0])
    ]

    # then
    assert recovered_ids == [10, 20, 30, 40]


def test_packed_ts_collate_preserves_lengths(unpacked_ts_batch):
    # given
    _, x_lengths, _, y_lengths = unpacked_ts_batch

    # then
    assert x_lengths.tolist() == [3, 5, 2, 4]
    assert y_lengths.tolist() == [3, 5, 2, 4]


def test_packed_ts_collate_repeats_targets_correctly(unpacked_ts_batch):

    # given
    _, _, unpacked_y, y_lengths = unpacked_ts_batch

    # then
    expected_y = [
        [0] * 3,
        [1] * 5,
        [2] * 2,
        [3] * 4,
    ]

    for i, length in enumerate(y_lengths.tolist()):

        recovered = (
            unpacked_y[i, :length]
            .reshape(-1)
            .tolist()
        )

        assert recovered == expected_y[i]


def test_packed_ts_collate_preserves_x_y_alignment(unpacked_ts_batch):

    # given
    unpacked_x, x_lengths, unpacked_y, _ = unpacked_ts_batch

    # then
    for i, length in enumerate(x_lengths.tolist()):

        x_id = int(unpacked_x[i, 0, 0].item())
        y_id = int(unpacked_y[i, 0, 0].item())

        assert x_id // 10 - 1 == y_id
