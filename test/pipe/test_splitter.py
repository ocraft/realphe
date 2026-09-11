import numpy as np
from xarray import DataArray, Dataset

from realphe.pipe.splitter import (
    KFoldSplitter,
    MultilabelStratifiedKFoldSplitter,
    get_fold,
    multilabel_train_test_split,
    sample,
    split,
    train_test_split,
)


def test_samples_dataset():
    dataset = Dataset()
    dataset['X'] = DataArray(np.arange(9).reshape(3, 3), dims=['sample', 'feature'])

    sample_set = sample(dataset, frac=0.33, random_seed=42)

    assert len(sample_set.sample) == round(len(dataset.sample) / 3)
    assert not sample_set.equals(dataset)


def test_train_test_splits_the_dataset():
    dataset = Dataset()
    dataset['X'] = DataArray(np.arange(9).reshape(3, 3), dims=['sample', 'feature'])

    train_set, test_set = train_test_split(dataset, frac=0.33, random_seed=42)

    assert len(test_set.sample) == round(len(dataset.sample) / 3)
    assert len(train_set.sample) == round(2 * len(dataset.sample) / 3)


def test_multilabel_train_test_split_for_dataset():
    dataset = Dataset({
        'X': DataArray([[1, 1],
                        [2, 2],
                        [3, 3],
                        [4, 4],
                        [5, 5],
                        [6, 6],
                        [7, 7],
                        [8, 8]],
                       dims=['sample', 'feature']),
        'Y': DataArray([[1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 1],
                        [1, 1, 0],
                        [0, 1, 1]],
                       dims=['sample', 'target'])
    })

    train_set, test_set = multilabel_train_test_split(
        dataset, stratify=True, random_seed=15, train_size=0.7)

    assert len(train_set.sample) == 5
    assert len(test_set.sample) == 3
    assert np.count_nonzero(test_set.Y.mean(axis=0)) == 3  # all labels are represented


def test_splits_dataset_with_kfold_splitter():
    n_samples = 6
    n_features = 3
    n_splits = 2
    n_samples_in_fold = n_samples / n_splits
    seed = 42
    dataset = Dataset()
    dataset['X'] = DataArray(np.arange(n_samples * n_features).reshape(n_samples, n_features),
                             dims=['sample', 'feature'])
    dataset['Z'] = DataArray(np.arange(n_samples * n_features).reshape(n_samples, n_features),
                             dims=['sample', 'feature'])
    dataset['Y'] = DataArray(np.arange(n_samples).reshape(-1, 1), dims=['sample', 'target'])

    splitter = KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    dataset = split(dataset, splitter)
    train, valid = get_fold(dataset, fold=0)

    assert splitter.n_splits == n_splits
    assert splitter.random_state == seed
    assert splitter.shuffle is True
    assert dataset.attrs['n_splits'] == n_splits
    assert train.X.shape == (n_samples_in_fold, len(dataset.feature))
    assert train.Y.shape == (n_samples_in_fold, 1)
    assert train.Z.shape == (n_samples_in_fold, n_features)
    assert valid.X.shape == (n_samples_in_fold, len(dataset.feature))
    assert valid.Z.shape == (n_samples_in_fold, n_features)


def test_splits_dataset_with_multilabel_kfold_splitter():
    n_splits = 2
    seed = 15
    dataset = Dataset({
        'X': DataArray([[1, 1],
                        [2, 2],
                        [3, 3],
                        [4, 4],
                        [5, 5],
                        [6, 6],
                        [7, 7],
                        [8, 8]],
                       dims=['sample', 'feature']),
        'Y': DataArray([[1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 1],
                        [1, 1, 0],
                        [0, 1, 1]],
                       dims=['sample', 'target'])
    })

    n_samples = len(dataset.sample)
    splitter = MultilabelStratifiedKFoldSplitter(n_splits=n_splits, shuffle=True, random_state=seed)
    dataset = split(dataset, splitter)
    train, valid = get_fold(dataset, fold=0)

    assert splitter.n_splits == n_splits
    assert splitter.random_state == seed
    assert splitter.shuffle is True
    assert len(train.sample) == n_samples / n_splits
    assert len(valid.sample) == n_samples / n_splits
    assert np.count_nonzero(valid.Y.mean(axis=0)) == 3  # all labels are represented
