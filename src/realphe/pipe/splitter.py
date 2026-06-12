from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, cast

from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples
from xarray import DataArray, Dataset


class Splitter(ABC):
    @abstractmethod
    def split(self, dataset: Dataset) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        raise NotImplementedError

    @property
    def n_splits(self) -> int:
        return 0

    @property
    def random_state(self) -> Optional[int]:
        return None

    @property
    def shuffle(self) -> bool:
        return False


def sample(dataset: Dataset,
           frac: Optional[float] = None,
           n_samples: Optional[int] = None,
           random_seed: Optional[int] = None) -> Dataset:
    n_samples = 1 if frac is None and n_samples is None else n_samples
    n_set = len(dataset.sample)
    n_samples = round(frac * n_set) if frac is not None else n_samples

    if random_seed:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    return dataset.isel(sample=DataArray(rng.choice(n_set, n_samples, replace=False),
                                         dims=['sample']))


def train_test_split(dataset: Dataset,
                     frac: float = 0.2,
                     random_seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    test_set = sample(dataset, frac=frac, random_seed=random_seed)
    train_set = dataset.drop_sel(sample=test_set.sample)
    return train_set, test_set


def multilabel_train_test_split(
    dataset: Dataset,
    train_size: Optional[float | int] = None,
    test_size: Optional[float | int] = None,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = True,
    target_var: str = 'Y',
    return_index: bool = False
) -> Tuple[Dataset, Dataset] | Tuple[np.ndarray, np.ndarray]:
    # pylint: disable=too-many-arguments

    if not stratify:
        X, Y = sklearn.model_selection.train_test_split(
            dataset.sample,
            test_size=test_size,
            train_size=train_size,
            random_state=random_seed,
            stratify=None,
            shuffle=shuffle)
        return dataset.sel(sample=X), dataset.sel(sample=Y)

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"

    # compute train and test samples size
    n_samples = _num_samples(dataset.sample)
    n_train, n_test = _validate_shuffle_split(
        n_samples,
        test_size,
        train_size,
        default_test_size=0.25)
    splitter = MultilabelStratifiedShuffleSplit(
        test_size=n_test,  # pyright: ignore[reportArgumentType]
        train_size=n_train,
        random_state=random_seed)

    train, test = next(splitter.split(X=dataset.sample, y=dataset[target_var]))
    if return_index:
        return (
            cast(np.ndarray, _safe_indexing(dataset.sample, train)),
            cast(np.ndarray, _safe_indexing(dataset.sample, test))
        )
    return (dataset.sel(sample=_safe_indexing(dataset.sample, train)),
            dataset.sel(sample=_safe_indexing(dataset.sample, test)))


def split(dataset: Dataset, splitter: Splitter, coord: str = 'fold') -> Dataset:
    dataset.coords[coord] = ('sample', np.zeros_like(
        dataset['sample'], dtype='int8'))
    for fold, (_, val) in enumerate(splitter.split(dataset)):
        dataset.coords[coord].values[val] = fold
    dataset.attrs['n_splits'] = splitter.n_splits
    return dataset


def get_fold(dataset: Dataset, fold: int, coord: str = 'fold') -> Tuple[Dataset, Dataset]:
    """Create a tuple containing copies of the `dataset` that has been `split`,
    where the first copy contains the data outside of the selected `fold`, and
    the second copy contains the data within the selected `fold`.
    """
    return (dataset.sel(sample=dataset[coord] != fold),
            dataset.sel(sample=dataset[coord] == fold))


def get_fold_train(dataset: Dataset, fold: int, coord: str = 'fold') -> Dataset:
    return dataset.sel(sample=dataset[coord] != fold)


def get_fold_val(dataset: Dataset, fold: int, coord: str = 'fold') -> Dataset:
    return dataset.sel(sample=dataset[coord] == fold)


class KFoldSplitter(Splitter):
    def __init__(self, n_splits=3, shuffle=False, random_state=None, var: str = 'Y'):
        self.splitter = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._var = var

    def split(self, dataset: Dataset) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        return self.splitter.split(np.zeros(len(dataset.sample)), dataset[self._var].values)

    @property
    def n_splits(self) -> int:
        return self.splitter.n_splits

    @property
    def random_state(self) -> Optional[int]:
        return self.splitter.random_state

    @property
    def shuffle(self) -> bool:
        return self.splitter.shuffle


class MultilabelStratifiedKFoldSplitter(Splitter):
    def __init__(self, n_splits=3, shuffle=False, random_state=None, var: str = 'Y'):
        self.splitter = MultilabelStratifiedKFold(n_splits=n_splits,
                                                  shuffle=shuffle,
                                                  random_state=random_state)
        self._var = var

    def split(self, dataset: Dataset) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        return self.splitter.split(np.zeros(len(dataset.sample)), dataset[self._var].values)

    @property
    def n_splits(self) -> int:
        return self.splitter.n_splits

    @property
    def random_state(self) -> Optional[int]:
        return self.splitter.random_state

    @property
    def shuffle(self) -> bool:
        return self.splitter.shuffle
