from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
from typing import Callable, List, Optional, cast

import pandas as pd
from typing_extensions import Self
import xarray as xr
from xarray import Dataset
from xarray.core.utils import Frozen


log = logging.getLogger(__name__)


# The Transformer should be a technical and internal unit that is efficient and capable of
# performing all transformations in place without copying. If a copy is necessary, it should be done
# at a higher level. Every transformation could by backed y one of: memory, file (out-of-core) and
# maybe in the future by distributed cluster.
class Transformer(ABC):

    def fit(self, dataset: Dataset | pd.DataFrame) -> Self:
        del dataset  # unused argument
        return self

    def partial_fit(self, dataset: Dataset | pd.DataFrame) -> Self:
        del dataset  # unused argument
        return self

    @abstractmethod
    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        raise NotImplementedError

    @property
    def can_fit(self):
        return type(self).fit != Transformer.fit

    @property
    def can_partial_fit(self):
        return type(self).partial_fit != Transformer.partial_fit


def partial_fit(
    transformers: List[Transformer]
) -> Callable[[Dataset | pd.DataFrame], Dataset | pd.DataFrame]:
    def wrapper(dataset: Dataset | pd.DataFrame):
        for transformer in transformers:
            if transformer.can_partial_fit:
                log.info('Partially fitting transformer: %s', transformer.__class__.__qualname__)
                transformer.partial_fit(dataset)
            else:
                dataset = transformer.transform(dataset)
        return dataset
    return wrapper


def fit(
    dataset: Dataset | pd.DataFrame,
    transformers: List[Transformer],
    out_path: Optional[Path] = None
) -> Dataset | pd.DataFrame:
    log.info('Transforming dataset ...')
    time_start = time.perf_counter()

    if not hasattr(dataset, 'chunks') or dataset.chunks is None or not out_path:
        MemoryBackend().fit(dataset, transformers)
    else:
        assert out_path, '`out_path` argument is required for out-of-core processing'
        assert isinstance(dataset, Dataset), 'only `Dataset` support `FileBackend`'

        FileBackend(out_path).fit(dataset, transformers)

    log.info('... finished transformation in [%d]s.', time.perf_counter() - time_start)

    return dataset


def transform(
    dataset: Dataset | pd.DataFrame,
    transformers: List[Transformer],
    out_path: Optional[Path] = None
) -> Dataset | pd.DataFrame:
    log.info('Transforming dataset ...')
    time_start = time.perf_counter()

    if not hasattr(dataset, 'chunks') or dataset.chunks is None or not out_path:
        dataset = MemoryBackend().transform(dataset, transformers)
    else:
        assert out_path, '`out_path` argument is required for out-of-core processing'
        assert isinstance(dataset, Dataset), 'only `Dataset` support `FileBackend`'
        dataset = FileBackend(out_path).transform(dataset, transformers)

    log.info('... finished transformation in [%d]s.', time.perf_counter() - time_start)

    return dataset


def fit_transform(
    dataset: Dataset | pd.DataFrame,
    transformers: List[Transformer],
    out_path: Optional[Path] = None
) -> Dataset | pd.DataFrame:
    log.info('Transforming dataset ...')
    time_start = time.perf_counter()

    if not hasattr(dataset, 'chunks') or dataset.chunks is None or not out_path:
        dataset = MemoryBackend().fit_transform(dataset, transformers)
    else:
        assert out_path, '`out_path` argument is required for out-of-core processing'
        assert isinstance(dataset, Dataset), 'only `Dataset` support `FileBackend`'
        dataset = FileBackend(out_path).fit_transform(dataset, transformers)

    log.info('... finished transformation in [%d]s.', time.perf_counter() - time_start)

    return dataset


class MemoryBackend:
    def fit(self, dataset: Dataset | pd.DataFrame, transformers: List[Transformer]):
        for transformer in transformers:
            if transformer.can_fit:
                log.info('Fitting transformer: %s', transformer.__class__.__qualname__)
                transformer.fit(dataset)

    def transform(
        self,
        dataset: Dataset | pd.DataFrame,
        transformers: List[Transformer]
    ) -> Dataset | pd.DataFrame:
        for transformer in transformers:
            log.info('Applying transformer: %s', transformer.__class__.__qualname__)
            dataset = transformer.transform(dataset)
        return dataset

    def fit_transform(
        self,
        dataset: Dataset | pd.DataFrame,
        transformers: List[Transformer]
    ) -> Dataset | pd.DataFrame:
        for transformer in transformers:
            self.fit(dataset, [transformer])

            log.info('Applying transformer: %s', transformer.__class__.__qualname__)
            dataset = transformer.transform(dataset)

        return dataset


class FileBackend:
    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path
        out_path.unlink(missing_ok=True)

    def fit(self, dataset: Dataset, transformers: List[Transformer]):
        xr.map_blocks(
            partial_fit(transformers), dataset, template=dataset
        ).load(scheduler='single-threaded')

    def fit_transform(
        self,
        dataset: Dataset,
        transformers: List[Transformer]
    ) -> Dataset:
        tmp_path = self._out_path.parent / (self._out_path.name + '.tmp')
        if self._out_path.exists():
            tmp_path.unlink(missing_ok=True)
            self._out_path.rename(tmp_path)

            dataset = xr.open_dataset(
                tmp_path,
                engine='h5netcdf',
                chunks=cast(Frozen, dataset.chunks).mapping
            )

        self.fit(dataset, transformers)

        def apply_transformers(transformers: List[Transformer]):
            def wrap(dataset: Dataset):
                for transformer in transformers:
                    log.info('Applying transformer: %s', transformer.__class__.__qualname__)
                    dataset = transformer.transform(dataset)
                return dataset
            return wrap

        xr.map_blocks(
            apply_transformers(transformers),
            dataset,
            template=dataset
        ).to_netcdf(
            self._out_path,
            compute=False,
            engine='h5netcdf'
        ).compute()

        tmp_path.unlink(missing_ok=True)

        if self._out_path.exists():
            dataset = xr.open_dataset(
                self._out_path,
                engine='h5netcdf',
                chunks=cast(Frozen, dataset.chunks).mapping
            )

        return dataset
