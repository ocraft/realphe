from typing import List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as StdScaler
from typing_extensions import Self
from xarray import Dataset

from .core import Transformer


class StandardScaler(Transformer):

    def __init__(self, var: Optional[List[str]] = None):
        self._var = var
        self._scaler = StdScaler(copy=False)

    def partial_fit(self, dataset: Dataset | pd.DataFrame) -> Self:
        ds = self._select(dataset)
        self._scaler.partial_fit(ds)
        return self

    def _select(self, dataset: Dataset | pd.DataFrame) -> pd.DataFrame:
        if isinstance(dataset, Dataset):
            keys = self._var if self._var is not None else dataset.keys()
            return cast(pd.DataFrame, dataset[keys].to_pandas())[keys]
        if isinstance(dataset, pd.DataFrame):
            return (dataset if self._var is None else dataset[self._var])

        raise NotImplementedError(f'StandardScaler for type {type(dataset)} is not implemented.')

    def fit(self, dataset: Dataset | pd.DataFrame) -> Self:
        ds = self._select(dataset)
        self._scaler.fit(ds.to_numpy(dtype=np.float32, copy=False))
        return self

    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        if self._scaler is None:
            raise ValueError('Transformer is not fitted')
        ds = self._select(dataset)

        if isinstance(dataset, Dataset):
            dataset[self._var if self._var is not None else dataset.keys()] = (
                pd.DataFrame(self._scaler.transform(ds.to_numpy(dtype=np.float32, copy=False)),
                             index=ds.index,
                             columns=ds.columns)
                .to_xarray()
            )
            return dataset
        if isinstance(dataset, pd.DataFrame):
            dataset[self._var if self._var is not None else dataset.columns] = (
                pd.DataFrame(self._scaler.transform(ds.to_numpy(dtype=np.float32, copy=False)),
                             index=ds.index,
                             columns=ds.columns)
            )
            return dataset

        raise NotImplementedError(f'StandardScaler for type {type(dataset)} is not implemented.')
