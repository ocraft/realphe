from typing import Dict, Sequence

import numpy as np
import pandas as pd
from xarray import Dataset

from .core import Transformer


class Drop(Transformer):
    def __init__(self, var: Sequence[str]):
        self._var = var

    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        if isinstance(dataset, Dataset):
            return dataset.drop_vars(self._var)
        if isinstance(dataset, pd.DataFrame):
            return dataset.drop(self._var, axis=1)

        raise NotImplementedError(f'Drop for type {type(dataset)} is not implemented.')


class DType(Transformer):
    def __init__(self, dtypes: Dict[str, str | np.dtype] | np.dtype):
        self._dtypes = dtypes

    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.astype(self._dtypes, copy=False)

        if isinstance(dataset, Dataset):
            if isinstance(self._dtypes, Dict):
                for var, dtype in self._dtypes.items():
                    dataset[var] = dataset[var].astype(
                        dtype, casting='unsafe', copy=False, keep_attrs=True)
            if isinstance(self._dtypes, np.dtype):
                dataset = dataset.astype(self._dtypes, casting='unsafe',
                                         copy=False, keep_attrs=True)

        return dataset
