from typing import List, cast

import numpy as np
import pandas as pd
from xarray import Dataset

from .core import Transformer


class OneHot(Transformer):

    def __init__(self, var: List[str]):
        self._var = var
        self._columns = None

    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        columns = []
        if isinstance(dataset, Dataset):
            for var in self._var:
                df = dataset[var].to_pandas()

                # The 'drop_first' is set to 'True' because it's important to have one
                # fewer features than categories to remove a linear dependency.
                dummies = pd.get_dummies(cast(pd.Series, df.astype('category')),
                                         prefix=var,
                                         drop_first=True,
                                         dtype='int8')
                dataset = dataset.merge(dummies.to_xarray()).drop_vars(var)
                columns.extend(list(dummies.columns))
        elif isinstance(dataset, pd.DataFrame):
            df = dataset[self._var]
            # The 'drop_first' is set to 'True' because it's important to have one
            # fewer features than categories to remove a linear dependency.
            dummies = pd.get_dummies(cast(pd.DataFrame, df.astype('category')),
                                     prefix=self._var,
                                     drop_first=True,
                                     dtype='int8')
            dataset = pd.concat([dataset, dummies], axis=1)
            dataset.drop(self._var, axis=1, inplace=True)
            columns.extend(list(dummies.columns))
        else:
            raise NotImplementedError(f'OneHot for type {type(dataset)} is not implemented.')

        # make up for the differences in existing categories
        return self._make_equal(columns, dataset)

    def _make_equal(self, columns, dataset) -> Dataset | pd.DataFrame:
        if not self._columns:
            # use this transform also as `fit`
            self._columns = columns
        else:
            feas = list(dataset.columns if isinstance(dataset, pd.DataFrame) else dataset.keys())

            columns_diff_0 = list(set(self._columns) - set(feas))
            columns_diff_1 = list(set(columns) - set(self._columns))

            if isinstance(dataset, Dataset):
                for col in columns_diff_0:
                    dataset[col] = (dataset[feas[0]].dims,
                                    np.zeros(shape=dataset[feas[0]].shape[0], dtype=np.int8))
                dataset = dataset.drop_vars(columns_diff_1)
            elif isinstance(dataset, pd.DataFrame):
                for col in columns_diff_0:
                    dataset = dataset.assign(
                        **{col: np.zeros(shape=dataset[feas[0]].shape[0], dtype=np.int8)}
                    )
                dataset.drop(columns_diff_1, axis=1, inplace=True)
        return dataset
