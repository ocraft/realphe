from typing import List, Optional

import numpy as np
import pandas as pd
from typing_extensions import Self
from xarray import Dataset

from realphe.pipe.transformer.core import Transformer


class SignalImputer(Transformer):
    def __init__(self,
                 var: Optional[List[str]] = None,
                 var_binary: Optional[List[str]] = None) -> None:
        self._means = None
        self._var = var
        self._var_binary = var_binary

    def fit(self, dataset: Dataset | pd.DataFrame) -> Self:
        # prepare dataframes
        if isinstance(dataset, Dataset):
            signals_df = (
                dataset
                .to_pandas()
                .reset_index()
                .set_index(['sample', 'step'])
            )
            if self._var is None:
                self._var = [i for i in dataset.keys() if i != 'step']
        elif isinstance(dataset, pd.DataFrame):
            signals_df = dataset
            if self._var is None:
                self._var = dataset.columns

        if isinstance(dataset, (Dataset, pd.DataFrame)):
            self._means = signals_df[self._var].mean(skipna=True)
        return self

    def transform(self, dataset: Dataset | pd.DataFrame) -> Dataset | pd.DataFrame:
        # prepare dataframes
        if isinstance(dataset, Dataset):
            signals_df = (
                dataset
                .to_pandas()
                .reset_index()
                .set_index(['sample', 'step'])
            )
        elif isinstance(dataset, pd.DataFrame):
            signals_df = dataset

        if isinstance(dataset, Dataset):
            if self._var_binary:
                signals_df[self._var_binary].replace({-1: 0}, inplace=True)
            dataset[self._var] = (signals_df[self._var]
                                  .groupby('sample')
                                  .ffill()
                                  .fillna(self._means)
                                  .reset_index()
                                  .set_index(['sample'])
                                  .to_xarray()[self._var]
                                  )
        if isinstance(dataset, pd.DataFrame):
            if self._var_binary:
                dataset[self._var_binary] = signals_df[self._var_binary].replace({-1: 0})
            dataset[self._var] = (signals_df[self._var]
                                  .groupby('sample')
                                  .ffill()
                                  .fillna(self._means)
                                  .reset_index()
                                  .set_index(['sample', 'step'])
                                  )

        return dataset


class CohortImputer(Transformer):
    def __init__(self) -> None:
        self._mean = {}

    def fit(self, dataset: Dataset) -> Self:
        self._mean = (dataset[['gender', 'height']]
                      .to_pandas()
                      .groupby('gender')[['height']]
                      .mean()
                      .to_dict()
                      )
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if 'height' in self._mean:
            dataset['n_height'] = dataset['height'].isnull().astype('int8')
            dataset['height'] = (
                dataset[['gender', 'height']]
                .to_pandas()
                .groupby('gender')[['height']]
                .apply(lambda x: x.fillna({'height': self._mean['height'][x.name]}))
                .reset_index()
                .set_index(['sample'])['height']
            )

        return dataset
