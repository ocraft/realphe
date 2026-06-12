from xarray import Dataset

from .core import Transformer


class FFill(Transformer):
    def __init__(self, coords, dim='feature', along='step', var='X') -> None:
        self.coords = coords
        self.dim = dim
        self.along = along
        self.var = var

    def transform(self, dataset: Dataset) -> Dataset:
        selector = {}
        selector[self.dim] = self.coords

        dataset[self.var].loc[selector] = dataset[self.var].sel(selector).ffill(dim=self.along)

        return dataset


class BFill(Transformer):
    def __init__(self, coords, dim='feature', along='step', var='X') -> None:
        self.coords = coords
        self.dim = dim
        self.along = along
        self.var = var

    def transform(self, dataset: Dataset) -> Dataset:
        selector = {}
        selector[self.dim] = self.coords

        dataset[self.var].loc[selector] = dataset[self.var].sel(selector).bfill(dim=self.along)

        return dataset


class FillNa(Transformer):
    def __init__(self, coords, dim='feature', var='X', value=0) -> None:
        self.coords = coords
        self.dim = dim
        self.var = var
        self.value = value

    def transform(self, dataset: Dataset) -> Dataset:
        selector = {}
        selector[self.dim] = self.coords

        dataset[self.var].loc[selector] = dataset[self.var].sel(selector).fillna(self.value)

        return dataset
