import numpy as np
from xarray import DataArray, Dataset

from realphe.pipe.transformer import BFill, FFill
from realphe.pipe.transformer.imputer import FillNa


def test_ffills_data():
    arr = np.arange(9, dtype='float').reshape(3, 3, 1)
    arr[1, 1, 0] = np.nan

    dataset = Dataset({
        'X': DataArray(arr, dims=['sample', 'step', 'feature'], coords={'feature': ['f']})
    })

    ffill = FFill(['f'])
    dataset = ffill.transform(dataset)

    assert np.array_equal(dataset.X.values, np.array(
        [[[0], [1], [2]],
         [[3], [3], [5]],
         [[6], [7], [8]]]
    ))


def test_bfills_data():
    arr = np.arange(9, dtype='float').reshape(3, 3, 1)
    arr[1, 0, 0] = np.nan

    dataset = Dataset({
        'X': DataArray(arr, dims=['sample', 'step', 'feature'], coords={'feature': ['f']})
    })

    ffill = BFill(['f'])
    dataset = ffill.transform(dataset)

    assert np.array_equal(dataset.X.values, np.array(
        [[[0], [1], [2]],
         [[4], [4], [5]],
         [[6], [7], [8]]]
    ))


def test_fills_na_data():
    arr = np.arange(9, dtype='float').reshape(3, 3, 1)
    arr[1, 1, 0] = np.nan
    arr[1, 0, 0] = np.nan

    dataset = Dataset({
        'X': DataArray(arr, dims=['sample', 'step', 'feature'], coords={'feature': ['f']})
    })

    fillna = FillNa(['f'])
    dataset = fillna.transform(dataset)

    assert np.array_equal(dataset.X.values, np.array(
        [[[0], [1], [2]],
         [[0], [0], [5]],
         [[6], [7], [8]]]
    ))
