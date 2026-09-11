import string

import numpy as np
import pytest
from xarray import DataArray, Dataset


@pytest.fixture
def dataset() -> Dataset:
    return Dataset({
        'a': DataArray(np.array([1, 2, 3], dtype=np.float32), dims=['sample']),
        'b': DataArray(np.array([7, 8, 9], dtype=np.float32), dims=['sample'])
    })


@pytest.fixture
def dataset2d() -> Dataset:
    return Dataset({
        'X': DataArray(np.arange(18, dtype='float32').reshape(6, 3),
                       dims=['sample', 'feature'],
                       coords={'feature': list(string.ascii_lowercase[0:3])}),
        'Y': DataArray(np.arange(6, dtype='int8').reshape(-1, 1), dims=['sample', 'target'])
    })


@pytest.fixture
def dataset3d() -> Dataset:
    return Dataset({
        'X': DataArray(np.arange(9, dtype='float').reshape(3, 3, 1),
                       dims=['sample', 'step', 'feature'],
                       coords={'feature': ['f']})
    })
