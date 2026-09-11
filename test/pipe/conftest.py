import numpy as np
import pytest
from xarray import DataArray, Dataset


@pytest.fixture
def dataset() -> Dataset:
    return Dataset({
        'X': DataArray(np.arange(60, dtype='float').reshape(20, 3, 1),
                       dims=['sample', 'step', 'feature'],
                       coords={'feature': ['f']}),
        'Y': DataArray(np.arange(20, dtype='int8').reshape(-1, 1), dims=['sample', 'target'])
    })
