from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from realphe.pipe.metrics import SampleIdx
from realphe.pipe.shared import SharedArray


def index_of_first_step(df: pd.DataFrame) -> SampleIdx:
    return np.unique(df.index.get_level_values('sample').values, True)[1]


def index_of_last_step(df: pd.DataFrame) -> SampleIdx:
    return np.append(
        np.unique(df.index.get_level_values('sample').values, True)[1][1:]-1,
        len(df)-1
    )


def time_series_idx(
    cohort_test_set: xr.Dataset,
    signals_test_df: pd.DataFrame
) -> Tuple[SharedArray, SampleIdx, SampleIdx]:
    test_first_step_idx = index_of_first_step(signals_test_df)
    test_last_step_idx = index_of_last_step(signals_test_df)
    test_y_true = td_y_true(cohort_test_set.target.values, test_last_step_idx)

    return test_y_true, test_first_step_idx, test_last_step_idx


def td_y_true(y_true: np.ndarray, last_step_idx: SampleIdx) -> SharedArray:
    if last_step_idx is not None:
        return SharedArray(
            np.repeat(y_true.astype(np.int8), np.diff(last_step_idx+1, 1, prepend=0), axis=0)
        )
    return SharedArray(y_true)
