import numpy as np
import pandas as pd

from realphe.task.phenotyping.setup.transformer import SignalImputer


def test_imputes_signals():
    signals = np.array([
        ['101', 1, 1, 1, np.nan, np.nan],
        ['101', 2, 2, np.nan, 3, np.nan],
        ['102', 1, np.nan, np.nan, np.nan, np.nan],
        ['102', 2, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)

    dataset = (pd.DataFrame(signals, columns=['sample', 'step', 'a', 'b', 'c', 'd'])
               .set_index(['sample'])
               .to_xarray()
               )

    imputer = SignalImputer()
    imputer.fit(dataset)
    dataset = imputer.transform(dataset)

    expected = np.array([
        [[1.0, 1.0, 3.0, np.nan],
         [2.0, 1.0, 3.0, np.nan]],
        [[1.5, 1.0, 3.0, np.nan],
         [1.5, 1.0, 3.0, np.nan]],
    ], dtype=np.float32)

    assert np.array_equal(
        (dataset
             .to_pandas()
             .reset_index()
             .set_index(['sample', 'step'])
             .values
             .reshape(2, 2, -1)
         ),
        expected, equal_nan=True)
