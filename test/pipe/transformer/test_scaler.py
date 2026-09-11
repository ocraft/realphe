import numpy as np

from realphe.pipe.transformer.scaler import StandardScaler


def test_scales_using_standard_scaler(dataset):
    scaler = StandardScaler().fit(dataset)
    dataset = scaler.transform(dataset)

    assert np.allclose(dataset.a.values, np.array([
        -1.22474492,
        0.0,
        1.22474492,
    ]))
    assert np.allclose(dataset.b.values, np.array([
        -1.22474492,
        0.0,
        1.22474492,
    ]))
