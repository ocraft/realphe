from realphe.pipe.transformer.shape import DType, Drop


def test_drops_features(dataset):
    drop_transformer = Drop(['a']).fit(dataset)

    assert 'a' in dataset.keys()
    dataset = drop_transformer.transform(dataset)
    assert 'a' not in dataset.keys()


def test_converts_dtypes(dataset2d):
    dtype_conv = DType({'X': 'int16', 'Y': 'float32'})

    assert dataset2d.X.dtype == 'float32'
    assert dataset2d.Y.dtype == 'int8'

    dataset2d = dtype_conv.transform(dataset2d)

    assert dataset2d.X.dtype == 'int16'
    assert dataset2d.Y.dtype == 'float32'
