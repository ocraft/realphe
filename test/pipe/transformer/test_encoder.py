from realphe.pipe.transformer.encoder import OneHot


def test_onehot_encodes_features(dataset):
    onehot = OneHot(['a']).fit(dataset)
    dataset = onehot.transform(dataset)

    assert all(fea in dataset.keys()
               for fea in ['a_2.0', 'a_3.0'])
    assert 'a' not in dataset.keys()
