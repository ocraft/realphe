from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from realphe.task.phenotyping.baseline.load import get_model_from_path
from realphe.task.phenotyping.baseline.solution import create_model
from realphe.task.phenotyping.setup.dataset import PhenotypingDataset, packed_sample_collate_fn


@pytest.mark.smoke
def test_synthetic_phenotyping_model_roundtrip(tmp_path: Path):
    """Exercise variable-length inputs, inference, and model persistence without MIMIC-IV."""
    rng = np.random.default_rng(7)
    lengths = [3, 2, 4]
    n_features = 3
    n_targets = 2

    values = []
    sample_ids = []
    step_ids = []
    for sample, length in enumerate(lengths):
        values.append(rng.normal(size=(length, n_features)).astype(np.float32))
        sample_ids.extend([sample] * length)
        step_ids.extend(range(length))

    signals = pd.DataFrame(
        np.concatenate(values),
        index=pd.MultiIndex.from_arrays(
            [sample_ids, step_ids], names=['sample', 'step']
        ),
        columns=[f'f{i}' for i in range(n_features)],
    )
    targets = pd.DataFrame(
        np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8),
        index=pd.Index(range(len(lengths)), name='sample'),
        columns=['y0', 'y1'],
    )

    dataset = PhenotypingDataset(signals, targets, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=packed_sample_collate_fn,
    )
    packed_x, y = next(iter(loader))

    hparams = OmegaConf.create({
        'model': {
            'lstm_units': 8,
            'dropout': 0.0,
        }
    })
    model = create_model(hparams, n_features, n_targets, targets.to_numpy())
    model.eval()

    with torch.inference_mode():
        pred_before = model(packed_x)

    assert pred_before.data.shape == (sum(lengths), n_targets)
    assert torch.isfinite(pred_before.data).all()
    assert y.shape == (len(lengths), n_targets)

    model_path = tmp_path / 'model.pt'
    torch.save(model, model_path)
    restored = get_model_from_path(model_path)

    with torch.inference_mode():
        pred_after = restored(packed_x)

    assert torch.equal(pred_before.batch_sizes, pred_after.batch_sizes)
    assert torch.allclose(pred_before.data, pred_after.data)
