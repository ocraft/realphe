import numpy as np
from omegaconf import OmegaConf
import torch

from realphe.task.phenotyping.baseline.load import get_model_from_path
from realphe.task.phenotyping.baseline.solution import create_model


def test_model_serialization(tmp_path):
    # given
    params = OmegaConf.create({
        "model": {
            "lstm_units": 8,
            "dropout": 0.3,
        }
    })
    model = create_model(
        params,
        3,
        2,
        np.random.randint(0, 2, size=(5, 2))
    )
    path = tmp_path / 'tmp.pt'

    # when
    torch.save(model, path)

    # then
    get_model_from_path(path)
