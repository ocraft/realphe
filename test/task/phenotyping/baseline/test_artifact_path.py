from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from realphe.task.phenotyping.baseline import solution


class FakeLive:
    instances = []

    def __init__(self, *args, **kwargs):
        self.artifacts = []
        FakeLive.instances.append(self)

    def log_artifact(self, path, **kwargs):
        self.artifacts.append(Path(path))


def test_training_logs_the_model_file_that_is_saved(monkeypatch, tmp_path):
    """Regression test for the PyTorch model artifact filename."""
    params = OmegaConf.create({
        'phenotyping': {
            'path': {'eval': str(tmp_path / 'eval')},
            'baseline': {'hparams': {'batch_size': 2, 'epochs': 1}},
        }
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(solution.OmegaConf, 'load', lambda _: params)
    monkeypatch.setattr(solution, 'transformers', lambda *args, **kwargs: ([], []))

    gen = SimpleNamespace(
        n_features=3,
        n_targets=2,
        y_values=np.array([[1, 0], [0, 1]], dtype=np.int8),
        element_spec=((2, 2, 3), (2, 2)),
    )
    monkeypatch.setattr(solution, 'raw_to_dataset', lambda *args, **kwargs: (gen, gen))
    monkeypatch.setattr(solution, 'get_loaders', lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(solution, 'create_model', lambda *args, **kwargs: torch.nn.Linear(3, 2))
    monkeypatch.setattr(solution, 'compile_model', lambda *args, **kwargs: {})
    monkeypatch.setattr(solution, 'fit', lambda model, *args, **kwargs: (model, {}))
    monkeypatch.setattr(solution, 'Live', FakeLive)

    saved_paths = []
    monkeypatch.setattr(solution.torch, 'save', lambda model, path: saved_paths.append(Path(path)))

    cohort = object()
    signals = object()
    solution.train(100, 0, cohort, signals, cohort, signals)

    expected = Path('model/phenotyping/baseline/100/0/model.pt')
    assert saved_paths == [expected]
    assert FakeLive.instances[-1].artifacts[-1] == expected
