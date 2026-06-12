from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import xarray as xr

from realphe.pipe.transformer.core import Transformer, transform
from realphe.task.phenotyping.setup.dataset import (
    PhenotypingDataset,
    packed_sample_collate_fn,
)

from .solution import (
    BaselinePhenotypingModel,
    get_dataset,
    set_random_seed,
    transformers,
)


def pipeline_single(
    cohort_set: xr.Dataset,
    signals_df: pd.DataFrame,
    seed: int,
    fold: int,
    device: torch.device = torch.device('cuda'),
    batch_size: int = 512,
) -> np.ndarray:
    set_random_seed(seed)

    return pipeline_single_predict(
        *pipeline_single_load(seed, fold),
        cohort_set,
        signals_df,
        device,
        batch_size
    )


def pipeline_single_load(
    seed: int,
    fold: int
) -> Tuple[BaselinePhenotypingModel, List[Transformer], List[Transformer]]:
    params = OmegaConf.load('params.yaml').phenotyping

    out_dir = Path(f'model/phenotyping/baseline/{seed}/{fold}')

    cohort_transformers, signals_transformers = transformers(params.baseline,
                                                             fitted=True,
                                                             out_dir=out_dir)

    model = get_model(seed, fold)

    return model, cohort_transformers, signals_transformers


def pipeline_single_predict(
    model: BaselinePhenotypingModel,
    cohort_transformers: List[Transformer],
    signals_transformers: List[Transformer],
    cohort_set: xr.Dataset,
    signals_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:

    cohort_set = cast(xr.Dataset, transform(cohort_set, cohort_transformers))
    signals_df = cast(pd.DataFrame, transform(signals_df, signals_transformers))
    data_gen = get_dataset(cohort_set, signals_df, shuffle=False)
    return predict(data_gen, model, device, batch_size)


def predict(
    data_gen: PhenotypingDataset,
    model: BaselinePhenotypingModel,
    device: torch.device = torch.device('cuda'),
    batch_size: int = 512,
) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        data_gen,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=packed_sample_collate_fn
    )

    model.to(device)
    model.eval()
    preds = []
    with torch.inference_mode():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device, non_blocking=True)

            preds.append(model(X_batch))

    return torch.cat(packed_predictions_to_list(preds)).detach().cpu().numpy()


def preprocess(
    cohort_set: xr.Dataset,
    signals_df: pd.DataFrame,
    seed: int,
    fold: int
) -> Tuple[xr.Dataset, pd.DataFrame]:
    params = OmegaConf.load('params.yaml').phenotyping

    # random seed
    set_random_seed(seed)

    out_dir = Path(f'model/phenotyping/baseline/{seed}/{fold}')

    cohort_transformers, signals_transformers = transformers(params.baseline,
                                                             fitted=True,
                                                             out_dir=out_dir)

    cohort_set = cast(xr.Dataset, transform(cohort_set, cohort_transformers))
    signals_df = cast(pd.DataFrame, transform(signals_df, signals_transformers))

    return cohort_set, signals_df


def get_model(seed: int, fold: int):
    out_dir = Path(f'model/phenotyping/baseline/{seed}/{fold}')
    return get_model_from_path(out_dir / 'model.pt')


def get_model_from_path(out_path: Path) -> BaselinePhenotypingModel:
    model = torch.load(out_path, weights_only=False)
    if model is None:
        raise ValueError(f'Cannot load model from [{out_path}].')
    model.eval()
    return model


def packed_predictions_to_list(packed_batches: List[PackedSequence]) -> List[torch.Tensor]:
    all_predictions = []

    for packed in packed_batches:
        padded, lengths = pad_packed_sequence(packed, batch_first=True)
        padded = torch.sigmoid(padded)

        # split into per-sample tensors
        for seq, length in zip(padded, lengths):
            all_predictions.append(
                seq[:length]
            )

    return all_predictions
