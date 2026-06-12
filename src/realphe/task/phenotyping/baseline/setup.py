import os
import time
from typing import List, Tuple, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm
import xarray as xr

from realphe.env import get_log
from realphe.pipe.splitter import MultilabelStratifiedKFoldSplitter, split


log = get_log(__name__)


def run():
    # load params
    params = OmegaConf.load('params.yaml')
    data_path_out = params.phenotyping.path.data.processed
    solution_name = OmegaConf.from_cli().solution
    os.makedirs(f'{data_path_out}/{solution_name}/splits', exist_ok=True)
    params_solution = params.phenotyping[solution_name]

    # load cohort
    cohort = (
        xr.open_dataset(
            f'{data_path_out}/setup/cohort.zarr',
            drop_variables=['icd9', 'icd9_name'],
            consolidated=False
        )
        .load()
    )

    # split dataset into folds
    n_folds = len(params_solution.evaluate.cv.folds)
    train_samples = cohort.sel(sample=cohort.split == 0).sample

    for seed in tqdm(params_solution.evaluate.cv.seeds):
        coord = f'fold_{seed}'
        cohort.coords[coord] = ('sample', np.full_like(cohort['sample'], -1, dtype='int8'))
        splitter = MultilabelStratifiedKFoldSplitter(
            n_splits=n_folds,
            shuffle=params_solution.evaluate.cv.shuffle,
            random_state=seed,
            var='target'
        )
        np.save(
            f'{data_path_out}/{solution_name}/splits/split_{seed}.npy',
            split(cohort.loc[{'sample': train_samples}], splitter, coord=coord)[coord].values
        )
        cohort = cohort.drop_vars([coord])

    (
        cohort
        .set_coords('split')
        [list(params_solution.cohort) + ['target', 'split']]
        .to_zarr(f'{data_path_out}/{solution_name}/cohort.zarr', consolidated=False)
    )
    del cohort

    # load signals
    (
        xr.open_mfdataset(
            [
                f'{data_path_out}/setup/signals.zarr',
                f'{data_path_out}/setup/signals_event_count.zarr',
            ],
            compat='override',
            consolidated=False)
        [
            list(params_solution.signals) +
            ['n_' + signal for signal in params_solution.signals] +
            ['step']
        ]
        .to_zarr(f'{data_path_out}/{solution_name}/signals.zarr', consolidated=False)
    )


def get_train_set(
    params: DictConfig,
    solution_name: str
) -> Tuple[xr.Dataset, pd.DataFrame]:
    log.info('Loading training set ...')
    time_start = time.perf_counter()

    params_path = params.path
    params_solution = params[solution_name]
    cohort_path, signals_path = data_path_for_solution(params_path, solution_name)

    cohort_set = (
        xr.open_dataset(f'{cohort_path}/cohort.zarr', consolidated=False)
        .drop_vars(['icd9', 'icd9_name'], errors='ignore')
        [list(params_solution.cohort) + ['target', 'split']]
        .load()
        .set_coords('split')
    )

    test_samples = cohort_set.sel(sample=cohort_set.split == 1).sample.values
    cohort_set = (
        cohort_set
        .sel(sample=cohort_set.split == 0)
        .drop_vars('split')
    )

    signals_df = (
        xr.open_mfdataset(f'{signals_path}/signals.zarr', consolidated=False)
        [
            list(params_solution.signals) +
            ['n_' + signal for signal in params_solution.signals] +
            ['step']
        ]
        .to_pandas()
        .reset_index()
    )
    signals_df.set_index(['sample', 'step'], inplace=True)
    signals_df.drop(index=test_samples, inplace=True)

    log.info('... finished loading training set (samples=[%d]) in [%d]s.',
             len(cohort_set.sample),
             time.perf_counter() - time_start)

    return cohort_set, signals_df


def data_path_for_solution(params_path, solution_name):
    cohort_path = (
        f'{params_path.data.processed}/{solution_name}'
        if os.path.exists(params_path.data.processed + f'/{solution_name}/cohort.zarr')
        else params_path.data.processed
    )
    signals_path = (
        f'{params_path.data.processed}/{solution_name}'
        if os.path.exists(params_path.data.processed + f'/{solution_name}/signals.zarr')
        else params_path.data.processed
    )
    return cohort_path, signals_path


def get_eval_set(
    params: DictConfig,
    solution_name: str
) -> Tuple[xr.Dataset, pd.DataFrame, xr.Dataset, pd.DataFrame]:
    log.info('Loading evaluation sets ...')
    time_start = time.perf_counter()

    params_path = params.path
    params_solution = params[solution_name]
    cohort_path, signals_path = data_path_for_solution(params_path, solution_name)

    cohort_set = (
        xr.open_dataset(f'{cohort_path}/cohort.zarr', consolidated=False)
        .drop_vars(['icd9', 'icd9_name'], errors='ignore')
        [list(params_solution.cohort) + ['target', 'split']]
        .load()
        .set_coords('split')
    )

    train_samples = cohort_set.sel(sample=cohort_set.split == 0).sample.values
    test_samples = cohort_set.sel(sample=cohort_set.split == 1).sample.values

    signals_df = (
        xr.open_mfdataset(f'{signals_path}/signals.zarr', consolidated=False)
        [
            list(params_solution.signals) +
            ['n_' + signal for signal in params_solution.signals] +
            ['step']
        ]
        .to_pandas()
        .reset_index()
    )
    signals_df.set_index(['sample', 'step'], inplace=True)

    log.info('... finished loading train set (samples=[%d]) and test set (samples=[%d]) in [%d]s.',
             len(train_samples),
             len(test_samples),
             time.perf_counter() - time_start)

    folds = [x for x in cast(List[str], cohort_set.coords) if x.startswith('fold_')]
    return (
        cohort_set.sel(sample=cohort_set.split == 0).drop_vars('split'),
        cast(pd.DataFrame, signals_df.loc[train_samples, :]),
        cohort_set.sel(sample=cohort_set.split == 1).drop_vars(['split'] + folds),
        cast(pd.DataFrame, signals_df.loc[test_samples, :])
    )


def load_split_for(
    params_path: DictConfig,
    solution_name: str,
    cohort: xr.Dataset,
    seed: int
) -> xr.Dataset:
    coord = f'fold_{seed}'
    cohort.coords[coord] = ('sample', np.full_like(cohort['sample'], -1, dtype='int8'))
    cohort.coords[coord].loc[:] = np.load(
        f'{params_path.data.processed}/{solution_name}/splits/split_{seed}.npy'
    )
    return cohort


if __name__ == '__main__':
    run()
