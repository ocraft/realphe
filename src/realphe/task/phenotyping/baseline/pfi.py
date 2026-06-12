import os
from pathlib import Path
import random
import time
from typing import List, Tuple, cast

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import xarray as xr

from realphe.env import get_log
from realphe.pipe.metrics import SampleIdx
from realphe.pipe.splitter import get_fold_val
from realphe.task.phenotyping.setup.dataset import PhenotypingDataset

from .load import get_model, predict, preprocess
from .setup import get_train_set, load_split_for
from .solution import BaselinePhenotypingModel
from .time_series import index_of_last_step

log = get_log(__name__)


def run():
    params = OmegaConf.from_cli()
    solution_name = params.solution
    seed = params.seed
    fold = params.fold

    # random seed
    set_random_seed(seed)

    # prepare data
    signals_df, targets_df, last_step_idx = prepare_data(seed, fold, solution_name)
    results_df = init_results(signals_df, targets_df)

    # get solution
    model = get_model(seed, fold)

    log.info('[%s] Computing permutation feature importance seed=%d fold=%d ...',
             solution_name, seed, fold)
    time_start = time.perf_counter()

    score(model, signals_df, targets_df, last_step_idx, results_df, 'BASELINE')

    for fea in tqdm(signals_df.columns):
        permute_score(signals_df, targets_df, last_step_idx, results_df, model, fea)

    log.info('... finished computing PFI in [%d]s.', time.perf_counter() - time_start)

    params_phenotyping = OmegaConf.load('params.yaml').phenotyping
    out_path = Path(f'{params_phenotyping.path.eval}/{solution_name}/interpret/{seed}/{fold}')
    os.makedirs(out_path, exist_ok=True)
    results_df.to_parquet(out_path / 'pfi.parquet', engine='fastparquet')


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def prepare_data(
    seed: int,
    fold: int,
    solution_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, SampleIdx]:
    cohort_val_set, signals_val_df, targets_df = prepare_val_set(seed, fold, solution_name)
    last_step_idx = index_of_last_step(signals_val_df)
    cohort_set, signals_df = preprocess(cohort_val_set, signals_val_df, seed, fold)

    cohort_df = (
        cohort_set
        .drop_dims(['label'])
        .to_pandas()
    )

    del cohort_val_set, signals_val_df, cohort_set
    return signals_df.join(cohort_df, on='sample'), targets_df, last_step_idx


def prepare_val_set(
    seed: int,
    fold: int,
    solution_name: str
) -> Tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    params_phenotyping = OmegaConf.load('params.yaml').phenotyping

    cohort_set, signals_df = get_train_set(params_phenotyping, solution_name)

    cohort_set = load_split_for(
        params_phenotyping.path,
        solution_name,
        cohort_set,
        seed
    )
    cohort_val_set = get_fold_val(cohort_set, fold, coord=f'fold_{seed}')
    targets_df = cohort_val_set.target.to_pandas()

    folds = [x for x in cast(List[str], cohort_val_set.coords) if x.startswith('fold_')]
    cohort_val_set = cohort_val_set.drop_vars(folds)
    del cohort_set

    signals_val_df = cast(pd.DataFrame, signals_df.loc[cohort_val_set.sample.values, :])
    del signals_df

    return cohort_val_set, signals_val_df, targets_df


def init_results(
    signals_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> pd.DataFrame:
    results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            (['BASELINE'] + list(signals_df.columns), targets_df.columns)
        ),
        columns=['ls_roc_auc', 'ls_pr_auc'],
        dtype=np.float32
    )
    results.index.names = ['feature', 'label']
    return results


def score(
    model: BaselinePhenotypingModel,
    signals_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    last_step_idx: SampleIdx,
    results_df: pd.DataFrame,
    fea: str
):
    data_gen = PhenotypingDataset(signals_df, targets_df, shuffle=False)

    Y_true = targets_df.values
    Y_pred = predict(data_gen, model)[last_step_idx, :]
    idx = pd.IndexSlice
    results_df.loc[idx[fea, :], 'ls_roc_auc'] = (
        np.array(roc_auc_score(Y_true, Y_pred, average=None), dtype=np.float32)
    )
    results_df.loc[idx[fea, :], 'ls_pr_auc'] = (
        np.array(average_precision_score(Y_true, Y_pred, average=None), dtype=np.float32)
    )


def permute_score(
    signals_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    last_step_idx: SampleIdx,
    results_df: pd.DataFrame,
    model: BaselinePhenotypingModel,
    fea: str
):
    save_col = permute(signals_df, fea)
    score(model, signals_df, targets_df, last_step_idx, results_df, fea)
    signals_df[fea] = save_col


def permute(signals_df: pd.DataFrame, fea: str) -> pd.Series:
    save_col = signals_df[fea].copy()
    col_to_permute = signals_df[fea].to_numpy()
    np.random.shuffle(col_to_permute)
    signals_df[fea] = col_to_permute
    return save_col


if __name__ == '__main__':
    run()
