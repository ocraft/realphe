import importlib
import os
import random
from types import ModuleType
from typing import Callable, List, Tuple, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import xarray as xr

from realphe.pipe.splitter import get_fold

from .setup import get_train_set, load_split_for

# def {name}_solution(
#     seed: int,
#     fold: int,
#     params_eval: DictConfig,
#     cohort_train_set: Dataset,
#     signals_train_df: pd.DataFrame,
#     cohort_val_set: Dataset,
#     signals_val_df: pd.DataFrame
# ):
SolutionTrain = Callable[[int, int, xr.Dataset, pd.DataFrame, xr.Dataset, pd.DataFrame], None]


def run():
    params = OmegaConf.from_cli()
    solution_name = params.solution
    solution = load_solution(solution_name)

    train(cast(SolutionTrain, solution.train), solution_name, seed=params.seed, fold=params.fold)


def load_solution(solution_name: str) -> ModuleType:
    return importlib.import_module('.solution', package=f'realphe.task.phenotyping.{solution_name}')


def train(solution_train: SolutionTrain, solution_name: str, seed: int, fold: int):
    # load params
    params = load_params()

    # random seed
    set_random_seed(seed)

    # prepare data
    cohort_set, signals_df = get_train_set(params, solution_name)
    cohort_train_set, cohort_val_set = prepare_fold(params, solution_name, cohort_set, seed, fold)
    del cohort_set

    # train
    solution_train(
        seed,
        fold,
        cohort_train_set,
        cast(pd.DataFrame, signals_df.loc[cohort_train_set.sample.values, :]),
        cohort_val_set,
        cast(pd.DataFrame, signals_df.loc[cohort_val_set.sample.values, :])
    )


def load_params() -> DictConfig:
    return OmegaConf.load('params.yaml').phenotyping


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def prepare_fold(
    params_phenotyping: DictConfig,
    solution_name: str,
    cohort_set: xr.Dataset,
    seed: int,
    fold: int
) -> Tuple[xr.Dataset, xr.Dataset]:
    cohort_set = load_split_for(params_phenotyping.path, solution_name, cohort_set, seed)
    cohort_train_set, cohort_val_set = get_fold(cohort_set, fold, coord=f'fold_{seed}')
    folds = [x for x in cast(List[str], cohort_train_set.coords) if x.startswith('fold_')]
    cohort_train_set = cohort_train_set.drop_vars(folds)
    cohort_val_set = cohort_val_set.drop_vars(folds)
    return cohort_train_set, cohort_val_set


if __name__ == '__main__':
    run()
