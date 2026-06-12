import gc
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from dvclive.live import Live
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
import tqdm
import xarray as xr

from realphe.engine import Callback
from realphe.engine.callbacks import DVCLive, EarlyStopping, ReduceLROnPlateau
from realphe.engine.layers import GaussianNoise
from realphe.engine.trainer import fit
from realphe.engine.utils import mc_dropout_predict
from realphe.env import get_log
from realphe.pipe.monitor import Monitor, Result
from realphe.pipe.nn import focal_loss_bias_init

log = get_log(__name__)


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


class ICD92CCS(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        units: int,
        dropout_h1: float,
        dropout_h2: float,
        noise_v: np.ndarray,
        gauss_noise: float,
        out_bias: np.ndarray,
    ):
        super().__init__()

        gaussian_noise = GaussianNoise(
            noise_v * gauss_noise
        )

        # Hidden layer 1
        fc1 = nn.Linear(n_features, units)
        act1 = nn.LeakyReLU(negative_slope=0.3)
        dropout1 = nn.Dropout(dropout_h1)

        # Hidden layer 2
        fc2 = nn.Linear(units, units)
        act2 = nn.LeakyReLU(negative_slope=0.3)
        dropout2 = nn.Dropout(dropout_h2)

        # Output layer
        out = nn.Linear(units, n_targets)

        self._dense_init(fc1)
        self._dense_init(fc2)
        nn.init.xavier_uniform_(out.weight)

        with torch.no_grad():
            out.bias.copy_(
                torch.as_tensor(out_bias, dtype=torch.float32)
            )

        self.mlp = nn.Sequential(
            gaussian_noise,
            fc1,
            act1,
            dropout1,
            fc2,
            act2,
            dropout2,
            out,
        )

    @staticmethod
    def _dense_init(layer):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)


class SigmoidOutput(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))


def run():
    # load params
    params_cli = OmegaConf.from_cli()
    solution_name = params_cli.solution

    params = load_params()
    phe_path = params.phenotyping.path.data.processed
    out_path = f'{params.coding.path.data.processed}/ccs/'
    os.makedirs(out_path, exist_ok=True)

    # load codings
    icd9, benchmark_ccs = load_codings(params.path.data.processed)

    # load phenotypes
    oof, test, target = load_phenotypes(phe_path, solution_name)

    # prepare training data
    oof_y_pred_icd9, oof_y_true_ccs, test_y_pred_icd9, test_y_true_ccs = prepare_training_data(
        oof, test, target, icd9, benchmark_ccs)

    # encode to ccs
    test_agg_ccs_pred = encode_to_ccs(
        oof_y_pred_icd9, oof_y_true_ccs, test_y_pred_icd9, test_y_true_ccs, params)

    test_agg_ccs_pred.to_parquet(
        f'{out_path}/test_{solution_name}.parquet',
        engine='fastparquet'
    )


def load_params() -> DictConfig:
    return cast(DictConfig, OmegaConf.load('params.yaml'))


def load_codings(path: str) -> Tuple[pd.DataFrame, Dict]:
    dict_path = f'{path}/dictionary'
    with open(f'{dict_path}/ccs_bench.json', encoding='utf-8') as file:
        benchmark_ccs = json.load(file)
    icd9 = pd.read_csv(f'{dict_path}/icd9.csv').set_index('icd9')
    return icd9, benchmark_ccs


def load_phenotypes(
    path: str,
    solution_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, xr.DataArray]:
    log.info('Loading phenotypes ...')
    time_start = time.perf_counter()
    cohort = (
        xr.open_dataset(f'{path}/{solution_name}/cohort.zarr', consolidated=False)
        .load()
    )
    log.info('... finished loading cohort in [%d]s.', time.perf_counter() - time_start)

    time_start = time.perf_counter()
    oof = pd.read_parquet(f'{path}/{solution_name}/oof.parquet',
                          engine='fastparquet').reset_index().set_index(['sample', 'step'])
    test = pd.read_parquet(f'{path}/{solution_name}/test.parquet',
                           engine='fastparquet').reset_index().set_index(['sample', 'step'])
    target = cohort.target

    log.info('... finished loading phenotypes in [%d]s.', time.perf_counter() - time_start)
    return oof, test, target


def prepare_training_data(
    oof: pd.DataFrame,
    test: pd.DataFrame,
    target: xr.DataArray,
    icd9: pd.DataFrame,
    benchmark_ccs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info('Preparing training data ...')
    time_start = time.perf_counter()

    oof_y_true_icd9 = pd.DataFrame(
        np.repeat(
            target.sel(sample=cast(pd.MultiIndex, oof.index).levels[0]),
            oof.groupby(level=0).size(),
            axis=0
        ),
        index=oof.index,
        columns=oof.columns
    )
    oof_y_true_icd9.columns = pd.MultiIndex.from_tuples(
        zip(oof_y_true_icd9.columns, oof_y_true_icd9.columns.map(icd9['ccs'])))
    oof_y_true_icd9.columns.names = ['icd9', 'ccs']

    oof_y_true_ccs = as_ccs(oof_y_true_icd9, benchmark_ccs)

    oof_y_pred_icd9 = oof
    oof_y_pred_icd9.columns = pd.MultiIndex.from_tuples(
        zip(oof_y_pred_icd9.columns, oof_y_pred_icd9.columns.map(icd9['ccs'])))
    oof_y_pred_icd9.columns.names = ['icd9', 'ccs']

    test_y_true_icd9 = pd.DataFrame(
        np.repeat(
            target.sel(sample=cast(pd.MultiIndex, test.index).levels[0]),
            test.groupby(level=0).size(),
            axis=0
        ),
        index=test.index,
        columns=test.columns
    )
    test_y_true_icd9.columns = pd.MultiIndex.from_tuples(
        zip(test_y_true_icd9.columns, test_y_true_icd9.columns.map(icd9['ccs']))
    )
    test_y_true_icd9.columns.names = ['icd9', 'ccs']

    test_y_true_ccs = as_ccs(test_y_true_icd9, benchmark_ccs)

    test_y_pred_icd9 = test
    test_y_pred_icd9.columns = pd.MultiIndex.from_tuples(
        zip(test_y_pred_icd9.columns, test_y_pred_icd9.columns.map(icd9['ccs']))
    )
    test_y_pred_icd9.columns.names = ['icd9', 'ccs']

    log.info('... finished preparing training data in [%d]s.', time.perf_counter() - time_start)
    return oof_y_pred_icd9, oof_y_true_ccs, test_y_pred_icd9, test_y_true_ccs


def as_ccs(df: pd.DataFrame, benchmark_ccs: dict, group: str = 'max') -> pd.DataFrame:
    df_mapped = df.copy().T
    if group == 'max':
        df_mapped = df_mapped.groupby(level=1).max().T
    if group == 'mean':
        df_mapped = df_mapped.groupby(level=1).mean().T

    return df_mapped[map(int, benchmark_ccs.keys())]


def encode_to_ccs(
    oof_y_pred_icd9: pd.DataFrame,
    oof_y_true_ccs: pd.DataFrame,
    test_y_pred_icd9: pd.DataFrame,
    test_y_true_ccs: pd.DataFrame,
    params: DictConfig
) -> pd.DataFrame:
    torch.multiprocessing.set_sharing_strategy('file_system')

    # last step
    oof_y_pred_icd9_ls = oof_y_pred_icd9.groupby(level=0).last()
    oof_y_true_ccs_ls = oof_y_true_ccs.groupby(level=0).last()
    test_y_pred_icd9_ls = test_y_pred_icd9.groupby(level=0).last()
    test_y_true_ccs_ls = test_y_true_ccs.groupby(level=0).last()

    # train and evaluate
    monitor = Monitor(
        eval_path=f'{params.coding.path.eval}/ccs/eval',
        dvcyaml=None
    )
    monitor.log_params(params.coding.ccs)

    test_agg_ccs_pred = pd.DataFrame(
        0,
        index=test_y_true_ccs_ls.index,
        columns=test_y_true_ccs_ls.columns,
        dtype=np.float64
    )
    oof_agg_ccs_pred = pd.DataFrame(
        0,
        index=oof_y_true_ccs_ls.index,
        columns=oof_y_true_ccs_ls.columns,
        dtype=np.float32
    )

    seeds = params.coding.ccs.evaluate.cv.seeds
    n_folds = params.coding.ccs.evaluate.cv.folds
    for i_seed, seed in tqdm.tqdm(enumerate(seeds)):
        set_random_seed(seed)

        splitter = create_splitter(n_folds, params.coding.ccs.evaluate.cv.shuffle, seed)
        for fold, (train, val) in tqdm.tqdm(enumerate(splitter.split(oof_y_pred_icd9_ls,
                                                                     oof_y_pred_icd9_ls))):
            model = train_ccs_mapper(sel(oof_y_pred_icd9_ls, train),
                                     sel(oof_y_true_ccs_ls, train),
                                     sel(oof_y_pred_icd9_ls, val),
                                     sel(oof_y_true_ccs_ls, val),
                                     params.coding,
                                     seed=seed,
                                     fold=fold)

            model = SigmoidOutput(model)

            test_y_pred_ccs_ls, std_pred = mc_dropout_predict(
                model, test_y_pred_icd9_ls.values, n_iter=200, seed=142, device=DEVICE)

            test_agg_ccs_pred = test_agg_ccs_pred.add(pd.DataFrame(
                test_y_pred_ccs_ls,
                index=test_y_true_ccs_ls.index,
                columns=test_y_true_ccs_ls.columns,
                dtype=np.float64
            ))

            val_pred = (
                model(torch.from_numpy(sel(oof_y_pred_icd9_ls, val)).to(DEVICE))
                .detach().cpu().numpy()
            )
            oof_agg_ccs_pred.iloc[val, :] += val_pred

            monitor.log_cv_fold(
                seed,
                metrics(
                    sel(oof_y_true_ccs_ls, val),
                    val_pred
                ),
                metrics(
                    test_y_true_ccs_ls.values,
                    test_y_pred_ccs_ls
                )
            )
            monitor.live.make_report()

            del model, val_pred, test_y_pred_ccs_ls, std_pred
            gc.collect()

        monitor.log_cv(
            seed,
            metrics(
                oof_y_true_ccs_ls.values,
                oof_agg_ccs_pred.values / ((i_seed + 1) * n_folds)
            ),
            metrics(
                test_y_true_ccs_ls.values,
                test_agg_ccs_pred.values / ((i_seed + 1) * n_folds)
            )
        )

    test_agg_ccs_pred = test_agg_ccs_pred.div(n_folds * len(seeds))
    test_agg_ccs_pred.columns = test_agg_ccs_pred.columns.astype(str)
    oof_agg_ccs_pred = oof_agg_ccs_pred.div(n_folds * len(seeds))
    oof_agg_ccs_pred.columns = oof_agg_ccs_pred.columns.astype(str)

    monitor.log_learning_curve(
        ['aucpr', 'aucroc', 'loss'],
        title='Coding::CCS::Learning Curve'
    )
    monitor.end(
        metrics(
            oof_y_true_ccs_ls.values,
            oof_agg_ccs_pred.values
        ),
        metrics(
            test_y_true_ccs_ls.values,
            test_agg_ccs_pred.values
        )
    )
    return test_agg_ccs_pred


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def create_splitter(n_folds: int, shuffle: bool, seed: int) -> MultilabelStratifiedKFold:
    return MultilabelStratifiedKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=seed
    )


def sel(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    if isinstance(df.index, pd.MultiIndex):
        return df.loc[pd.IndexSlice[df.index.levels[0][idx], :]].values
    return df.iloc[idx, :].values


def train_ccs_mapper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: DictConfig,
    seed: int,
    fold: int
) -> nn.Module:
    hparams = params.ccs.hparams

    model = create_model(X_train, y_train, hparams)
    train_loader, val_loader = get_loaders(X_train, y_train, X_val, y_val, hparams.batch_size)

    live = Live(
        dir=f'{params.path.eval}/ccs/train/{seed}/fold_{fold}',
        report='html',
        dvcyaml=None,
        save_dvc_exp=False
    )

    model = model.to(DEVICE)
    compiled = compile_model(model, hparams, y_train.shape[1], live)
    model, history = fit(
        model,
        train_loader,
        val_loader,
        hparams.epochs,
        torch.device(DEVICE),
        **compiled
    )

    del live, history, compiled
    gc.collect()

    del train_loader, val_loader
    gc.collect()

    return model


def create_model(X_train: np.ndarray, y_train: np.ndarray, hparams: DictConfig) -> ICD92CCS:
    out_bias = focal_loss_bias_init(y_train, alpha=0.5, gamma=0.0).astype(np.float32)
    noise_v = X_train.std(axis=0).astype(np.float32)

    return ICD92CCS(
        n_features=X_train.shape[1],
        n_targets=y_train.shape[1],
        units=hparams.units,
        dropout_h1=hparams.dropout_h1,
        dropout_h2=hparams.dropout_h2,
        noise_v=noise_v,
        gauss_noise=hparams.gauss_noise,
        out_bias=out_bias,
    )


def get_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='fork',
        pin_memory=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='fork',
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, val_loader


def compile_model(
    model: nn.Module,
    hparams: DictConfig,
    n_targets: int,
    live: Optional[Live] = None
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), **hparams.adam)

    callbacks: List[Callback] = [
        EarlyStopping(
            restore_best_weights=True,
            **hparams.early_stop
        ),
        ReduceLROnPlateau(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **hparams.lr_scheduler.scheduler),
            monitor=hparams.lr_scheduler.monitor
        )
    ]

    if live is not None:
        callbacks.append(DVCLive(live=live))

    return {
        'optimizer': optimizer,
        'criterion': nn.BCEWithLogitsLoss(),
        'metrics': {
            'aucroc': MultilabelAUROC(n_targets),
            'aucpr':  MultilabelAveragePrecision(n_targets),
        },
        'callbacks': callbacks,
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Result:
    return {
        'roc_auc': cast(List[float], roc_auc_score(y_true, y_pred, average=None)),
        'pr_auc': cast(List[float], average_precision_score(y_true, y_pred, average=None))
    }


if __name__ == '__main__':
    run()
