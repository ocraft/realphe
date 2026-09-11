import gc
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from dvclive.live import Live
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import tqdm
import xarray as xr

from realphe.engine import Callback
from realphe.engine.callbacks import DVCLive, EarlyStopping, ReduceLROnPlateau
from realphe.engine.trainer import fit
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


class Death30dICU(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        units: int,
        dropout_h1: float,
        out_bias: np.ndarray,
    ):
        super().__init__()

        fc1 = nn.Linear(n_features, units)
        act1 = nn.LeakyReLU(negative_slope=0.3)
        dropout1 = nn.Dropout(dropout_h1)

        # Output layer
        out = nn.Linear(units, n_targets)

        self._dense_init(fc1)
        nn.init.xavier_uniform_(out.weight)

        with torch.no_grad():
            out.bias.copy_(
                torch.as_tensor(out_bias, dtype=torch.float32)
            )

        self.mlp = nn.Sequential(
            fc1,
            act1,
            dropout1,
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
    out_path = f'{params.outcome.path.data.processed}/death_30d_icu/'
    os.makedirs(out_path, exist_ok=True)

    # load phenotypes
    cohort, oof, test = load_phenotypes(phe_path, solution_name)

    # prepare training data
    y_oof, y_test = to_outcome(cohort, oof, test)

    # train and evaluate
    y_oof_agg_pred, y_test_agg_pred = predict_death(oof, y_oof, test, y_test, params)

    y_oof_agg_pred.to_parquet(
        f'{out_path}/oof_{solution_name}.parquet',
        engine='fastparquet'
    )
    y_test_agg_pred.to_parquet(
        f'{out_path}/test_{solution_name}.parquet',
        engine='fastparquet'
    )


def load_params() -> DictConfig:
    return cast(DictConfig, OmegaConf.load('params.yaml'))


def load_phenotypes(path: str, solution_name: str) -> Tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    log.info('Loading phenotypes ...')
    time_start = time.perf_counter()
    cohort = (
        xr.open_dataset(f'{path}/setup/cohort.zarr', consolidated=False)
        .load()
    )
    log.info('... finished loading cohort in [%d]s.', time.perf_counter() - time_start)

    oof = pd.read_parquet(f'{path}/{solution_name}/oof.parquet',
                          engine='fastparquet').reset_index().set_index(['sample', 'step'])
    test = pd.read_parquet(f'{path}/{solution_name}/test.parquet',
                           engine='fastparquet').reset_index().set_index(['sample', 'step'])
    log.info('... finished loading phenotypes in [%d]s.', time.perf_counter() - time_start)
    return cohort, oof, test


def to_outcome(
    cohort: xr.Dataset,
    oof: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    y_test = pd.DataFrame(np.repeat(
        cohort.sel(sample=cast(pd.MultiIndex, test.index).levels[0])['thirtyday_expire_flag'],
        test.groupby(level=0).size(),
        axis=0
    ), index=test.index, columns=['thirtyday_expire_flag'])

    y_oof = pd.DataFrame(np.repeat(
        cohort.sel(sample=cast(pd.MultiIndex, oof.index).levels[0])['thirtyday_expire_flag'],
        oof.groupby(level=0).size(),
        axis=0
    ), index=oof.index, columns=['thirtyday_expire_flag'])

    return y_oof, y_test


def predict_death(
    oof: pd.DataFrame,
    y_oof: pd.DataFrame,
    test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: DictConfig
):
    torch.multiprocessing.set_sharing_strategy('file_system')

    monitor = Monitor(
        eval_path=f'{params.outcome.path.eval}/death_30d_icu/eval',
        dvcyaml=None
    )
    monitor.log_params(params.outcome.death_30d_icu)

    y_test_agg_pred = pd.DataFrame(
        0,
        index=y_test.index,
        columns=y_test.columns,
        dtype=np.float64
    )
    y_oof_agg_pred = pd.DataFrame(
        0,
        index=y_oof.index,
        columns=y_oof.columns,
        dtype=np.float32
    )

    seeds = params.outcome.death_30d_icu.evaluate.cv.seeds
    n_folds = params.outcome.death_30d_icu.evaluate.cv.folds
    for i_seed, seed in tqdm.tqdm(enumerate(seeds)):
        set_random_seed(seed)

        splitter = create_splitter(n_folds, params.outcome.death_30d_icu.evaluate.cv.shuffle, seed)
        for fold, (train, val) in tqdm.tqdm(enumerate(
            splitter.split(
                oof.values,
                y_oof.values,
                groups=cast(pd.MultiIndex, oof.index).get_level_values(0)
            )
        )):
            model = train_task(sel(oof, train),
                               sel(y_oof, train),
                               sel(oof, val),
                               sel(y_oof, val),
                               params.outcome,
                               seed=seed,
                               fold=fold)
            model = SigmoidOutput(model)

            y_test_pred = model(torch.from_numpy(test.values).to(DEVICE)).detach().cpu().numpy()

            y_test_agg_pred = y_test_agg_pred.add(pd.DataFrame(
                y_test_pred,
                index=y_test.index,
                columns=y_test.columns,
                dtype=np.float64
            ))

            y_val_pred = model(torch.from_numpy(sel(oof, val)).to(DEVICE)).detach().cpu().numpy()
            del model
            y_oof_agg_pred.iloc[val, :] += y_val_pred

            monitor.log_cv_fold(
                seed,
                metrics(
                    sel(y_oof, val),
                    y_val_pred
                ),
                metrics(
                    y_test.values,
                    y_test_pred
                )
            )
            monitor.live.make_report()
            del y_test_pred, y_val_pred
            gc.collect()

        monitor.log_cv(
            seed,
            metrics(
                y_oof.values,
                y_oof_agg_pred.values / ((i_seed + 1) * n_folds)
            ),
            metrics(
                y_test.values,
                y_test_agg_pred.values / ((i_seed + 1) * n_folds)
            )
        )

    y_test_agg_pred = y_test_agg_pred.div(n_folds * len(seeds))
    y_test_agg_pred.columns = y_test_agg_pred.columns.astype(str)
    y_oof_agg_pred = y_oof_agg_pred.div(n_folds * len(seeds))
    y_oof_agg_pred.columns = y_oof_agg_pred.columns.astype(str)

    monitor.log_learning_curve(
        ['aucpr', 'aucroc', 'loss'],
        title='Outcome::30 Days Mortality::Learning Curve'
    )
    monitor.end(
        metrics(
            y_oof.values,
            y_oof_agg_pred.values
        ),
        metrics(
            y_test.values,
            y_test_agg_pred.values
        )
    )
    return y_oof_agg_pred, y_test_agg_pred


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def create_splitter(n_folds: int, shuffle: bool, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=seed
    )


def sel(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return df.iloc[idx, :].values


def train_task(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: DictConfig,
    seed: int,
    fold: int
) -> nn.Module:
    hparams = params.death_30d_icu.hparams

    model = create_model(X_train, y_train, hparams)
    train_loader, val_loader = get_loaders(X_train, y_train, X_val, y_val, hparams.batch_size)

    live = Live(
        dir=f'{params.path.eval}/death_30d_icu/train/{seed}/fold_{fold}',
        report='html',
        dvcyaml=None,
        save_dvc_exp=False
    )

    model = model.to(DEVICE)
    compiled = compile_model(model, hparams, live)
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


def create_model(X_train: np.ndarray, y_train: np.ndarray, hparams: DictConfig) -> Death30dICU:
    out_bias = focal_loss_bias_init(y_train, alpha=0.5, gamma=0.0).astype(np.float32)

    return Death30dICU(
        n_features=X_train.shape[1],
        n_targets=y_train.shape[1],
        units=hparams.units,
        dropout_h1=hparams.dropout_h1,
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
        pin_memory=True,
        shuffle=True,
        multiprocessing_context='fork'
    )

    val_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False,
        multiprocessing_context='fork'
    )
    return train_loader, val_loader


def compile_model(
    model: nn.Module,
    hparams: DictConfig,
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
            'aucroc': BinaryAUROC(),
            'aucpr':  BinaryAveragePrecision(),
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
