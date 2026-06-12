from contextlib import contextmanager
import gc
import multiprocessing as mp
import os
import random
from typing import Any, Dict, List, Optional, Tuple, cast

from dvclive.live import Live
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchmetrics.classification import BinaryAUROC
import tqdm
import xarray as xr

from realphe.engine import Callback
from realphe.engine.callbacks import DVCLive, EarlyStopping
from realphe.engine.trainer import fit
from realphe.env import get_log
from realphe.pipe.metrics import roc_auc
from realphe.pipe.monitor import Monitor
from realphe.pipe.splitter import KFoldSplitter, get_fold, split
from realphe.pipe.transformer import fit_transform, transform
from realphe.pipe.transformer.encoder import OneHot
from realphe.pipe.transformer.scaler import StandardScaler
from realphe.pipe.transformer.shape import Drop

from .dataset import PhenotypingDataset, packed_sample_collate_fn
from .transformer import CohortImputer, SignalImputer


log = get_log(__name__)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class AdversarialModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        lstm_units: int,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bias=True
        )
        self.output = nn.Linear(lstm_units, 1, bias=True)

        for name, param in self.lstm.named_parameters():

            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

                if 'bias_ih' in name:
                    # unit_forget_bias=True
                    # PyTorch gate order: [input, forget, cell, output]
                    hidden_size = self.lstm.hidden_size
                    with torch.no_grad():
                        param[hidden_size:2 * hidden_size].fill_(1.0)

        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, X):
        _, (h_n, _) = self.lstm(X)
        return self.output(h_n).squeeze(0)  # logits


def run():
    # load params
    params = load_params()
    set_random_seed(params.evaluate.adversarial.random_seed)

    # prepare data
    cohort_set, signals_df = get_input_set(params)

    # out-of-fold cross-validation
    evaluate(cohort_set, signals_df, params)


def load_params():
    return OmegaConf.load('params.yaml').phenotyping


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_input_set(params: DictConfig) -> Tuple[xr.Dataset, pd.DataFrame]:
    params_path = params.path
    params_av = params.evaluate.adversarial

    cohort_set = xr.open_dataset(
        f'{params_path.data.processed}/setup/cohort.zarr', consolidated=False)
    cohort_set = (
        cohort_set
        .drop_vars(['target', 'icd9', 'icd9_name'])
        .drop_vars(list(params.cohort.outcomes))
        .drop_dims(['diagnose', 'label'])
        [['sample'] + list(params.cohort.features) + ['split']]

    )
    cohort_set.load()

    signals_df = (
        xr.open_mfdataset(
            [
                f'{params_path.data.processed}/setup/signals.zarr',
                f'{params_path.data.processed}/setup/signals_event_count.zarr',
            ],
            compat='override',
            consolidated=False
        )[
            ['sample', 'step'] +
            list(params.signals.features) +
            ['n_' + f for f in params.signals.features if f not in ['time_at']]
        ]
        .to_pandas()
        .reset_index()
    )
    signals_df.set_index(['sample', 'step'], inplace=True)

    # split dataset into folds
    splitter = KFoldSplitter(n_splits=params_av.cv.folds,
                             shuffle=params_av.cv.shuffle,
                             random_state=params_av.random_seed,
                             var='split')

    return split(cohort_set, splitter), signals_df


def evaluate(cohort_set: xr.Dataset, signals_df: pd.DataFrame, params: DictConfig):
    params_av = params.evaluate.adversarial

    oof = xr.Dataset({
        'Y_pred': xr.DataArray(
            np.zeros_like(cohort_set['split'].values, dtype='float64'),
            dims=['sample'],
            coords=cohort_set['split'].coords
        )
    })

    with monitor_for(cohort_set, params, oof) as monitor:
        for fold in tqdm.tqdm(range(params_av.cv.folds)):
            process_fold(cohort_set, signals_df, fold, params, oof, monitor)


@contextmanager
def monitor_for(input_set: xr.Dataset, params: DictConfig, oof: xr.Dataset):
    params_av = params.evaluate.adversarial

    monitor = Monitor(eval_path=f'{params.path.eval}/setup/adversarial/eval', dvcyaml=None)
    monitor.log_params(params_av)

    yield monitor

    monitor.log_cv(params_av.random_seed, results(input_set.split.values, oof.Y_pred.values))
    monitor.live.log_sklearn_plot(
        "roc",
        input_set.split.values,
        oof.Y_pred.values,
        "aucroc.json",
        title="Adversarial Validation::OOF::AUC-ROC"
    )
    monitor.live.log_sklearn_plot(
        "precision_recall",
        input_set.split.values,
        oof.Y_pred.values,
        "aucprc.json",
        title="Adversarial Validation::OOF::AUC-PRC"
    )
    monitor.live.log_sklearn_plot(
        "confusion_matrix",
        input_set.split.values.squeeze(),
        (oof.Y_pred.values > 0.5).squeeze().astype(int),
        "cm.json",
        title="Adversarial Validation::OOF::Confusion Matrix"
    )
    monitor.end()


def results(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'roc_auc': roc_auc(y_true, y_pred)
    }


def process_fold(cohort_set, signals_df, fold, params, oof, monitor):
    params_av = params.evaluate.adversarial

    cohort_train_set, cohort_val_set, signals_train_df, signals_val_df = prepare_fold(
        cohort_set, signals_df, fold, params)

    train_gen = get_dataset(cohort_train_set, signals_train_df, params_av)
    val_gen = get_dataset(cohort_val_set, signals_val_df, params_av, shuffle=False)

    model = get_model(params_av.hparams,
                      len(cohort_train_set.keys()) + len(signals_train_df.columns))

    del signals_train_df, signals_val_df
    gc.collect()

    oof.Y_pred.loc[cohort_val_set.sample] = train(fold, model, train_gen, val_gen, params)

    log_metrics(monitor, params_av, oof, cohort_val_set)

    del cohort_train_set, cohort_val_set, train_gen, val_gen, model
    gc.collect()


def prepare_fold(cohort_set: xr.Dataset,
                 signals_df: pd.DataFrame,
                 fold: int,
                 params: DictConfig) -> Tuple[xr.Dataset, xr.Dataset, pd.DataFrame, pd.DataFrame]:

    cohort_train_set, cohort_val_set = get_fold(cohort_set, fold)
    signals_train_df = cast(pd.DataFrame, signals_df.loc[cohort_train_set.sample, :].copy())
    signals_val_df = cast(pd.DataFrame, signals_df.loc[cohort_val_set.sample, :].copy())

    cohort_transformers = [
        Drop(['admission_time'] +
             [x for x in cast(List[str], cohort_train_set.coords) if x.startswith('fold_')]),
        CohortImputer(),
        StandardScaler(['age', 'height', 'hospstay_seq']),
        OneHot(['first_careunit', 'admission_type', 'admission_location', 'last_service',
                'insurance', 'marital_status', 'race'])
    ]

    signal_features_nominal = [
        'ectopy_type',
        'ectopy_type_secondary',
        'heart_rhythm',
        'temperature_site',
        'ventilator_mode',
        'ventilator_mode_hamilton',
        'ventilator_type',
        'o2_delivery_device_1',
        'o2_delivery_device_2',
        'o2_delivery_device_3',
        'o2_delivery_device_4',
        'mental_status',
    ]

    signal_features_binary = [
        'alarms_on',
        'st_segment_monitoring_on',
        'parameters_checked',
        'gcs_unable',
        'capillary_refill_rate_l',
        'capillary_refill_rate_r',
        'high_risk_gt_51_interventions',
        'history_of_falling_3m',
        'iv_saline_lock',
        'secondary_diagnosis',
        'gauge_18_dressing_occlusive',
        'gauge_18_placed_in_outside_facility',
        'gauge_20_dressing_occlusive',
        'gauge_20_placed_in_outside_facility',
        'gauge_20_placed_in_the_field',
        'eye_care'
    ]

    signal_features = list(params.signals.features)
    signal_features.remove('time_at')
    signal_features_to_scale = [fts for fts in signal_features + ['n_' + f for f in signal_features]
                                if fts not in signal_features_nominal + signal_features_binary]

    signals_transformers = [
        Drop(['time_at']),
        SignalImputer(signal_features, signal_features_binary),
        StandardScaler(signal_features_to_scale),
        OneHot(signal_features_nominal)
    ]

    cohort_train_set = fit_transform(cohort_train_set, cohort_transformers)
    cohort_val_set = transform(cohort_val_set, cohort_transformers)

    signals_train_df = fit_transform(signals_train_df, signals_transformers)
    signals_val_df = transform(signals_val_df, signals_transformers)

    return (
        cast(xr.Dataset, cohort_train_set),
        cast(xr.Dataset, cohort_val_set),
        cast(pd.DataFrame, signals_train_df),
        cast(pd.DataFrame, signals_val_df)
    )


def get_dataset(
    cohort_set: xr.Dataset,
    signals_df: pd.DataFrame,
    params_av: DictConfig,
    shuffle: bool = True
) -> PhenotypingDataset:

    cohort_df = cohort_set.to_pandas()
    target = cohort_df['split']
    cohort_df.drop('split', inplace=True, axis=1)

    return PhenotypingDataset(
        signals_df.join(cohort_df, on='sample'),
        target.to_frame(),
        shuffle=shuffle,
        seed=params_av.random_seed,
        reshuffle_each_iteration=shuffle
    )


def get_model(hparams: DictConfig, n_features: int) -> AdversarialModel:
    return AdversarialModel(
        n_features=n_features,
        lstm_units=hparams.model.lstm_units
    )


def train(
    fold: int,
    model: nn.Module,
    train_gen: PhenotypingDataset,
    val_gen: PhenotypingDataset,
    params: DictConfig
) -> np.ndarray:

    # setup

    params_av = params.evaluate.adversarial
    hparams = params_av.hparams

    train_loader, val_loader = get_loaders(train_gen, val_gen, hparams.batch_size)

    live = Live(
        dir=f'{params.path.eval}/setup/adversarial/train/{params_av.random_seed}/fold_{fold}',
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
    del compiled, live, history
    gc.collect()

    del train_gen, train_loader
    gc.collect()

    # return
    model.eval()
    preds = []
    with torch.inference_mode():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)

            preds.append(torch.sigmoid(model(X_batch)).squeeze().cpu().detach().numpy())

    del val_gen, val_loader
    gc.collect()

    return np.concatenate(preds)


def get_loaders(
    train_gen: PhenotypingDataset,
    val_gen: PhenotypingDataset,
    batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        train_gen,
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn,
        multiprocessing_context='fork'
    )

    val_loader = torch.utils.data.DataLoader(
        val_gen,
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn,
        multiprocessing_context='fork'
    )
    return train_loader, val_loader


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[PackedSequence, torch.Tensor]:
    X_batch, Y_batch = packed_sample_collate_fn(batch)
    return X_batch, Y_batch.half()


def log_metrics(monitor: Monitor, params_av: DictConfig, oof_set: xr.Dataset, val_set: xr.Dataset):
    monitor.log_cv_fold(
        params_av.random_seed,
        results(val_set.split.values, oof_set.Y_pred.loc[val_set.sample].values)
    )
    monitor.log_learning_curve(
        ['aucroc', 'loss'], title="Adversarial Validation::Learning Curve")
    monitor.live.make_report()


def compile_model(
    model: nn.Module,
    hparams: DictConfig,
    live: Optional[Live] = None
) -> Dict[str, Any]:

    callbacks: List[Callback] = [
        EarlyStopping(
            restore_best_weights=True,
            **hparams.early_stop
        )
    ]

    if live is not None:
        callbacks.append(DVCLive(live=live))

    return {
        'optimizer': torch.optim.Adam(model.parameters(), **hparams.adam),
        'criterion': nn.BCEWithLogitsLoss(),
        'metrics': {
            'aucroc': BinaryAUROC(),
        },
        'callbacks': callbacks
    }


if __name__ == '__main__':
    run()
