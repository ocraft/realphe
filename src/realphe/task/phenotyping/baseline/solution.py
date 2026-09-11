import gc
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from dvclive.live import Live
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
import xarray as xr

from realphe.engine import Callback
from realphe.engine.callbacks import DVCLive, EarlyStopping, StepLR
from realphe.engine.loss import WeightedBCEWithLogitsLoss
from realphe.engine.metrics import MeanAveragePrecisionAtK
from realphe.engine.trainer import fit
from realphe.env import get_log
from realphe.pipe.nn import focal_loss_bias_init
from realphe.pipe.transformer import Transformer, fit_transform, transform
from realphe.pipe.transformer.encoder import OneHot
from realphe.pipe.transformer.scaler import StandardScaler
from realphe.task.phenotyping.setup.dataset import PhenotypingDataset, packed_ts_collate_fn
from realphe.task.phenotyping.setup.transformer import CohortImputer, SignalImputer


log = get_log(__name__)


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


class BaselinePhenotypingModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        lstm_units: int,
        dropout: float,
        y_bias: np.ndarray
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(lstm_units, n_targets, bias=True)

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
            with torch.no_grad():
                self.output.bias.copy_(torch.from_numpy(y_bias))

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        return lstm_out._replace(data=self.output(self.dropout(lstm_out.data)))


def train(
    seed: int,
    fold: int,
    cohort_train_set: xr.Dataset,
    signals_train_df: pd.DataFrame,
    cohort_val_set: xr.Dataset,
    signals_val_df: pd.DataFrame
):
    # setup
    set_random_seed(seed)

    params = OmegaConf.load('params.yaml').phenotyping
    out_dir = Path(f'model/phenotyping/baseline/{seed}/{fold}')
    os.makedirs(out_dir, exist_ok=True)

    # prepare data stream
    cohort_transformers, signals_transformers = transformers(params.baseline)

    train_gen, val_gen = raw_to_dataset(
        cohort_train_set,
        signals_train_df,
        cohort_val_set,
        signals_val_df,
        cohort_transformers,
        signals_transformers,
        seed
    )

    del cohort_train_set, signals_train_df, cohort_val_set, signals_val_df

    with open(out_dir / 'cohort_transformers.pkl', 'wb') as dump_file:
        pickle.dump(cohort_transformers, dump_file)

    with open(out_dir / 'signals_transformers.pkl', 'wb') as dump_file:
        pickle.dump(signals_transformers, dump_file)

    del cohort_transformers, signals_transformers

    # get loaders
    train_loader, val_loader = get_loaders(train_gen, val_gen, params.baseline.hparams.batch_size)

    # create model
    model = create_model(
        params.baseline.hparams,
        train_gen.n_features,
        train_gen.n_targets,
        train_gen.y_values
    )

    # train
    log.info(
        'Training baseline on '
        '(X_train=[%s], Y_train=[%s]) and '
        '(X_val=[%s], Y_val=[%s]) ...',
        train_gen.element_spec[0],
        train_gen.element_spec[1],
        val_gen.element_spec[0],
        val_gen.element_spec[1]
    )
    time_start = time.perf_counter()

    model = model.to(DEVICE)

    live = Live(
        dir=f'{params.path.eval}/baseline/train/{seed}/fold_{fold}',
        report='html',
        dvcyaml=None,
        save_dvc_exp=False
    )

    compiled = compile_model(model, params, train_gen.n_targets, live)
    model, history = fit(
        model,
        train_loader,
        val_loader,
        params.baseline.hparams.epochs,
        torch.device(DEVICE),
        **compiled
    )

    log.info('... finished training baseline in [%d]s.', time.perf_counter() - time_start)

    # save model
    model_path = out_dir / 'model.pt'
    torch.save(model, model_path)

    live.log_artifact(out_dir / 'cohort_transformers.pkl',
                      type='model',
                      labels=['transform', 'cohort'])
    live.log_artifact(out_dir / 'signals_transformers.pkl',
                      type='model',
                      labels=['transform', 'signals'])
    live.log_artifact(model_path, type='model')

    del live, compiled, history
    gc.collect()

    del model, train_loader, val_loader, train_gen, val_gen
    gc.collect()


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def transformers(
    params: DictConfig,
    fitted: bool = False,
    out_dir: Optional[Path] = None
) -> Tuple[List[Transformer], List[Transformer]]:
    if fitted and out_dir:
        with open(out_dir / 'cohort_transformers.pkl', 'rb') as dump_file:
            cohort_transformers = pickle.load(dump_file)

        with open(out_dir / 'signals_transformers.pkl', 'rb') as dump_file:
            signals_transformers = pickle.load(dump_file)

        return cohort_transformers, signals_transformers

    cohort_transformers = [
        CohortImputer(),
        StandardScaler(['age', 'height', 'hospstay_seq']),
        OneHot(['first_careunit', 'admission_type', 'insurance', 'marital_status', 'race'])
    ]

    signal_features_nominal = [
        'ectopy_type',
        'ectopy_type_secondary',
        'heart_rhythm',
        'temperature_site',
        'ventilator_mode',
        'ventilator_mode_hamilton',
        'ventilator_type',
        'mental_status'
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

    to_drop = [
    ]

    signal_features = list(params.signals)
    features_eng = []
    for fea in to_drop:
        signal_features.remove(fea)
    signal_features_to_scale = [fts for fts in signal_features + ['n_' + f for f in signal_features]
                                if fts not in signal_features_nominal + signal_features_binary]
    signal_features_to_scale += features_eng

    signals_transformers = [
        SignalImputer(signal_features, signal_features_binary),
        StandardScaler(signal_features_to_scale),
        OneHot(signal_features_nominal)
    ]
    return cohort_transformers, signals_transformers


def raw_to_dataset(
    cohort_train_set: xr.Dataset,
    signals_train_df: pd.DataFrame,
    cohort_val_set: xr.Dataset,
    signals_val_df: pd.DataFrame,
    cohort_transformers: List[Transformer],
    signals_transformers: List[Transformer],
    seed: int
) -> Tuple[PhenotypingDataset, PhenotypingDataset]:
    cohort_train_set = cast(xr.Dataset, fit_transform(cohort_train_set, cohort_transformers))
    signals_train_df = cast(pd.DataFrame, fit_transform(signals_train_df, signals_transformers))
    train_gen = get_dataset(cohort_train_set, signals_train_df, shuffle=True, seed=seed)

    cohort_val_set = cast(xr.Dataset, transform(cohort_val_set, cohort_transformers))
    signals_val_df = cast(pd.DataFrame, transform(signals_val_df, signals_transformers))
    val_gen = get_dataset(cohort_val_set, signals_val_df, shuffle=False)

    return train_gen, val_gen


def get_dataset(
    cohort_set: xr.Dataset,
    signals_df: pd.DataFrame,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> PhenotypingDataset:
    target = cohort_set.target
    cohort_df = (
        cohort_set
        .drop_dims(['label'])
        .to_pandas()
    )
    return PhenotypingDataset(
        signals_df.join(cohort_df, on='sample'),
        target.to_pandas(),
        shuffle=shuffle,
        seed=seed,
        reshuffle_each_iteration=shuffle
    )


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
        collate_fn=packed_ts_collate_fn,
        multiprocessing_context='fork'
    )

    val_loader = torch.utils.data.DataLoader(
        val_gen,
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=packed_ts_collate_fn,
        multiprocessing_context='fork'
    )

    return train_loader, val_loader


def create_model(
    hparams: DictConfig,
    n_features,
    n_targets,
    Y_train: np.ndarray
) -> BaselinePhenotypingModel:
    return BaselinePhenotypingModel(
        n_features,
        n_targets,
        hparams.model.lstm_units,
        hparams.model.dropout,
        focal_loss_bias_init(Y_train, alpha=0.5, gamma=0.0)
    )


def compile_model(
    model: nn.Module,
    params: DictConfig,
    n_targets: int,
    live: Optional[Live] = None
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), **params.baseline.hparams.adam)
    map_k = params.evaluate.main.map_k

    callbacks: List[Callback] = [
        EarlyStopping(
            restore_best_weights=True,
            **params.baseline.hparams.early_stop
        ),
        StepLR(torch.optim.lr_scheduler.StepLR(
            optimizer,
            **params.baseline.hparams.lr_scheduler
        ))
    ]
    if live is not None:
        callbacks.append(DVCLive(live=live))

    return {
        'optimizer': optimizer,
        'criterion': WeightedBCEWithLogitsLoss(
            **params.baseline.hparams.wbce,
        ),
        'metrics': {
            'fs_aucroc': MultilabelAUROC(n_targets),
            'ls_aucroc': MultilabelAUROC(n_targets),
            'fs_aucpr':  MultilabelAveragePrecision(n_targets),
            'ls_aucpr': MultilabelAveragePrecision(n_targets),
            f'fs_mAP@{map_k}': MeanAveragePrecisionAtK(k=map_k),
            f'ls_mAP@{map_k}': MeanAveragePrecisionAtK(k=map_k),
        },
        'callbacks': callbacks,
    }
