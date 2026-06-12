from functools import partial
import gc
import multiprocessing as mp
import os
import time
from typing import cast

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
import xarray as xr

from realphe.env import get_log
from realphe.pipe.metrics import bootstrap, div_zero, map_at_k, threshold
from realphe.pipe.shared import SharedArray

from .time_series import index_of_first_step, index_of_last_step

log = get_log(__name__)


def run():
    params_phenotyping = OmegaConf.load('params.yaml').phenotyping
    params_cli = OmegaConf.from_cli()
    solution_name = params_cli.solution
    out_path = params_phenotyping.path.data.processed
    interpret_path = f'{params_phenotyping.path.eval}/{solution_name}/interpret'

    log.info('Loading cohort ...')
    time_start = time.perf_counter()
    cohort = (
        xr.open_dataset(f'{out_path}/{solution_name}/cohort.zarr', consolidated=False)
        .load()
    )
    log.info('... finished loading cohort in [%d]s.', time.perf_counter() - time_start)

    # OOF
    oof = pd.read_parquet(f'{out_path}/{solution_name}/oof.parquet',
                          engine='fastparquet').reset_index().set_index(['sample', 'step'])

    log.info('Bootstrap oof confidence intervals for [%s] ...', solution_name)
    time_start = time.perf_counter()
    oof_result_df = bootstrap_ci(cohort.target.loc[cast(pd.MultiIndex, oof.index).levels[0], :],
                                 oof,
                                 sample_size=0.2)
    os.makedirs(f'{interpret_path}/oof', exist_ok=True)
    oof_result_df.to_csv(f'{interpret_path}/oof/ci.csv')
    log.info('... finished bootstrapping oof confidence intervals for [%s] in [%d]s.',
             solution_name,
             time.perf_counter() - time_start)
    del oof

    # Test
    test = pd.read_parquet(f'{out_path}/{solution_name}/test.parquet',
                           engine='fastparquet').reset_index().set_index(['sample', 'step'])

    log.info('Bootstrap test confidence intervals for [%s] ...', solution_name)
    time_start = time.perf_counter()
    test_result_df = bootstrap_ci(cohort.target.loc[cast(pd.MultiIndex, test.index).levels[0], :],
                                  test,
                                  sample_size=0.2)
    os.makedirs(f'{interpret_path}/test', exist_ok=True)
    test_result_df.to_csv(f'{interpret_path}/test/ci.csv')
    del test
    log.info('... finished bootstrapping test confidence intervals for [%s] in [%d]s.',
             solution_name,
             time.perf_counter() - time_start)


def bootstrap_ci(
    y_true: xr.DataArray,
    y_pred: pd.DataFrame,
    n_jobs: int = -1,
    sample_size: float = 1.0
) -> pd.DataFrame:
    if n_jobs != -1 and n_jobs > mp.cpu_count():
        n_jobs = mp.cpu_count()

    labels = y_pred.columns
    first_step_idx = index_of_first_step(y_pred)
    last_step_idx = index_of_last_step(y_pred)
    y_true_sa = SharedArray(
        np.repeat(
            y_true.values.astype(np.int8),
            y_pred.groupby(level=0).size(),
            axis=0
        )
    )
    y_pred_sa = SharedArray(y_pred.values)
    del y_pred
    gc.collect()

    # first step
    log.info('... first step ...')
    y_true_fs_sa = SharedArray(y_true.values.astype(np.int8))
    y_pred_fs_sa = SharedArray(y_pred_sa.read()[first_step_idx, :])
    ci_fs = bootstrap_ci_for(y_true_fs_sa, y_pred_fs_sa)
    y_true_fs_sa.unlink()
    y_pred_fs_sa.unlink()
    del y_true_fs_sa, y_pred_fs_sa

    # last step
    log.info('... last step ...')
    y_true_ls_sa = SharedArray(y_true.values.astype(np.int8))
    y_pred_ls_sa = SharedArray(y_pred_sa.read()[last_step_idx, :])
    ci_ls = bootstrap_ci_for(y_true_ls_sa, y_pred_ls_sa)
    y_true_ls_sa.unlink()
    y_pred_ls_sa.unlink()
    del y_true_ls_sa, y_pred_ls_sa

    # time distributed
    log.info('... time distributed ...')
    ci_td = bootstrap_ci_for(y_true_sa, y_pred_sa, sample_size=sample_size, n_jobs=n_jobs)
    y_true_sa.unlink()
    y_pred_sa.unlink()
    del y_true_sa, y_pred_sa

    out = pd.DataFrame(index=list(labels)+['mean'], dtype=np.float32)

    ci_td_roc_auc = ci_td[0]
    ci_td_pr_auc = ci_td[1]
    ci_td_mapk = ci_td[2]
    ci_td_cm = ci_td[3]
    ci_ls_roc_auc = ci_ls[0]
    ci_ls_pr_auc = ci_ls[1]
    ci_ls_mapk = ci_ls[2]
    ci_ls_cm = ci_ls[3]
    ci_fs_roc_auc = ci_fs[0]
    ci_fs_pr_auc = ci_fs[1]
    ci_fs_mapk = ci_fs[2]
    ci_fs_cm = ci_fs[3]
    out['td_roc_auc_low'] = ci_td_roc_auc[0]
    out['td_roc_auc_mean'] = ci_td_roc_auc[1]
    out['td_roc_auc_up'] = ci_td_roc_auc[2]
    out['td_pr_auc_low'] = ci_td_pr_auc[0]
    out['td_pr_auc_mean'] = ci_td_pr_auc[1]
    out['td_pr_auc_up'] = ci_td_pr_auc[2]
    out['ls_roc_auc_low'] = ci_ls_roc_auc[0]
    out['ls_roc_auc_mean'] = ci_ls_roc_auc[1]
    out['ls_roc_auc_up'] = ci_ls_roc_auc[2]
    out['ls_pr_auc_low'] = ci_ls_pr_auc[0]
    out['ls_pr_auc_mean'] = ci_ls_pr_auc[1]
    out['ls_pr_auc_up'] = ci_ls_pr_auc[2]
    out['fs_roc_auc_low'] = ci_fs_roc_auc[0]
    out['fs_roc_auc_mean'] = ci_fs_roc_auc[1]
    out['fs_roc_auc_up'] = ci_fs_roc_auc[2]
    out['fs_pr_auc_low'] = ci_fs_pr_auc[0]
    out['fs_pr_auc_mean'] = ci_fs_pr_auc[1]
    out['fs_pr_auc_up'] = ci_fs_pr_auc[2]
    out['td_mAP@10_low'] = ci_td_mapk[0]
    out['td_mAP@10_mean'] = ci_td_mapk[1]
    out['td_mAP@10_up'] = ci_td_mapk[2]
    out['ls_mAP@10_low'] = ci_ls_mapk[0]
    out['ls_mAP@10_mean'] = ci_ls_mapk[1]
    out['ls_mAP@10_up'] = ci_ls_mapk[2]
    out['fs_mAP@10_low'] = ci_fs_mapk[0]
    out['fs_mAP@10_mean'] = ci_fs_mapk[1]
    out['fs_mAP@10_up'] = ci_fs_mapk[2]
    out[['td_precision_low', 'td_recall_low', 'td_f1_score_low',
        'td_sensitivity_low', 'td_specificity_low']] = ci_td_cm[0].T
    out[['td_precision_mean', 'td_recall_mean', 'td_f1_score_mean',
        'td_sensitivity_mean', 'td_specificity_mean']] = ci_td_cm[1].T
    out[['td_precision_up', 'td_recall_up', 'td_f1_score_up',
        'td_sensitivity_up', 'td_specificity_up']] = ci_td_cm[2].T
    out[['ls_precision_low', 'ls_recall_low', 'ls_f1_score_low',
        'ls_sensitivity_low', 'ls_specificity_low']] = ci_ls_cm[0].T
    out[['ls_precision_mean', 'ls_recall_mean', 'ls_f1_score_mean',
        'ls_sensitivity_mean', 'ls_specificity_mean']] = ci_ls_cm[1].T
    out[['ls_precision_up', 'ls_recall_up', 'ls_f1_score_up',
        'ls_sensitivity_up', 'ls_specificity_up']] = ci_ls_cm[2].T
    out[['fs_precision_low', 'fs_recall_low', 'fs_f1_score_low',
        'fs_sensitivity_low', 'fs_specificity_low']] = ci_fs_cm[0].T
    out[['fs_precision_mean', 'fs_recall_mean', 'fs_f1_score_mean',
        'fs_sensitivity_mean', 'fs_specificity_mean']] = ci_fs_cm[1].T
    out[['fs_precision_up', 'fs_recall_up', 'fs_f1_score_up',
        'fs_sensitivity_up', 'fs_specificity_up']] = ci_fs_cm[2].T
    return out


def bootstrap_ci_for(
    y_true_sa: SharedArray,
    y_pred_sa: SharedArray,
    sample_size: float = 1.0,
    n_jobs: int = -1
):
    def roc_auc(y_true, y_pred):
        scores = roc_auc_score(y_true, y_pred, average=None)
        return np.r_[scores, np.mean(scores)]

    def pr_auc(y_true, y_pred):
        scores = average_precision_score(y_true, y_pred, average=None)
        return np.r_[scores, np.mean(scores)]

    log.info('... AUC-ROC ...')
    ci_roc_auc = bootstrap(
        y_true_sa,
        y_pred_sa,
        roc_auc,
        sample_size=sample_size,
        n_jobs=n_jobs
    )
    log.info('... AUC-PR ...')
    ci_pr_auc = bootstrap(
        y_true_sa,
        y_pred_sa,
        pr_auc,
        sample_size=sample_size,
        n_jobs=n_jobs
    )
    log.info('... mAP@10 ...')
    ci_mapk = bootstrap(
        y_true_sa,
        y_pred_sa,
        partial(map_at_k, k=10),
        sample_size=sample_size,
        n_jobs=n_jobs
    )
    log.info('... Confusion Matrix ...')
    ci_cm = bootstrap(
        y_true_sa,
        y_pred_sa,
        confusion_metrics,
        sample_size=sample_size,
        n_jobs=n_jobs
    )
    return ci_roc_auc, ci_pr_auc, ci_mapk, ci_cm


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm_val = threshold(multilabel_confusion_matrix)(y_true, y_pred)

    tp = cm_val[:, 1, 1]
    fp = cm_val[:, 0, 1]
    fn = cm_val[:, 1, 0]
    tn = cm_val[:, 0, 0]
    scores = np.array([
        div_zero(tp, (tp+fp)),
        div_zero(tp, (tp+fn)),
        div_zero((2*tp), ((2*tp)+fp+fn)),
        div_zero(tp, (tp+fn)),
        div_zero(tn, (tn+fp))
    ], dtype=np.float32)

    return np.c_[scores, np.mean(scores, axis=1)]


if __name__ == '__main__':
    run()
