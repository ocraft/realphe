from functools import partial, wraps
import gc
import importlib
import os
from pathlib import Path
import random
from re import sub
from typing import Callable, Dict, List, Tuple, cast

from joblib import Parallel, delayed
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
import tqdm
import xarray as xr

from realphe.env import get_log
from realphe.pipe.metrics import SampleIdx, div_zero, map_at_k, pr_roc_auc, threshold
from realphe.pipe.monitor import Monitor
from realphe.pipe.shared import SharedArray, SharedDataFrame
from realphe.pipe.splitter import get_fold_val

from .setup import get_eval_set, load_split_for
from .time_series import (
    index_of_first_step,
    index_of_last_step,
    td_y_true,
    time_series_idx,
)

log = get_log(__name__)

OOF_OUT = 'oof.parquet'
TEST_OUT = 'test.parquet'


# def {name}(
#     seed: int,
#     fold: int,
#     cohort_set: Dataset,
#     signals_df: pd.DataFrame
# ) -> np.ndarray:
SolutionPredict = Callable[[xr.Dataset, pd.DataFrame, int, int], np.ndarray]


def run():
    params = OmegaConf.from_cli()
    solution_name = params.solution

    solution = load_solution(solution_name)
    evaluate(solution, solution_name)


def load_solution(solution_name: str) -> SolutionPredict:
    return getattr(
        importlib.import_module('.load', package=f'realphe.task.phenotyping.{solution_name}'),
        'pipeline_single')


def evaluate(solution: SolutionPredict, solution_name: str):
    # load params
    params_phenotyping, params_eval = load_params(solution_name)
    out_dir = f'{params_phenotyping.path.data.processed}/{solution_name}'
    os.makedirs(out_dir, exist_ok=True)

    # random seed
    set_random_seed(params_eval.random_seed)

    # prepare data
    cohort_train_set, signals_train_df, cohort_test_set, signals_test_df = (
        get_eval_set(params_phenotyping, solution_name)
    )

    resume = os.path.exists(f'{out_dir}/{OOF_OUT}') or os.path.exists(f'{out_dir}/{TEST_OUT}')
    log.info('Resume from checkpoint: [%s]', resume)

    monitor = Monitor(
        eval_path=f'{params_phenotyping.path.eval}/{solution_name}/eval',
        dvcyaml=None,
        resume=resume
    )
    monitor.log_params(params_phenotyping[solution_name])
    monitor.log_learning_curve(
        ['ls_aucpr', 'ls_aucroc', 'loss', f'ls_mAP@{params_eval.map_k}',
         'fs_aucpr', 'fs_aucroc', f'fs_mAP@{params_eval.map_k}'],
        title=f'{name_of(solution_name)}::Learning Curve'
    )

    seeds_to_evaluate = params_eval.cv.seeds
    if resume:
        seeds_to_evaluate = [seed for seed in seeds_to_evaluate
                             if str(seed) not in set(monitor._track.keys())]

    oof_y_true, oof_first_step_idx, oof_last_step_idx = time_series_idx(cohort_train_set,
                                                                        signals_train_df)

    test_y_true, test_first_step_idx, test_last_step_idx = time_series_idx(cohort_test_set,
                                                                           signals_test_df)

    for seed in tqdm.tqdm(seeds_to_evaluate):
        cohort_train_set = load_split_for(
            params_phenotyping.path,
            solution_name,
            cohort_train_set,
            seed
        )
        oof_seed, test_seed = cross_validate(
            solution,
            solution_name,
            seed,
            cohort_train_set,
            signals_train_df,
            cohort_test_set,
            signals_test_df,
            params_eval,
            monitor,
            test_y_true,
            test_first_step_idx,
            test_last_step_idx
        )

        monitor.log_cv(
            seed,
            metrics(
                oof_y_true,
                oof_seed.array(),
                oof_first_step_idx,
                oof_last_step_idx,
                params_eval,
                f'oof seed={seed}'
            ),
            metrics(
                test_y_true,
                test_seed.array(),
                test_first_step_idx,
                test_last_step_idx,
                params_eval,
                f'test seed={seed}'
            )
        )
        gc.collect()

        if os.path.exists(f'{out_dir}/{OOF_OUT}'):
            map_result(Path(f'{out_dir}/{OOF_OUT}'),
                       lambda df: df.add(oof_seed.read().loc[df.index]))
        else:
            oof_seed.read().to_parquet(f'{out_dir}/{OOF_OUT}', engine='fastparquet')

        oof_seed.unlink()
        del oof_seed
        gc.collect()

        if os.path.exists(f'{out_dir}/{TEST_OUT}'):
            map_result(Path(f'{out_dir}/{TEST_OUT}'),
                       lambda df: df.add(test_seed.read().loc[df.index]))
        else:
            test_seed.read().to_parquet(f'{out_dir}/{TEST_OUT}', engine='fastparquet')

        test_seed.unlink()
        del test_seed
        gc.collect()

    map_result(Path(f'{out_dir}/{OOF_OUT}'), lambda df: df.div(len(params_eval.cv.seeds)))
    map_result(Path(f'{out_dir}/{TEST_OUT}'), lambda df: df.div(len(params_eval.cv.seeds)))

    oof_y_pred = SharedArray(
        pd.read_parquet(f'{out_dir}/{OOF_OUT}', engine='fastparquet').values
    )
    test_y_pred = SharedArray(
        pd.read_parquet(f'{out_dir}/{TEST_OUT}', engine='fastparquet').values
    )
    monitor.end(
        metrics(
            oof_y_true,
            oof_y_pred,
            oof_first_step_idx,
            oof_last_step_idx,
            params_eval,
            'oof final'
        ),
        metrics(
            test_y_true,
            test_y_pred,
            test_first_step_idx,
            test_last_step_idx,
            params_eval,
            'test final'
        )
    )

    oof_y_true.unlink()
    oof_y_pred.unlink()
    test_y_true.unlink()
    test_y_pred.unlink()
    del (test_first_step_idx, test_last_step_idx, test_y_true,
         oof_y_true, oof_first_step_idx, oof_last_step_idx, oof_y_pred, test_y_pred)


def load_params(solution_name: str) -> Tuple[DictConfig, DictConfig]:
    params_phenotyping = OmegaConf.load('params.yaml').phenotyping
    params_eval = cast(DictConfig, OmegaConf.merge(
        params_phenotyping.evaluate.main,
        params_phenotyping[solution_name].evaluate
    ))
    return params_phenotyping, params_eval


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def cross_validate(
    solution: SolutionPredict,
    solution_name: str,
    seed: int,
    cohort_train_set: xr.Dataset,
    signals_train_df: pd.DataFrame,
    cohort_test_set: xr.Dataset,
    signals_test_df: pd.DataFrame,
    params_eval: DictConfig,
    monitor: Monitor,
    test_y_true: SharedArray,
    test_first_step_idx: SampleIdx,
    test_last_step_idx: SampleIdx

) -> Tuple[SharedDataFrame, SharedDataFrame]:
    # random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    oof_seed = SharedDataFrame(
        pd.DataFrame(0.0,
                     index=signals_train_df.index,
                     columns=cohort_train_set.target.label,
                     dtype=np.float32)
    )

    test_seed = SharedDataFrame(
        pd.DataFrame(0.0,
                     index=signals_test_df.index,
                     columns=cohort_test_set.target.label,
                     dtype=np.float32)
    )

    # out-of-fold cross-validation
    for fold in tqdm.tqdm(params_eval.cv.folds):
        # oof set
        cohort_val_set = get_fold_val(cohort_train_set, fold, coord=f'fold_{seed}')
        folds = [x for x in cast(List[str], cohort_val_set.coords) if x.startswith('fold_')]
        cohort_val_set = cohort_val_set.drop_vars(folds)

        signals_val_df = cast(pd.DataFrame,
                              signals_train_df.loc[cohort_val_set.sample.values, :].copy())
        val_first_step_idx = index_of_first_step(signals_val_df)
        val_last_step_idx = index_of_last_step(signals_val_df)
        val_pred = SharedArray(solution(
            cohort_val_set,
            signals_val_df,
            seed,
            fold
        ))

        del signals_val_df
        gc.collect()

        oof_seed[cohort_val_set.sample, :] = val_pred.read()

        val_y_true = td_y_true(cohort_val_set.target.values, val_last_step_idx)
        del cohort_val_set
        gc.collect()

        # test set
        test_pred = SharedArray(solution(
            cohort_test_set.copy(deep=True),
            signals_test_df.copy(),
            seed,
            fold
        ))
        test_seed.map(lambda df: df.add(test_pred.read()))

        # scores
        val_result = metrics(
            val_y_true,
            val_pred,
            val_first_step_idx,
            val_last_step_idx,
            params_eval,
            f'cv seed={seed} fold={fold}'
        )
        monitor.log_cv_fold(
            seed,
            val_result,
            metrics(
                test_y_true,
                test_pred,
                test_first_step_idx,
                test_last_step_idx,
                params_eval,
                f'test seed={seed} fold={fold}'
            )
        )

        val_y_true.unlink()
        val_pred.unlink()
        test_pred.unlink()
        del val_y_true, val_pred, test_pred, val_first_step_idx, val_last_step_idx
        gc.collect()

        monitor.log_eval_test_correlation(
            list(val_result.keys()),
            title=f"{name_of(solution_name)}::Validation/Test Correlation"
        )

        monitor.live.make_report()

    test_seed.map(lambda df: df.div(len(params_eval.cv.folds)))

    return oof_seed, test_seed


def metrics(
    y_true: SharedArray,
    y_pred: SharedArray,
    first_step_idx: SampleIdx,
    last_step_idx: SampleIdx,
    params_eval: DictConfig,
    set_name: str,
    return_mean: bool = False,
    n_jobs: int = -1
) -> Dict[str, float]:

    def shared(metric_name: str, metric, step_idx: SampleIdx = None):
        @wraps(metric)
        def wrapper():

            inner_log = get_log(__name__)

            inner_log.info('Calculating %s on %s ...', metric_name, set_name)
            y_true_sm_ = y_true.read()
            y_pred_sm_ = y_pred.read()

            if step_idx is None:
                result = metric(y_true_sm_, y_pred_sm_)
            else:
                result = metric(y_true_sm_[step_idx, :], y_pred_sm_[step_idx, :])

            if (isinstance(result, np.ndarray) and len(result) == 1) or isinstance(result, float):
                inner_log.info('... %s = %s', metric_name, result)
            elif isinstance(result, np.ndarray) and len(result.shape) < 3:
                inner_log.info('... %s = %s', metric_name, float(np.mean(result)))
            else:
                inner_log.info('... %s', metric_name)

            return result

        return wrapper

    def result_of(val):
        if return_mean:
            return float(np.mean(val))
        return val

    log.info('Calculating td_pr_auc and td_roc_auc on %s ...', set_name)
    td_pr_auc_val, td_roc_auc_val = pr_roc_auc(y_true, y_pred, n_jobs=-1)
    log.info('... td_pr_auc = %s', float(np.mean(td_pr_auc_val)))
    log.info('... td_roc_auc = %s', float(np.mean(td_roc_auc_val)))

    td_mapk, td_cm_val = (
        Parallel(n_jobs=n_jobs)(delayed(shared(metric_name, metric))() for metric_name, metric in {
            f'td_mAP@{params_eval.map_k}': partial(map_at_k, k=params_eval.map_k),
            'td_multilabel_confusion_matrix': threshold(multilabel_confusion_matrix)
        }.items())
    )

    ls_mapk, ls_pr_roc_val, ls_roc_auc_val, ls_cm_val = (
        Parallel(n_jobs=n_jobs)(delayed(shared(metric_name, metric, last_step_idx))()
                                for metric_name, metric in {
            f'ls_mAP@{params_eval.map_k}': partial(map_at_k, k=params_eval.map_k),
            'ls_pr_auc': partial(average_precision_score, average=None),
            'ls_roc_auc': partial(roc_auc_score, average=None),
            'ls_multilabel_confusion_matrix': threshold(multilabel_confusion_matrix)
        }.items())
    )

    fs_mapk, fs_pr_roc_val, fs_roc_auc_val, fs_cm_val = (
        Parallel(n_jobs=n_jobs)(delayed(shared(metric_name, metric, first_step_idx))()
                                for metric_name, metric in {
            f'fs_mAP@{params_eval.map_k}': partial(map_at_k, k=params_eval.map_k),
            'fs_pr_auc': partial(average_precision_score, average=None),
            'fs_roc_auc': partial(roc_auc_score, average=None),
            'fs_multilabel_confusion_matrix': threshold(multilabel_confusion_matrix)
        }.items())
    )

    if (
        fs_mapk is None or
        fs_pr_roc_val is None or
        fs_roc_auc_val is None or
        fs_cm_val is None or
        ls_mapk is None or
        ls_pr_roc_val is None or
        ls_roc_auc_val is None or
        ls_cm_val is None or
        td_mapk is None or
        td_pr_auc_val is None or
        td_roc_auc_val is None or
        td_cm_val is None
    ):
        raise ValueError('Not all metrics were computed!')

    fs_tp = fs_cm_val[:, 1, 1]
    fs_fp = fs_cm_val[:, 0, 1]
    fs_fn = fs_cm_val[:, 1, 0]
    fs_tn = fs_cm_val[:, 0, 0]

    ls_tp = ls_cm_val[:, 1, 1]
    ls_fp = ls_cm_val[:, 0, 1]
    ls_fn = ls_cm_val[:, 1, 0]
    ls_tn = ls_cm_val[:, 0, 0]

    td_tp = td_cm_val[:, 1, 1]
    td_fp = td_cm_val[:, 0, 1]
    td_fn = td_cm_val[:, 1, 0]
    td_tn = td_cm_val[:, 0, 0]

    results = {
        'fs_roc_auc': fs_roc_auc_val,
        f'fs_mAP@{params_eval.map_k}': fs_mapk,
        'fs_pr_auc': fs_pr_roc_val,
        'fs_precision': div_zero(fs_tp, (fs_tp+fs_fp)),
        'fs_recall': div_zero(fs_tp, (fs_tp+fs_fn)),
        'fs_f1_score': div_zero((2*fs_tp), ((2*fs_tp)+fs_fp+fs_fn)),
        'fs_TP': fs_tp,
        'fs_FP': fs_fp,
        'fs_FN': fs_fn,
        'fs_TN': fs_tn,
        'fs_sensitivity': div_zero(fs_tp, (fs_tp+fs_fn)),
        'fs_specificity': div_zero(fs_tn, (fs_tn+fs_fp)),
        'ls_roc_auc': ls_roc_auc_val,
        f'ls_mAP@{params_eval.map_k}': ls_mapk,
        'ls_pr_auc': ls_pr_roc_val,
        'ls_precision': div_zero(ls_tp, (ls_tp+ls_fp)),
        'ls_recall': div_zero(ls_tp, (ls_tp+ls_fn)),
        'ls_f1_score': div_zero((2*ls_tp), ((2*ls_tp)+ls_fp+ls_fn)),
        'ls_TP': ls_tp,
        'ls_FP': ls_fp,
        'ls_FN': ls_fn,
        'ls_TN': ls_tn,
        'ls_sensitivity': div_zero(ls_tp, (ls_tp+ls_fn)),
        'ls_specificity': div_zero(ls_tn, (ls_tn+ls_fp)),
        'td_roc_auc': td_roc_auc_val,
        f'td_mAP@{params_eval.map_k}': td_mapk,
        'td_pr_auc': td_pr_auc_val,
        'td_precision': div_zero(td_tp, (td_tp+td_fp)),
        'td_recall': div_zero(td_tp, (td_tp+td_fn)),
        'td_f1_score': div_zero((2*td_tp), ((2*td_tp)+td_fp+td_fn)),
        'td_TP': td_tp,
        'td_FP': td_fp,
        'td_FN': td_fn,
        'td_TN': td_tn,
        'td_sensitivity': div_zero(td_tp, (td_tp+td_fn)),
        'td_specificity': div_zero(td_tn, (td_tn+td_fp))
    }

    return {k: result_of(v) for k, v in results.items()}


def name_of(func_name: str) -> str:
    func_name = sub(r"(_|-)+", " ", func_name).title()
    return ''.join(func_name)


def map_result(
    file_path: str | Path,
    func: Callable[[pd.DataFrame], pd.DataFrame],
):
    file_path = Path(file_path)
    tmp_path = file_path.parent / (file_path.name + '.tmp')
    (
        func(pd.read_parquet(file_path, engine='fastparquet'))
        .to_parquet(tmp_path, engine='fastparquet')
    )

    if tmp_path.exists():
        os.remove(file_path)
        os.rename(tmp_path, file_path)


if __name__ == '__main__':
    run()
