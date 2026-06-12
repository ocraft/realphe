import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from dvclive.live import Live
from dvclive.live import ParamLike
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from .metrics import confidence_interval


Result = Mapping[str, float | List[float]]


class Monitor:
    def __init__(self,
                 eval_path: Path | str,
                 resume: bool = False,
                 dvcyaml: Optional[str] = 'dvc.yaml'):
        self._live_cv: Optional[Live] = None
        self._eval_path = str(eval_path)
        self._resume = resume
        self.start(eval_path, resume, dvcyaml)

    def start(self,
              eval_path: Path | str,
              resume: bool = False,
              dvcyaml: Optional[str] = 'dvc.yaml'):
        track_path = f'{self._eval_path}/_track.json'
        if resume and os.path.exists(track_path):
            with open(track_path, encoding='utf-8') as file:
                self._track = json.load(file)
        else:
            self._track = {}
        self._live_seed = Live(dir=str(eval_path),
                               dvcyaml=dvcyaml,
                               report='html',
                               resume=resume,
                               save_dvc_exp=False)
        self._live_cv = None

    @property
    def live(self):
        return self._live_seed

    def log_params(self, params):
        self._live_seed.log_params(Monitor.params_to_log(params))

    @staticmethod
    def params_to_log(params) -> Dict[str, ParamLike]:
        resolved_params = OmegaConf.to_container(params, resolve=True)
        if resolved_params is None or isinstance(resolved_params, (List, str)):
            raise ValueError('Invalid parameters')
        return {str(key): value for key, value in resolved_params.items()}

    def log_cv_fold(self, seed: int, val: Result, test: Optional[Result] = None):
        with_mean = isinstance(val[next(iter(val))], float)
        if seed not in self._track:
            self._live_cv = Live(dir=f'{self._eval_path}/{seed}',
                                 dvcyaml=None,
                                 report='html',
                                 resume=False,
                                 save_dvc_exp=False)
            self._track[seed] = {}
            for metric_name in val:
                self._track[seed][metric_name] = {
                    'seed': seed,
                    'oof': None,
                    'test': {
                        'agg': None,
                        'scores': []
                    },
                    'val': {
                        'scores': []
                    }
                }

        if self._live_cv is None:
            raise ValueError('Internal state error: self._live_fold should be initialized')

        for metric_name, val_score in val.items():
            if not with_mean:
                val_score = float(np.mean(val_score))
            self._track[seed][metric_name]['val']['scores'].append(val_score)
            self._live_cv.log_metric(f'{metric_name}/eval', cast(float, val_score))

            if test is not None:
                test_score = test[metric_name]
                if not with_mean:
                    test_score = float(np.mean(test_score))
                else:
                    test_score = cast(float, test_score)
                self._track[seed][metric_name]['test']['scores'].append(test_score)
                self._live_cv.log_metric(f'{metric_name}/test', test_score)

        self._live_cv.next_step()

        if not with_mean:
            fold = self._live_cv.step-1
            fold_metrics_path = f'{self._eval_path}/{seed}/fold_{fold}'
            os.makedirs(f'{fold_metrics_path}/eval', exist_ok=True)
            pd.DataFrame(val).to_csv(f'{fold_metrics_path}/eval/metrics.csv')
            if test is not None:
                os.makedirs(f'{fold_metrics_path}/test', exist_ok=True)
                pd.DataFrame(test).to_csv(f'{fold_metrics_path}/test/metrics.csv')

    def log_cv(self, seed: int, oof: Result, test: Optional[Result] = None):
        with_mean = isinstance(oof[next(iter(oof))], float)
        if self._live_cv is None:
            raise ValueError('Internal state error: self._live_cv should be initialized')

        for metric_name, metric_score in oof.items():
            if not with_mean:
                metric_score = float(np.mean(metric_score))
            self._track[seed][metric_name]['oof'] = metric_score
            if test:
                test_score = test[metric_name]
                if not with_mean:
                    test_score = float(np.mean(test_score))
                self._track[seed][metric_name]['test']['agg'] = test_score

        agg_metrics = {}
        for metric_name in oof:
            agg_metrics[metric_name] = {}
            agg_metrics[metric_name]['oof'] = self._track[seed][metric_name]['oof']
            agg_metrics[metric_name]['val'] = (
                Monitor.summarize(self._track[seed][metric_name]['val']['scores'])
            )

            if test:
                agg_metrics[metric_name]['test'] = (
                    Monitor.summarize(self._track[seed][metric_name]['test']['scores'])
                )
                agg_metrics[metric_name]['test']['agg'] = (
                    self._track[seed][metric_name]['test']['agg']
                )

        self._live_cv.summary.update(agg_metrics)
        self._live_cv.end()

        if self._live_seed is not None:
            self._live_seed.step = seed
            for metric_name in oof:
                self._live_seed.log_metric(
                    f'{metric_name}/oof', self._track[seed][metric_name]['oof'])
                if test:
                    self._live_seed.log_metric(f'{metric_name}/test',
                                               self._track[seed][metric_name]['test']['agg'])
            self._live_seed.next_step()

            track_path = f'{self._eval_path}/_track.json'
            with open(track_path, 'w', encoding='utf-8') as file:
                json.dump(self._track, file)

        if not with_mean:
            seed_metrics_path = f'{self._eval_path}/{seed}'
            os.makedirs(f'{seed_metrics_path}/oof', exist_ok=True)
            pd.DataFrame(oof).to_csv(f'{seed_metrics_path}/oof/metrics.csv')
            if test is not None:
                os.makedirs(f'{seed_metrics_path}/test', exist_ok=True)
                pd.DataFrame(test).to_csv(f'{seed_metrics_path}/test/metrics.csv')

    @staticmethod
    def summarize(score_list: List[float]):
        ci = confidence_interval(score_list)
        return {
            'mean': np.mean(score_list) if score_list else None,
            'var': (np.var(score_list, ddof=1)
                    if len(score_list) > 1 else 0.0
                    if len(score_list) == 1 else None),
            'min': np.min(score_list) if score_list else None,
            'max': np.max(score_list) if score_list else None,
            'ci': ci
        }

    def end(self, oof: Optional[Result] = None, test: Optional[Result] = None):
        agg_metrics = self._aggregate_of_seeds()
        if oof is not None:
            with_mean = isinstance(oof[next(iter(oof))], float)
            for metric_name, metric_score in oof.items():
                if not with_mean:
                    metric_score = float(np.mean(metric_score))
                agg_metrics[metric_name]['oof']['agg'] = metric_score
            if not with_mean:
                os.makedirs(f'{self._eval_path}/oof', exist_ok=True)
                pd.DataFrame(oof).to_csv(f'{self._eval_path}/oof/metrics.csv')

        if test is not None:
            with_mean = isinstance(test[next(iter(test))], float)
            for metric_name, metric_score in test.items():
                if not with_mean:
                    metric_score = float(np.mean(metric_score))
                agg_metrics[metric_name]['test']['agg'] = metric_score
            if not with_mean:
                os.makedirs(f'{self._eval_path}/test', exist_ok=True)
                pd.DataFrame(test).to_csv(f'{self._eval_path}/test/metrics.csv')

        self._live_seed.summary.update(agg_metrics)
        self._live_seed.end()

        track_path = f'{self._eval_path}/_track.json'
        if os.path.exists(track_path):
            os.remove(track_path)
        return agg_metrics

    def _aggregate_of_seeds(self) -> Dict[str, Dict[str, Any]]:
        all_metrics = {}
        tr = self._track
        for metric_name in tr[list(tr.keys())[0]].keys():
            all_metrics[metric_name] = {
                'oof': [m[metric_name]['oof'] for m in tr.values()],
                'test': {
                    'agg': [m[metric_name]['test']['agg'] for m in tr.values()
                            if m[metric_name]['test']['agg'] is not None],
                    'scores': list(itertools.chain(*[m[metric_name]['test']['scores']
                                                     for m in tr.values()
                                                     if m[metric_name]['test']['agg'] is not None]))
                },
                'val': {
                    'scores': list(itertools.chain(*[m[metric_name]['val']['scores']
                                                     for m in tr.values()]))
                }
            }

        agg_metrics = {}
        for metric_name, metric in all_metrics.items():
            agg_metrics[metric_name] = {
                'oof': Monitor.summarize(metric['oof']),
                'val': Monitor.summarize(metric['val']['scores']),
            }
            if metric['test']['scores']:
                agg_metrics[metric_name]['test'] = Monitor.summarize(metric['test']['scores'])
                agg_metrics[metric_name]['test']['agg.cv'] = (
                    Monitor.summarize(metric['test']['agg']) if metric['test']['agg'] else None
                )

        return agg_metrics

    def log_learning_curve(
        self,
        metrics: List[str],
        data_list: List[str] = ['train', 'eval'],
        title: str = 'Learning Curve'
    ):
        root_dir = f'{self._eval_path}/../train'
        for metric in metrics:
            plots = []
            for data_item in data_list:
                n_steps = 0
                datas = []

                for seed in next(os.walk(root_dir))[1]:
                    seed_path = f'{root_dir}/{seed}'
                    folds = len(next(os.walk(seed_path))[1])
                    for fold in range(folds):
                        datafile = (f'{root_dir}/{seed}/fold_{fold}/plots/metrics/'
                                    f'{data_item}/{metric.replace('_', '/')}.tsv')
                        data = np.loadtxt(fname=datafile, delimiter="\t", skiprows=1).reshape(-1, 2)

                        n_steps = max(n_steps, data.shape[0])
                        datas.append(data)

                plot = np.stack([np.pad(data[:, 1],
                                        (0, n_steps - data.shape[0]),
                                        constant_values=np.nan)
                                 for data in datas],
                                axis=0)
                mu = np.nanmean(plot, axis=0)
                std = np.nanstd(plot, axis=0)
                plot = np.empty(shape=(n_steps, 4))
                plot[:, 0] = list(range(n_steps))
                plot[:, 1] = mu
                plot[:, 2] = mu-std
                plot[:, 3] = mu+std
                plot_df = pd.DataFrame(plot, columns=['step', metric, 'std_l', 'std_h'])
                plot_df['data'] = data_item
                plots.append(plot_df)

            self.live.log_plot(
                f"learning_curve/{metric}",
                pd.concat(plots),
                x='step',
                y=metric,
                title=f'{title}::{metric}',
                template="conf/template/learning_curve.json"
            )

    def log_eval_test_correlation(
            self,
            metrics: List[str],
            title: str = 'Validation/Test Correlation'):
        for metric in metrics:
            plot_df = pd.DataFrame(columns=['seed', 'eval', 'test'], dtype=np.float64)

            for seed in self._track:
                datafile = (f'{self._eval_path}/../eval/{seed}/plots/metrics/'
                            f'{metric}')
                data_eval = np.loadtxt(fname=f'{datafile}/eval.tsv',
                                       delimiter="\t",
                                       dtype=np.float64,
                                       skiprows=1).reshape(-1, 2)
                data_test = np.loadtxt(fname=f'{datafile}/test.tsv',
                                       delimiter="\t",
                                       dtype=np.float64,
                                       skiprows=1).reshape(-1, 2)

                data_eval[:, 0] = seed
                data_test[:, 0] = seed
                plot_df = pd.concat([
                    plot_df,
                    pd.DataFrame(data=np.concatenate((data_eval, data_test[:, 1].reshape(-1, 1)),
                                                     axis=1),
                                 columns=plot_df.columns, dtype=np.float64)
                ], axis=0, ignore_index=True)

            self.live.log_plot(
                f"eval_test_corr/{metric}",
                plot_df,
                x='eval',
                y='test',
                title=f'{title}::{metric}',
                template="conf/template/scatter.json"
            )
