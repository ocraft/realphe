import csv
import json
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml

from sklearn.metrics import accuracy_score
from realphe.pipe.monitor import Monitor


def test_log_scores(tmp_path):
    seed = 15
    simulate_monitor(tmp_path, seed)

    assert_expected_dir_struct(seed, tmp_path)
    assert [['step', 'oof'], ['15', '0.75']] == get_tsv(tmp_path / 'plots/metrics/acc/oof.tsv')
    assert [['step', 'test'], ['15', '0.5']] == get_tsv(tmp_path / 'plots/metrics/acc/test.tsv')
    assert [['step', 'eval'],
            ['0', '0.5'],
            ['1', '1.0']] == get_tsv(tmp_path / f'{seed}/plots/metrics/acc/eval.tsv')
    assert [['step', 'test'],
            ['0', '0.25'],
            ['1', '0.25']] == get_tsv(tmp_path / f'{seed}/plots/metrics/acc/test.tsv')

    assert {
        'acc': {
            'oof': {
                'min': 0.75,
                'max': 0.75,
                'mean': 0.75,
                'var': 0.0,
                'agg': 0.75,
                'ci': 0.0
            },
            'val': {
                'min': 0.5,
                'max': 1.0,
                'mean': 0.75,
                'var': 0.125,
                'ci': 3.1765511840436735,
            },
            'test': {
                'min': 0.25,
                'max': 0.25,
                'mean': 0.25,
                'var': 0.0,
                'agg.cv': {
                    'max': 0.5,
                    'mean': 0.5,
                    'min': 0.5,
                    'var': 0.0,
                    'ci': 0.0
                },
                'agg': 0.5,
                'ci': 0.0
            }
        },
        'step': 15
    } == get_json(tmp_path / 'metrics.json')
    assert {
        'acc': {
            'oof': 0.75,
            'val': {
                'min': 0.5,
                'max': 1.0,
                'mean': 0.75,
                'var': 0.125,
                'ci': 3.1765511840436735,
            },
            'test': {
                'min': 0.25,
                'max': 0.25,
                'mean': 0.25,
                'var': 0.0,
                'agg': 0.5,
                'ci': 0.0
            }

        },
        'step': 1
    } == get_json(tmp_path / f'{seed}/metrics.json')


def simulate_monitor(tmp_path, seed):
    os.environ['DVC_STUDIO_OFFLINE'] = 'true'
    monitor = Monitor(eval_path=tmp_path, dvcyaml=None)
    test_set = np.array([[1, 0],
                         [0, 1],
                         [0, 1],
                         [1, 0]])

    # step for each seed: cv
    # step for each fold in cv: train:val:test

    def compute(y_true, y_pred):
        return {
            'acc': float(accuracy_score(y_true, y_pred))
        }

   # cv fold 0
    monitor.log_cv_fold(
        seed,
        compute(
            np.array([[0, 1],
                      [1, 0]]),
            np.array([[1, 0],
                      [1, 0]])
        ),
        compute(
            test_set,
            np.array([[1, 0],
                      [1, 0],
                      [1, 0],
                      [0, 1]])
        )
    )
    # cv fold 1
    monitor.log_cv_fold(
        seed,
        compute(
            np.array([[1, 0],
                      [1, 0]]),
            np.array([[1, 0],
                      [1, 0]])
        ),
        compute(
            test_set,
            np.array([[0, 1],
                      [0, 1],
                      [1, 0],
                      [0, 1]])
        )
    )

    monitor.log_cv(
        seed,
        compute(
            np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [1, 0]]),
            np.array([[1, 0],
                      [1, 0],
                      [1, 0],
                      [1, 0]])
        ),
        compute(
            test_set,
            np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [0, 1]])
        )
    )

    # finish after all seeds
    monitor.end(
        compute(
            np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [1, 0]]),
            np.array([[1, 0],
                      [1, 0],
                      [1, 0],
                      [1, 0]])
        ),
        compute(
            test_set,
            np.array([[1, 0],
                      [0, 1],
                      [1, 0],
                      [0, 1]])
        )
    )
    return monitor


def assert_expected_dir_struct(seed, tmp_path):
    expected_eval_dir_struct = {
        '': {
            '.': {
                'metrics.json': None,
                'report.html': None
            },
            'plots': {},
            f'{seed}': {
                'metrics.json': None,
                'report.html': None
            },
        },
        'plots': {'metrics': {}},
        'plots/metrics': {
            'acc': {
                'oof.tsv': None,
                'test.tsv': None,
            }
        },
        f'{seed}': {'plots': {}},
        f'{seed}/plots': {'metrics': {}},
        f'{seed}/plots/metrics': {
            'acc': {
                'eval.tsv': None,
                'test.tsv': None,
            }
        },
    }

    assert expected_eval_dir_struct == get_dir_struct(tmp_path)


DirStruct = Dict[str, Union['DirStruct', None]]


def get_dir_struct(start_path: Path) -> DirStruct:
    directory_structure = {}
    for root, _, files in os.walk(start_path):
        current_level = directory_structure
        root = os.path.relpath(root, start_path)
        for part in os.path.split(root):
            current_level = current_level.setdefault(part, {})
        for file in files:
            current_level[file] = None
    return directory_structure


def get_tsv(file_path: Path):
    with open(file_path, encoding='utf-8') as file:
        data = []
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            data.append(row)

        return data


def get_json(file_path: Path):
    with open(file_path, encoding='utf-8') as file:
        return json.load(file)


def get_yaml(file_path: Path):
    with open(file_path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def test_learning_curve(tmp_path):
    monitor = simulate_monitor(tmp_path / 'eval', 1)

    Path(tmp_path / 'train/1/fold_0/plots/metrics/train').mkdir(parents=True, exist_ok=True)
    with open(tmp_path / 'train/1/fold_0/plots/metrics/train/auc.tsv', 'w', encoding='utf-8') as file:
        tsv_output = csv.writer(file, delimiter='\t')
        tsv_output.writerow(['step', 'auc'])
        tsv_output.writerow(['0', '0.5'])
        tsv_output.writerow(['1', '0.7'])
        tsv_output.writerow(['2', '0.6'])

    Path(tmp_path / 'train/1/fold_1/plots/metrics/train').mkdir(parents=True, exist_ok=True)
    with open(tmp_path / 'train/1/fold_1/plots/metrics/train/auc.tsv', 'w', encoding='utf-8') as file:
        tsv_output = csv.writer(file, delimiter='\t')
        tsv_output.writerow(['step', 'auc'])
        tsv_output.writerow(['0', '1.0'])
        tsv_output.writerow(['1', '0.7'])

    monitor.log_learning_curve(['auc'], ['train'])

    assert get_json(tmp_path / 'eval/plots/custom/learning_curve/auc.json') == [
        {
            "step": 0.0,
            "auc": 0.75,
            "std_l": 0.5,
            "std_h": 1.0,
            "data": "train"
        },
        {
            "step": 1.0,
            "auc": 0.7,
            "std_l": 0.7,
            "std_h": 0.7,
            "data": "train"
        },
        {
            "step": 2.0,
            "auc": 0.6,
            "std_l": 0.6,
            "std_h": 0.6,
            "data": "train"
        }
    ]


def test_eval_test_correlation(tmp_path):
    monitor = simulate_monitor(tmp_path / 'eval', 1)

    Path(tmp_path / 'eval/1/plots/metrics/auc').mkdir(parents=True, exist_ok=True)
    with open(tmp_path / 'eval/1/plots/metrics/auc/eval.tsv', 'w', encoding='utf-8') as file:
        tsv_output = csv.writer(file, delimiter='\t')
        tsv_output.writerow(['step', 'auc'])
        tsv_output.writerow(['0', '0.5'])
        tsv_output.writerow(['1', '0.7'])
        tsv_output.writerow(['2', '0.6'])

    Path(tmp_path / 'eval/1/plots/metrics/auc').mkdir(parents=True, exist_ok=True)
    with open(tmp_path / 'eval/1/plots/metrics/auc/test.tsv', 'w', encoding='utf-8') as file:
        tsv_output = csv.writer(file, delimiter='\t')
        tsv_output.writerow(['step', 'auc'])
        tsv_output.writerow(['0', '1.0'])
        tsv_output.writerow(['1', '0.7'])
        tsv_output.writerow(['1', '0.7'])

    monitor.log_eval_test_correlation(['auc'])

    assert get_json(tmp_path / 'eval/plots/custom/eval_test_corr/auc.json') == [
        {
            "seed": 1.0,
            "eval": 0.5,
            "test": 1.0
        },
        {
            "seed": 1.0,
            "eval": 0.7,
            "test": 0.7
        },
        {
            "seed": 1.0,
            "eval": 0.6,
            "test": 0.7
        }
    ]
