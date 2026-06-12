import os

import numpy as np
from omegaconf import OmegaConf
import xarray as xr

from realphe.env import get_log
from realphe.pipe.splitter import multilabel_train_test_split


log = get_log(__name__)


def run():
    # load params
    params = OmegaConf.load('params.yaml')
    data_path_in = params.phenotyping.path.data.interim + '/setup/'
    data_path_out = params.phenotyping.path.data.processed + '/setup/'
    os.makedirs(data_path_in, exist_ok=True)
    os.makedirs(data_path_out, exist_ok=True)

    # load datasets
    cohort = xr.open_dataset(f'{data_path_in}/cohort.zarr', consolidated=False).load()

    # split dataset to train and test
    train_samples, test_samples = multilabel_train_test_split(
        cohort,
        **params.phenotyping.test_set,
        target_var='target',
        return_index=True)

    cohort['split'] = xr.DataArray(np.zeros(len(cohort.sample), dtype='int8'), dims=['sample'])
    cohort['split'].loc[{'sample': test_samples}] = 1

    cohort.to_zarr(f'{data_path_out}/cohort.zarr', consolidated=False, mode='w')

    log.info('Train set has [%d] samples, Test set has [%d] samples.',
             len(train_samples), len(test_samples))


if __name__ == '__main__':
    run()
