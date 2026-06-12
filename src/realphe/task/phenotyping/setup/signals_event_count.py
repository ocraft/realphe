import glob
import math
import os
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
import xarray as xr

from realphe.data.ehr.mimiciv.query import MimicDb
from realphe.data.ehr.mimiciv.shape import set_cohort, signals_event_count
from realphe.env import get_log


log = get_log(__name__)


N_CHUNK = 300000


def run():
    log.info('Loading signals event count ...')
    time_start = time.perf_counter()

    # load params
    params = OmegaConf.load('params.yaml')
    in_path = params.phenotyping.path.data.interim + '/setup/'
    out_path = params.phenotyping.path.data.processed + '/setup/'
    os.makedirs(in_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # load cohort
    mimic_db = MimicDb(**params.db)
    cohort = xr.open_dataset(f'{in_path}/cohort.zarr', consolidated=False)

    n_chunks = math.ceil(len(cohort.sample) / N_CHUNK)

    # load signals
    for i, chunk in enumerate(np.array_split(cohort.sample, n_chunks)):
        log.info('Chunk [%d]; Samples [%d]', i, len(chunk))
        with set_cohort(mimic_db, cohort.sel({'sample': chunk})):
            chunk_set = signals_event_count(mimic_db,
                                            params.phenotyping.signals)
            chunk_set.to_zarr(f'{out_path}/signals_event_count-{i}.zarr',
                              consolidated=False,
                              mode='w')

    combined = xr.open_mfdataset(
        f'{out_path}/signals_event_count-*.zarr',
        concat_dim='sample',
        combine='nested',
        parallel=True,
        consolidated=False
    )
    combined.to_zarr(f'{out_path}/signals_event_count.zarr', consolidated=False, mode='w')

    # remove temporary chunks
    for chunk in glob.glob(f'{out_path}/signals_event_count-*.zarr'):
        shutil.rmtree(chunk)

    log.info('... finished loading signals event count in [%d]s.',
             time.perf_counter() - time_start)


if __name__ == '__main__':
    run()
