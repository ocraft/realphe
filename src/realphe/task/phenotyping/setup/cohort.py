import os
import time

from omegaconf import OmegaConf

from realphe.data.ehr.mimiciv.query import MimicDb
from realphe.data.ehr.mimiciv.shape import CohortSetup, cohort, diagnoses, set_cohort
from realphe.env import get_log


log = get_log(__name__)


def run():
    # load params
    params = OmegaConf.load('params.yaml')
    os.makedirs(params.phenotyping.path.data.interim + '/setup/', exist_ok=True)
    mimic_db = MimicDb(**params.db)

    # load cohort
    log.info('Loading cohort ...')
    time_start = time.perf_counter()
    dataset = cohort(mimic_db, CohortSetup(**params.phenotyping.cohort))
    log.info('... finished loading cohort in [%d]s.', time.perf_counter() - time_start)

    # load diagnoses
    log.info('Loading diagnoses ...')
    time_start = time.perf_counter()
    with set_cohort(mimic_db, dataset):
        dataset = dataset.merge(diagnoses(mimic_db, params.phenotyping.cohort))
    log.info('... finished loading diagnoses in [%d]s.', time.perf_counter() - time_start)

    # save prepared cohort
    dataset.to_zarr(f'{params.phenotyping.path.data.interim}/setup/cohort.zarr',
                    consolidated=False,
                    mode='w')


if __name__ == '__main__':
    run()
