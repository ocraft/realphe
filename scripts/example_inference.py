"""Apply an already trained fold-specific RealPhe model to the prepared test set.

Run after the complete DVC workflow has produced the processed evaluation data
and trained models, for example after ``make install``.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from realphe.task.phenotyping.baseline.load import pipeline_single
from realphe.task.phenotyping.baseline.setup import get_eval_set


params = OmegaConf.load("params.yaml").phenotyping
_, _, cohort_test, signals_test = get_eval_set(params, "baseline")

device = torch.device("cuda")

phenotype_probabilities = pipeline_single(
    cohort_set=cohort_test,
    signals_df=signals_test,
    seed=100,
    fold=0,
    device=device,
    batch_size=512,
)

assert phenotype_probabilities.ndim == 2
assert np.isfinite(phenotype_probabilities).all()
assert ((phenotype_probabilities >= 0) & (phenotype_probabilities <= 1)).all()

print(f"Device: {device}")
print(f"Phenotype-probability matrix: {phenotype_probabilities.shape}")
