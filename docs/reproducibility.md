# Software validation and reproducibility

Reliability is evaluated at complementary levels. The baseline model evaluation repeats 10-fold multilabel-stratified
cross-validation across seven random seeds; each seed determines a fold assignment and the model initialization. This
design reduces dependence of reported performance on a particular partition or initialization. Software/workflow
reliability is supported by deterministic execution controls, pinned dependencies, automated tests, DVC provenance, and
the second-system reproduction documented below. These checks characterize reliability of the evaluated research
workflow and should not be interpreted as evidence of clinical reliability, cross-institutional generalizability, or
prospective deployment performance.

RealPhe v1.0.1 was validated at the software level by rerunning the complete workflow on a computational setup different
from the system used for the associated methodological study [^1]. Both runs used MIMIC-IV 2.0 and the same experimental
formulation. Because the published reference used TensorFlow-based v0.1.0 [^2] whereas the reproduction uses
PyTorch-based v1.0.1, this validation is an end-to-end reproduction of the published reference results across both
implementation and environment changes; it is not a bitwise same-version repeatability test.

## Computational setups

| | Original study | RealPhe v1.0.0 reproduction |
|---|---|---|
| Operating environment | WSL2 under Windows | Ubuntu 24.04.5 LTS, Linux 7.0.0-31-generic |
| CPU | Intel Core i7 | AMD Ryzen 7 9700X |
| RAM | 32 GB | 64 GiB |
| GPU | NVIDIA GeForce RTX 3060, 12 GB VRAM | NVIDIA GeForce RTX 5060 Ti |
| ML implementation | TensorFlow/Keras | PyTorch |
| Python | 3.10.8 (archived v0.1.0 environment) | 3.12.13 (v1.0.0 lock) |
| Framework version | TensorFlow 2.11.1 (archived v0.1.0 environment) | PyTorch 2.11.0+cu128 |

## Software execution characteristics

Software-level timing and memory measurements were obtained on the RealPhe reproduction system. A representative
fold-specific training stage was run for seed 100, fold 0. The stage used 46,264 training and 5,179 validation ICU stays,
with sequences of up to 168 time steps, 484 model-input features, and 561 phenotype targets. The complete invoked training
stage required **74.2 s**. Peak sampled aggregate process-tree proportional set size (PSS) was **16.3 GiB**, and peak
sampled GPU-memory use was **1.84 GiB**. The benchmark uses a default 0.2-s polling delay. PSS is reported instead of the
sum of per-process RSS to avoid double-counting memory pages shared with worker processes. The timed stage includes data
loading, fold-specific preprocessing, optimization, model serialization, and experiment logging.

For inference, a trained seed-100/fold-0 model processed the complete test cohort of 13,045 ICU stays and 436,706 valid
2-hour time steps with batch size 512 in a median of **1.20 s** across 10 runs (range **1.19-1.25 s**), corresponding to
approximately **364,000 valid time steps/s**. Initial dataset loading and preprocessing were excluded from inference timing;
batch collation and host-to-device transfer were included. The training and inference benchmarks can be reproduced with
`scripts/benchmark_training.sh` and `scripts/benchmark_inference.py`, respectively.

## Metric aggregation across phenotypes

Final test predictions are averaged across the 10 fold-specific models within each seed and then across the seven configured
seeds. AUC-ROC and AUC-PR are calculated separately for each of the 561 diagnostic phenotypes and reported as unweighted
macro-means. In the reproduced test set, all 561 phenotypes contained both positive and negative examples in every reported
evaluation context. The minimum counts were 75 positives and 7,638 negatives for first-step/last-step evaluation and 1,893
positives and 260,058 negatives for time-distributed evaluation. Consequently, all 561 phenotype-level AUC values were
defined and included in each macro-mean; no special undefined-label handling or exclusion was required. mAP@10 is not a
label-wise macro-average: it calculates average precision at 10 for each evaluated row from the ranked 561 phenotype
predictions and then averages these row-level values.

## Lightweight smoke test

A MIMIC-IV-independent smoke test is available through `make test-smoke`. It constructs three synthetic variable-length
trajectories of lengths 3, 2, and 4 with three input features and two phenotype targets. The expected prediction tensor
therefore contains 9 valid time-step rows and 2 target columns. The test verifies the `(9, 2)` output shape, finite values,
preserved packed-sequence structure, and numerical agreement before and after saving/reloading `model.pt`. In the locked
Python 3.12.13 environment, the standalone smoke-test command completed with `1 passed`. These explicit structural and
round-trip checks provide expected behavior without relying on a device-specific floating-point checksum.

## Principal phenotyping metrics

The published values are the test-set ensemble results reported in the associated Expert Systems with Applications
study [^1]. Reproduced values are the aggregate test metrics generated by the RealPhe v1.0.1 workflow.

| Context | Metric | Published | Reproduced | Absolute difference |
|---|---|---:|---:|---:|
| First step | mAP@10 | 0.160 | 0.156 | 0.004 |
| First step | AUC-PR | 0.109 | 0.106 | 0.003 |
| First step | AUC-ROC | 0.777 | 0.776 | 0.001 |
| Time-distributed | mAP@10 | 0.211 | 0.204 | 0.007 |
| Time-distributed | AUC-PR | 0.177 | 0.175 | 0.002 |
| Time-distributed | AUC-ROC | 0.829 | 0.828 | 0.001 |
| Last step | mAP@10 | 0.226 | 0.218 | 0.008 |
| Last step | AUC-PR | 0.184 | 0.179 | 0.005 |
| Last step | AUC-ROC | 0.843 | 0.842 | 0.001 |

Because the historical implementation [^2] used a different machine-learning stack and computational environment, and
the published reference values are available to three decimal places, an absolute difference of 0.01 on metrics bounded
between 0 and 1 is used here as a transparent practical tolerance rather than requiring bitwise identity. All nine
reported differences satisfy this criterion. Under this criterion, agreement across all three evaluation contexts and
all three principal metrics supports preservation of the predictive behavior of the associated study [^1] by the v1.0.1
PyTorch implementation within the stated tolerance. Using the full-precision RealPhe v1.0.1 outputs against the
published three-decimal reference values, the largest difference is approximately 0.0082 (last-step mAP@10). The
corresponding AUC-ROC differences are approximately 0.0006, 0.0012, and 0.0013 for first-step, time-distributed, and
last-step evaluation, respectively.

## Deterministic execution and ensemble behavior

The historical TensorFlow implementation [^2] set random seeds, while the PyTorch implementation adds stricter
deterministic execution controls. In v1.0.1, the training stack sets Python, NumPy, and PyTorch seeds, disables cuDNN
benchmarking, and requests deterministic PyTorch algorithms. These controls reduce incidental stochastic variation
between independently trained models. Across the seven random seeds in the reproduction, the seed-specific 10-fold
ensemble metrics were tightly clustered; the largest performance range among the nine principal metrics was
approximately 0.0014.

Single-model performance remains similar to that of the published TensorFlow implementation, but the lower variance
between PyTorch models reduces the additional gain obtained from ensemble averaging. Consequently, several final
multi-seed ensemble metrics are modestly lower than the published TensorFlow values, although all remain within the
stated reproducibility tolerance. The published-versus-reproduced metric differences above therefore reflect
preservation of predictive behavior within the specified tolerance rather than exact numerical identity.

## Source of the published reference

[^1]: P. Picheta and S. Deniziak, “Optimizing real-time phenotyping in critical care using machine learning on electronic health records,” *Expert Systems with Applications*, vol. 320, 132084, 2026. DOI: 10.1016/j.eswa.2026.132084.

[^2]: P. Picheta and S. Deniziak, “Supplementary Materials for Optimizing Real-Time Phenotyping in Critical Care Using Machine Learning on Electronic Health Records,” Mendeley Data, version 2, Mar. 2026, doi: 10.17632/n4jn62rh2m.2. https://data.mendeley.com/datasets/n4jn62rh2m/2
