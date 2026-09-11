# Real-time Phenotyping Framework

This repository provides a reproducible workflow for real-time computational phenotyping in critical care using machine
learning on electronic health record (EHR) data [^1]. It contains utilities for downloading and preparing MIMIC-IV data,
constructing clinical concepts [^2], extending the prepared dataset with derived features, running the
experimental pipeline, and comparing experiment outputs with DVC/DVCLive [^3].

The framework is intended for research workflows where the same data-processing and model-evaluation procedure must be
repeatable across runs on the same computational setup. In RealPhe, real-time phenotyping means causal phenotype
estimation that can be updated whenever a new clinical event becomes available: each prediction uses only information
available at that time. For model development and retrospective evaluation, irregular EHR events are aggregated into
2-hour intervals; this discretization defines the training and evaluation representation rather than the intended
frequency of inference updates. Static cohort covariates are available from the beginning of the ICU stay, while
time-varying inputs contain only observations available up to the prediction time. The current software evaluates this
setting offline on retrospectively constructed EHR trajectories and does not implement a live streaming or bedside
deployment interface.

## Relation to the associated methodological study

An proof-of-concept implementation, RealPhe v0.1.0, supported the associated methodological study [^1] and was already
publicly available in the archived supplementary materials [^4]. That study introduced and evaluated the real-time
phenotyping methodology. RealPhe provides the standalone software implementation of this methodology together with the
data-processing, training, evaluation, interpretability, and experiment-tracking workflow required to reproduce and
extend the experiments.

Relative to v0.1.0, the v1.0 series preserves the published experimental formulation while substantially re-engineering
the software. The TensorFlow/Keras model stack was replaced with PyTorch, a reusable training layer was introduced under
`realphe.engine`, workflow utilities were integrated into the `realphe` package namespace, and the main multidimensional
pipeline artifacts were migrated from NetCDF to Zarr. These architectural changes were introduced in v1.0.0 and are
retained in revised v1.0.1. Version 1.0.1 further extends automated testing, benchmarking, and documentation, and aligns
model-artifact logging with the current PyTorch model path without changing the phenotyping model computation.

## Main features

- Download and prepare MIMIC-IV 2.0 data from PhysioNet.
- Build and extend EHR-derived clinical concepts for phenotyping experiments.
- Run a reproducible machine-learning pipeline for real-time ICU phenotyping.
- Track experiment parameters, metrics, and plots with DVC/DVCLive.
- Generate evaluation outputs under `eval/`, including metrics, parameter files, plots, and an HTML report.
- Compare repeated experimental runs using the standard DVC/DVCLive tooling.

## Requirements

Install the following tools before running the pipeline:

- [Docker](https://docs.docker.com/get-docker/)
- [uv](https://docs.astral.sh/uv/getting-started/installation)
- [GNU Make](https://www.gnu.org/software/make/manual/make.html)

Recommended hardware:

- At least 32 GB RAM
- At least 150 GB free disk space
- NVIDIA GPU with at least 4 GB VRAM

The workflow is designed for a Unix-like shell environment. Linux is recommended, especially for GPU-enabled execution.

## Data access

This project uses [MIMIC-IV 2.0](https://physionet.org/content/mimiciv/2.0/). Access to MIMIC-IV requires a PhysioNet account with approved credentialed access.

Before downloading the data, set the following environment variables:

```bash
export PHYSIONET_USER="your_physionet_username"
export PHYSIONET_PASSWORD="your_physionet_password"
```

These variables are required by the data-download target. Do not commit credentials to the repository.

### Data privacy and deployment scope

RealPhe currently uses a centralized research workflow. After credentialed access, MIMIC-IV is processed locally within
the user's research environment. The software does not implement privacy-preserving cross-institutional training or
patient-level data-sharing protocols, including federated learning, differential privacy, or secure aggregation.
Privacy-preserving distributed training is a potential direction for multicenter studies and would require dedicated
implementation, governance, and validation.

## Setup

### 1. Download and register raw data

```bash
make physionet-download DB=MIMICIV_20
uv run dvc add data/raw/physionet.org
```

The `DB=MIMICIV_20` argument selects the MIMIC-IV 2.0 configuration. Keep this value unless you intentionally use
another supported database configuration.

The `dvc add` command registers the downloaded raw PhysioNet data with DVC. This helps keep large data files outside
regular Git history while preserving reproducibility of the data state used by the pipeline.

### 2. Prepare the dataset

Run the preprocessing targets in order:

```bash
make data-setup DB=MIMICIV_20
make data-concepts DB=MIMICIV_20
make data-extend DB=MIMICIV_20
```

These stages perform the main data-preparation workflow:

1. `data-setup` prepares the local database/data layout required by the project.
2. `data-concepts` constructs the clinical concepts used by the phenotyping pipeline.
3. `data-extend` generates extended or derived data representations required by downstream experiments.

Depending on the machine and storage backend, these steps may take a long time and require substantial disk space.

## Running the framework

### 1. Start infrastructure services

If the local Docker network does not already exist, create it once:

```bash
docker network create -d bridge local-network
```

Then start the required services:

```bash
docker compose up
```

Keep the services running while executing pipeline steps that depend on them.

### 2. Run the pipeline

After data preparation and infrastructure startup, run:

```bash
make install
```

This command installs the project environment and runs all experiments. Only steps affected by changes are executed.
## Outputs

The main experiment and evaluation outputs are written under `eval/`. This directory contains DVCLive-compatible outputs, including:

- metrics,
- parameters,
- plots,
- generated reports, including `report.html` when produced by the evaluation workflow.

Because the outputs follow the DVC/DVCLive format, repeated runs can be compared using DVC/DVCLive functionality. This
is useful for comparing model variants, random seeds, cross-validation folds, and repeated experimental runs.

### Permutation feature importance

The phenotyping PFI stage operates on a held-out validation fold after fold-specific preprocessing. Each feature is one
column of the final model input, including encoded variables and static cohort variables joined to the time-indexed
representation. The supplied configuration evaluates five seed-specific models. For each seed/fold run, NumPy's
random-number generator is initialized once from the model seed. Each feature column is then permuted once by a single
global shuffle across all valid rows from all patients and time steps, while all other columns remain unchanged. The
model is reevaluated using last-step predictions, and per-phenotype AUC-ROC and AUC-PR are stored together with the
unpermuted baseline in `eval/phenotyping/baseline/interpret/<seed>/<fold>/pfi.parquet`; the change from baseline is the
feature-importance effect. This procedure measures sensitivity to breaking the association between the selected input
feature and the remaining inputs. It does not preserve complete patient trajectories or within-patient temporal
alignment for the permuted feature and does not provide a causal importance estimate.

## Software validation and reproducibility

Reliability is addressed at complementary levels. At the model-evaluation level, the baseline workflow repeats 10-fold
multilabel-stratified cross-validation across seven random seeds; each seed determines a fold assignment and the model
initialization. This design reduces dependence of reported performance on a particular partition or initialization. At
the software/workflow level, deterministic execution controls, pinned dependencies, automated tests, DVC provenance, and
end-to-end reproduction on a second computational system provide complementary checks of pipeline behavior. These
controls support reliability of the evaluated research workflow; they do not establish clinical reliability,
cross-institutional generalizability, or prospective deployment performance.

RealPhe is designed for repeatable execution under controlled stochasticity within a fixed computational environment.
The current PyTorch implementation sets Python, NumPy, and PyTorch random seeds, disables cuDNN benchmarking, and
requests deterministic PyTorch algorithms for model-fitting tasks. Compared with the historical TensorFlow
implementation, these settings provide stricter control of stochastic execution and reduce incidental variation between
independently trained models. Python and package versions are pinned in `pyproject.toml` and `uv.lock`, while DVC
records workflow dependencies and derived artifacts.

As a software-level validation of the PyTorch implementation, the complete v1.0.1 pipeline was rerun on a second
computational setup using the same MIMIC-IV 2.0 source data and experimental configuration. The original study used WSL2
under Windows, an Intel Core i7 processor, 32 GB RAM, and an NVIDIA GeForce RTX 3060 with 12 GB VRAM. The archived
v0.1.0 environment specifies Python 3.10.8 and TensorFlow 2.11.1. The reproduction used Ubuntu 24.04.5 LTS, an AMD Ryzen
7 9700X, 64 GiB RAM, an NVIDIA GeForce RTX 5060 Ti, Python 3.12.13, and PyTorch 2.11.0. Because the comparison changes
both the computational environment and the machine-learning implementation (TensorFlow-based v0.1.0 versus PyTorch-based
v1.0.1), it is an end-to-end reproduction of the published reference results rather than a bitwise same-version
repeatability test.

The PyTorch implementation achieves similar single-model performance to the published TensorFlow implementation, but its
more deterministic execution settings reduce variance between independently trained models. Consequently, ensemble
averaging provides a smaller additional performance gain, and several final ensemble metrics are modestly lower than the
published TensorFlow values. An absolute difference of 0.01 on metrics bounded between 0 and 1 is used as a transparent
practical reproducibility tolerance; all nine principal phenotyping metrics remain below this threshold, supporting
preservation of the published predictive behavior within the stated tolerance. Detailed values and interpretation are
provided in [`docs/reproducibility.md`](docs/reproducibility.md).

### Trained-model inference example

After the full pipeline has produced the Zarr-backed processed evaluation data and trained fold-specific models, the
inference example can be run directly with:

```bash
uv run python scripts/example_inference.py
```

The script uses `get_eval_set()` rather than reading intermediate files directly, so the example follows the same
`cohort.zarr` / `signals.zarr` loading path as the current workflow. It loads the trained seed `100`, fold `0` model,
uses the CUDA inference path used in the reference experiments and checks that the returned phenotype probabilities are
finite and lie in `[0, 1]`. After following the documented setup and completing the full workflow on the second
reproduction system, the corrected example was executed successfully on CUDA and returned
`Phenotype-probability matrix: (436706, 561)`; all assertions passed. The same
evaluation-loader and trained-model paths are exercised by the complete-workflow validation and inference benchmark
below.

### Inference benchmark

After the full pipeline has produced processed evaluation data and trained models, batched model-inference cost can be
measured with:

```bash
uv run python scripts/benchmark_inference.py --seed 100 --fold 0 --repeats 10
```

On the reference Ubuntu 24.04.5 / NVIDIA GeForce RTX 5060 Ti setup, one trained fold-specific model processed the complete
test cohort of 13,045 ICU stays (436,706 valid 2-hour time steps) with batch size 512 in a median of 1.20 s across 10 runs
(range 1.19-1.25 s), corresponding to approximately 364,000 valid time steps/s. The amortized computational cost was
0.092 ms per ICU stay and 0.0027 ms per valid time step. Initial dataset loading and preprocessing are excluded from the
timed region; batch collation and host-to-device transfer are included. These values characterize offline batched
prediction cost, not bedside request latency. Because the LSTM is unidirectional and the model input at each time step
contains only information available up to that time, one forward pass over a prepared sequence emits causal predictions
at all valid time steps; retrospective prefixes are not recomputed independently.

### Training resource benchmark

A representative fold-specific training stage can be measured with:

```bash
./scripts/benchmark_training.sh 100 0
```

The benchmark invokes the normal training entry point with `solution=baseline`, samples aggregate process-tree proportional
set size (PSS) and GPU memory during execution, and records GNU `time` output. On the reference Ubuntu 24.04.5 / RTX 5060 Ti
system, seed `100`, fold `0` used 46,264 training and 5,179 validation ICU stays with 561 targets. The complete invoked stage
finished in 74.2 s, with peak sampled aggregate process-tree PSS of 16.3 GiB and peak sampled GPU-memory use of 1.84 GiB.
The benchmark uses a default 0.2-s polling delay. The measurement covers data loading, fold-specific preprocessing,
optimization, model serialization, and experiment logging. PSS is used rather than summing per-process RSS because
DataLoader worker processes may share memory pages.

### Metric aggregation

For final test evaluation, predictions are first averaged across the 10 fold-specific models within each configured seed and
then across the seven seeds. AUC-ROC and AUC-PR are calculated separately for each of the 561 phenotypes and summarized as
unweighted macro-means. In the reproduced test set, every phenotype had both positive and negative examples in first-step,
last-step, and time-distributed evaluation, so all 561 phenotype-specific AUC values were defined and included in each
macro-mean; no special undefined-label handling or exclusion was required. mAP@10 is instead computed per evaluated row
from the ranked phenotype predictions and then averaged across rows.

For best reproducibility:

- use the same Git revision,
- use the same MIMIC-IV/DVC data version,
- use the pinned dependency environment,
- retain the configured random seeds and deterministic execution settings,
- avoid modifying raw data manually,
- keep generated experiment outputs tracked through DVC/DVCLive where appropriate.

## Testing

RealPhe includes a lightweight automated test suite that does not require access to MIMIC-IV. The suite comprises cases
(including parametrized cases) covering the weighted temporal loss, mAP@k implementation, early stopping, preprocessing
transformers, random and multilabel-stratified splitting, variable-length phenotyping datasets and packed-sequence
collation, experiment-monitoring outputs, and model serialization. A dedicated regression test verifies that training
saves and registers the same `model.pt` path with DVCLive.

The synthetic smoke test uses three trajectories of lengths 3, 2, and 4 with three input features and two phenotype
targets. The expected model output therefore contains 9 valid time-step predictions by 2 targets. The test checks this
`(9, 2)` shape, finite predictions, preservation of packed-sequence structure, and numerical agreement before and after
saving/reloading `model.pt`. A successful standalone run reports `1 passed`; no MIMIC-IV data are required. These
structural and round-trip numerical checks provide explicit expected behavior without relying on a device-specific
floating-point checksum.

For the RealPhe software in the locked Python 3.12.13 environment, all tests passed. The smoke test also passed when
run independently. The package-wide line-coverage report was 25%; this includes MIMIC-IV-dependent analysis and task
modules that are not exercised by the synthetic/unit suite. The complete workflow reproduction described in
[`docs/reproducibility.md`](docs/reproducibility.md) provides complementary system-level validation of the principal
pipeline behavior.

Run the complete test suite with:

```bash
make test
```

Run only the lightweight smoke test with:

```bash
make test-smoke
```

Coverage can be inspected with:

```bash
make test-cov
```

The test commands use the locked project environment together with the two pinned test-only dependencies in
`test/requirements.txt`. These automated tests complement, rather than replace, the full MIMIC-IV software-validation
run described in [`docs/reproducibility.md`](docs/reproducibility.md).

## Suggested workflow

A typical fresh setup consists of the following sequence:

```bash
# 1. Configure PhysioNet credentials
export PHYSIONET_USER="your_physionet_username"
export PHYSIONET_PASSWORD="your_physionet_password"

# 2. Download and track raw data
make physionet-download DB=MIMICIV_20
uv run dvc add data/raw/physionet.org

# 3. Prepare data
make data-setup DB=MIMICIV_20
make data-concepts DB=MIMICIV_20
make data-extend DB=MIMICIV_20

# 4. Start required services
docker network create -d bridge local-network  # only if the network does not exist
docker compose up

# 5. Install the project environment and run all experiments
make install
```

## Troubleshooting

### PhysioNet authentication fails

Check that `PHYSIONET_USER` and `PHYSIONET_PASSWORD` are set in the same shell session where you run `make
physionet-download`. Also verify that your PhysioNet account has approved access to MIMIC-IV 2.0.

### Docker network already exists

If `docker network create` reports that `local-network` already exists, this is not an error. Continue with:

```bash
docker compose up
```

### Not enough disk space

MIMIC-IV and derived intermediate files can require substantial storage. Ensure that at least 150 GB of free disk space
is available before downloading and preprocessing the data.

### GPU is not detected

Check the NVIDIA driver and Docker GPU runtime configuration. Also verify that the Python environment can access the GPU
from inside the execution environment used by the pipeline.

### DVC/DVCLive outputs are missing

Confirm that the evaluation stage completed successfully and inspect the `eval/` directory. Metrics, parameters, plots,
and generated reports should be created there when the corresponding evaluation workflow is executed.

## References

[^1]: P. Picheta and S. Deniziak, “Optimizing real-time phenotyping in critical care using machine learning on electronic health records,” Expert Syst. Appl., vol. 320, p. 132084, Jul. 2026, doi: 10.1016/j.eswa.2026.132084.

[^2]: A. E. W. Johnson, D. J. Stone, L. A. Celi, and T. J. Pollard, “The MIMIC Code Repository: enabling reproducibility in critical care research,” J. Am. Med. Inform. Assoc., vol. 25, no. 1, pp. 32–39, Jan. 2018, doi: 10.1093/jamia/ocx084.

[^3]: A. Barrak, E. E. Eghan, and B. Adams, “On the Co-evolution of ML Pipelines and Source Code - Empirical Study of DVC Projects,” in 2021 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), Mar. 2021, pp. 422–433. doi: 10.1109/SANER50967.2021.00046.

[^4]: P. Picheta and S. Deniziak, “Supplementary Materials for Optimizing Real-Time Phenotyping in Critical Care Using Machine Learning on Electronic Health Records,” Mendeley Data, version 2, Mar. 2026, doi: 10.17632/n4jn62rh2m.2. https://data.mendeley.com/datasets/n4jn62rh2m/2
