# Real-time Phenotyping Framework

This repository provides a reproducible workflow for real-time computational phenotyping in critical care using machine
learning on electronic health record (EHR) data [^1]. It contains utilities for downloading and preparing MIMIC-IV data,
constructing clinical concepts [^2], extending the prepared dataset with derived features, running the
experimental pipeline, and comparing experiment outputs with DVC/DVCLive [^3].

The framework is intended for research workflows where the same data-processing and model-evaluation procedure must be
repeatable across runs on the same computational setup.

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

## Reproducibility

The framework is designed to provide full reproducibility across repeated experimental runs performed on the same
computational setup.

Minor differences in final metrics may occur only when experiments are executed on different setups. Possible causes
include differences in hardware, GPU implementations, numerical precision, dependency versions, or rounding behavior.
Such differences are not expected within the same setup and should be treated only as a possibility across different
environments.

For best reproducibility:

- use the same Git revision,
- use the same DVC data version,
- use the same dependency lock file/environment,
- use the same GPU model and driver stack when possible,
- avoid modifying raw data manually,
- keep generated experiment outputs tracked through DVC/DVCLive where appropriate.

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
