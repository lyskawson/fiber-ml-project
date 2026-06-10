# fiber-ml-project

Research and deployment project: modelling temperature (T) and relative humidity (RH)
from distributed optical fiber measurements using the **Luna OBR-4600** sensor.

## Project goals

Developing an ML pipeline covering 6 tasks:

1. Static regression of T and RH
2. Dynamic regression of T and RH (time-aware)
3. Sensor operating regime classification
4. Anomaly detection
5. T/RH hysteresis analysis
6. Spatio-temporal structure analysis of the signal

Trusted AI requirement: full reproducibility, data leakage control, model interpretability.

## Quickstart

```bash
git clone https://github.com/lyskawson/fiber-ml-project.git
cd fiber-ml-project

# Set up environment (Python 3.11)
uv sync --extra dev

# Download data from Hugging Face Hub (~1.7 GB, ~2 min on a fast connection)
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py

# Validate environment
uv run pytest tests/ -v
```

## Dataset access

Raw measurements (~1.5 GB) and processed Zarr (~162 MB) are stored on
[Hugging Face Hub](https://huggingface.co/datasets/lyskawson/fiber-ml-luna-obr-4600)
as a private dataset.

### Setup for a new team member

1. Create an account: https://huggingface.co/join
2. Send your HF username to the lead (lyskawson) — you will receive **Write** access (all team members are equal contributors)
3. Generate a token with **Write** role: https://huggingface.co/settings/tokens
4. Download the dataset:

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'
uv run python scripts/download_from_hf.py                  # everything
uv run python scripts/download_from_hf.py --what raw       # raw only
uv run python scripts/download_from_hf.py --what processed # Zarr only
```

### Re-uploading changed data

Anyone on the team with a Write token can upload changes:

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxx'   # Write scope
uv run python scripts/upload_to_hf.py
```

> **Warning**: Never commit HF tokens to git or paste them in chats/issues/PRs.
> Keep them in an env var (`export HF_TOKEN=...`) or in a local `.env` (gitignored).

## Repository structure
```text
.
├── src/fiber_ml/          # Main Python package
│   ├── ingest/            # .txt parser, manifest, Zarr conversion
│   ├── preprocessing/     # (TBD) normalisation, segmentation
│   ├── features/          # (TBD) feature engineering
│   ├── models/            # (TBD) ML models
│   ├── eval/              # (TBD) metrics, plots
│   └── utils/             # Paths, helpers
├── scripts/
│   ├── 01_build_manifest.py    # raw .txt -> manifest.csv
│   ├── 02_ingest_to_zarr.py    # raw .txt -> Zarr dataset
│   ├── upload_to_hf.py         # local -> HF Hub
│   └── download_from_hf.py     # HF Hub -> local
├── tests/                 # pytest tests (run against data/sample/)
├── data/sample/           # 2 measurement files — in git for CI/tests
├── data/raw/              # 700 files (~1.5 GB) — gitignored, on HF Hub
├── data/manifest.csv      # File -> conditions map — in git
├── data_processed/        # Zarr dataset — gitignored, on HF Hub
├── docs/                  # Format description, ADRs, project documentation
├── notebooks/             # EDA exploration
└── reports/               # Metrics, plots (generated)
...
```

## Dataset regeneration workflow

The Zarr dataset is generated deterministically from raw data via:

```bash
uv run python scripts/01_build_manifest.py \
    --data-dir data/raw \
    --output data/manifest.csv

uv run python scripts/02_ingest_to_zarr.py \
    --manifest data/manifest.csv \
    --output data_processed/dataset.zarr
```

After regeneration, re-upload to HF: `uv run python scripts/upload_to_hf.py`.

## Team workflow

### Branch naming
task/<num>-<short-desc>    # e.g. task/1-static-regression-T
fix/<short-desc>           # e.g. fix/spectral-shift-nan-handling
docs/<short-desc>          # e.g. docs/add-data-format-spec

### Conventional commits
feat: add static T regression model
fix: handle NaN in spectral shift channel
docs: update data_format.md with supervisor clarification
chore: bump zarr to 2.18
test: add regression test for sparse spectral shift

### Pull Request

- Branch from `main`, PR into `main`
- CI must pass (lint + typecheck + tests)
- Minimum 1 reviewer

## Onboarding — first steps

After setting up the environment, each team member should:

1. Read `CONTEXT.md` in full (architecture, open questions, anti-patterns)
2. Pick one of the 8 ML tasks from the "ML Tasks" section in `CONTEXT.md`
3. Create a `task/X-...` branch and an exploratory notebook in `notebooks/`
4. **Before starting feature engineering**: open question #1 (Spectral Shift) must be clarified with the supervisor

## Team

See [CONTRIBUTORS.md](CONTRIBUTORS.md).
