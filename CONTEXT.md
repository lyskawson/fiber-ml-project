# Project Context — fiber-ml-project

> For LLM agents and team members. Read this before working on any task.
> Keep this file updated as decisions are made.

## Project Mission

Model temperature (T) and relative humidity (RH) from distributed fiber optic backscatter data
produced by a Luna OBR-4600 sensor. 6 ML tasks (see below). Trusted AI requirement:
full reproducibility, data versioning (DVC), no leakage, interpretability.

## Tech Stack

- **Python 3.11**, managed with `uv`
- **Data**: `zarr>=2.16,<3.0` (NOT v3), `xarray`, `dask[complete]`, `numcodecs`
- **ML**: `scikit-learn`, `torch`, `lightgbm`
- **Versioning**: `dvc[gdrive]>=3.0`, Google Drive remote
- **Quality**: `ruff` (lint + format), `mypy` (strict), `pytest`, `pre-commit`
- **CI/CD**: GitHub Actions

## Data Structure

### Raw files

700 `.txt` files in `data/raw/T{X}_RH{Y}/Pomiar{N}.txt`
- X ∈ {35,45,55,65,75} (°C)
- Y ∈ {20,30,40,50,60,70,80} (%RH)
- 35 conditions × 20 replicates = 700 files, ~1.6 GB total

### File format

```
14 lines header (key-value metadata)
1 blank line
1 column header line
41648 data lines (tab-separated, scientific notation E+000, CRLF)
```

### Zarr dataset schema (data_processed/dataset.zarr)

```python
xr.Dataset(
    data_vars={
        "data":          (experiment, position, channel)  # float32, shape (700, 41648, 5)
        "T":             (experiment,)                    # int8, Celsius
        "RH":            (experiment,)                    # int8, %RH
        "replicate":     (experiment,)                    # int8
        "acquired_at":   (experiment,)                    # datetime64[ns]
        "experiment_id": (experiment,)                    # str, e.g. "T35_RH20_R01"
    },
    coords={
        "channel": ["length_1_m", "length_2_m", "amplitude_db_mm",
                    "spectral_shift_ghz", "spectral_shift_quality"],
        "position": 0..41647
    }
)
```

Chunking: `(1, 41648, 5)` — one experiment per chunk.
Compression: `Blosc(cname='zstd', clevel=5, shuffle=BITSHUFFLE)`.

## Open Questions

1. **Spectral Shift sparsity** — `spectral_shift_ghz` and `length_2_m` are non-null only for
   the first ~808 positions (length ≈ 2.592–2.607 m). `opis_pomiarow_ML.txt` specifies feature
   extraction from ranges 2.65999–2.80083 m and 3.22034–3.36018 m, where Spectral Shift is empty.
   **Decision pending supervisor clarification. Do NOT engineer features from Spectral Shift yet.**

2. **Duplicate file markers** — `T65_RH50/` and `T75_RH40/` contain files with ` (1)` suffix
   (e.g. `Pomiar10 (1).txt`). These are flagged in manifest (`has_duplicate_marker=True`) and
   skipped during Zarr ingest until manual verification.

3. **Two feature ranges per sensor point** — `opis_pomiarow_ML.txt` states that each spatial
   point has a (y1, y2) pair from the two measurement ranges. This implies the feature space
   is 2D per position, not 1D. Awaiting clarification on how to align the two ranges.

## Data quality fix — duplicate filename markers (resolved 2026-05-04)

20 files in `T65_RH50/` and `T75_RH40/` originally had ` (1)` suffix in
filename, likely from re-download from Google Drive. Investigation showed
these were the ONLY copies for their replica numbers (not actual duplicates).
Files were renamed to canonical form (`Pomiar{N}.txt`) and ingested.

Manifest pre-fix: 680 unique experiments (T65_RH50: 8, T75_RH40: 12).
Manifest post-fix: 700 unique experiments (all conditions: 20 replicas).

Verification: numerical replicate IDs in original `(1)` files were disjoint
from non-`(1)` files in the same condition folder, confirming they represent
distinct measurements rather than duplicates.

## ML Tasks (all TODO)

| # | Task | Type |
|---|------|------|
| 1 | Static T regression | Regression |
| 2 | Static RH regression | Regression |
| 3 | Dynamic T regression | Regression (time-aware) |
| 4 | Dynamic RH regression | Regression (time-aware) |
| 5 | Operating regime classification | Classification |
| 6 | Anomaly detection | Unsupervised |
| 7 | Hysteresis analysis | Analysis |
| 8 | Spatio-temporal structure analysis | Analysis |

## Architectural Decisions

See `docs/decisions/`:
- [ADR-0001: Zarr over HDF5](docs/decisions/0001-zarr-over-hdf5.md)

## Anti-patterns

- **Never commit raw data** — DVC tracks `data/raw/` and `data_processed/`
- **No hardcoded paths** — use `src/fiber_ml/utils/paths.py`
- **No pd.read_csv without explicit dtypes** at ingest boundaries
- **No feature engineering on Spectral Shift** until open question #1 is resolved
- **No LiveData** (Python equivalent: no callbacks replacing generators/async)
- **No God classes** — each module has one responsibility

## How to Give an LLM Context

When asking an LLM for help with this project, always include:
1. This `CONTEXT.md` file
2. The relevant source module (e.g. `src/fiber_ml/ingest/parser.py`)
3. A sample data file if the question is format-related (`data/sample/T35_RH20/Pomiar1.txt`)
4. The specific test that is failing (if applicable)

**Do not paste raw 41648-row data files.** Use `head -20` or the parsed DataFrame summary.

## Data quality fixes (resolved 2026-05-04)

### Filename duplicate markers

20 files in `T65_RH50/` and `T75_RH40/` originally had ` (1)` suffix
(non-breaking space + `(1)` pattern), likely artifact from re-download
from Google Drive. Investigation showed these were the ONLY copies for
their replica numbers — not actual byte-level duplicates of canonical files.

Replicate numbers in `(1)`-suffixed files were disjoint from non-suffixed
files in same folder, confirming distinct measurements:
- T65_RH50: non-suffixed {1,3,5,9,11,16,17,20} ∪ suffixed {2,4,6,7,8,10,12,13,14,15,18,19} = full set 1..20
- T75_RH40: non-suffixed {1..11,20} ∪ suffixed {12..19} = full set 1..20

Resolution: explicit `mv` to canonical names, then re-ingested.
Final dataset: shape (700, 41648, 5), 35 conditions × 20 replicates,
zero anomalies.

Note: bash/zsh parameter expansion (`${var/ (1)/}`) failed to match the
suffix pattern despite glob `*(1)*` matching files — likely Unicode whitespace
mismatch (NBSP U+00A0 vs ASCII space U+0020). Explicit single-quoted `mv`
worked because single-quotes treat the string as literal bytes.
