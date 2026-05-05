# Project Context — fiber-ml-project

> For LLM agents and team members. Read this before working on any task.
> Keep this file updated as decisions are made.

## Project Mission

Model temperature (T) and relative humidity (RH) from distributed fiber optic backscatter data
produced by a Luna OBR-4600 sensor. 6 ML tasks (see below). Trusted AI requirement:
full reproducibility, data versioning, no leakage, interpretability.

## Tech Stack

- **Python 3.11**, managed with `uv`
- **Data**: `zarr>=2.16,<3.0` (NOT v3), `xarray`, `dask[complete]`, `numcodecs`
- **ML**: `scikit-learn`, `torch`, `lightgbm`
- **Storage**: Hugging Face Hub (private dataset) for raw + processed
- **Quality**: `ruff` (lint + format), `mypy` (strict), `pytest`, `pre-commit`
- **CI/CD**: GitHub Actions

## Data Structure

### Raw files

700 `.txt` files in `data/raw/T{X}_RH{Y}/Pomiar{N}.txt`
- X ∈ {35,45,55,65,75} (°C)
- Y ∈ {20,30,40,50,60,70,80} (%RH)
- 35 conditions × 20 replicates = 700 files, ~1.5 GB total
- Stored on HF Hub: https://huggingface.co/datasets/lyskawson/fiber-ml-luna-obr-4600

### File format
14 lines header (key-value metadata)
1 blank line
1 column header line
41648 data lines (tab-separated, scientific notation E+000, CRLF)

Header note: `Spectral Shift` and `length_2_m` are populated only for the
first ~808 positions (length 2.592–2.607 m); rest is NaN. See Open Question #1.

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
Total compressed size: ~162 MB.

## Storage architecture

### Decision: HF Hub instead of DVC+Drive

Original plan was DVC + Google Drive. Changed because:
1. Google Drive OAuth flow blocked by Google's third-party app verification
2. DVC's native Hugging Face support (`dvc[hf]`) does not exist
3. Service account workaround adds onboarding friction per team member

What we use now:
- **Code + manifest + sample data**: Git/GitHub (https://github.com/lyskawson/fiber-ml-project)
- **Raw + processed data**: HF Hub private dataset
- **Integrity**: sha256 hashes per file in `data/manifest.csv`
- **Reproducibility**: Git commit hash + deterministic ingest pipeline

What we lost vs DVC:
- Automatic content-addressed caching (DVC's killer feature)
- `dvc repro` pipeline auto-execution

What we kept:
- Reproducibility through manifest + scripts
- Single-source-of-truth for raw data
- Team collaboration on code

### Migration path

If strict reproducibility becomes critical (e.g. for publication), migration
to DVC + Backblaze B2 (S3-compatible) is straightforward (5-10 min setup).

## Open Questions

1. **Spectral Shift sparsity** — `spectral_shift_ghz` and `length_2_m` are non-null only for
   the first ~808 positions (length ≈ 2.592–2.607 m). `opis_pomiarow_ML.txt` specifies feature
   extraction from ranges 2.65999–2.80083 m and 3.22034–3.36018 m, where Spectral Shift is empty.
   **Decision pending supervisor clarification. Do NOT engineer features from Spectral Shift yet.**

2. **Two channels (CH1_REF, CH2_PI)** — project description mentions two parallel fiber channels,
   but `.txt` files contain only one set of columns. Hypothesis: the two length ranges
   (2.66–2.80 and 3.22–3.36) are CH1_REF and CH2_PI on the same fiber. Awaiting confirmation.

3. **"Quality Factor" in Załącznik section 2.2** — recommended schema mentions Quality Factor
   field, but OBR-4600 export does not contain such named column. Possibilities:
   (a) it refers to Spectral Shift Quality (which we have),
   (b) computed from amplitude (e.g. local SNR),
   (c) field from the other sensor mentioned in załącznik title (ODiSI 7100).

4. **"Active Profile"** — Załącznik section 2.2 defines this as "data after global offset
   correction". This is a preprocessing output, not a raw field. To be implemented in
   `src/fiber_ml/preprocessing/` after offset correction algorithm is decided.

## Data quality fixes (resolved 2026-05-04)

### Filename duplicate markers

20 files in `T65_RH50/` and `T75_RH40/` originally had ` (1)` suffix in
filename, likely from re-download from Google Drive. Investigation showed
these were the ONLY copies for their replica numbers — not actual byte-level
duplicates of canonical files.

Replicate numbers in `(1)`-suffixed files were disjoint from non-suffixed
files in the same condition folder, confirming distinct measurements:
- T65_RH50: non-suffixed {1,3,5,9,11,16,17,20} ∪ suffixed {2,4,6,7,8,10,12,13,14,15,18,19} = 1..20
- T75_RH40: non-suffixed {1..11,20} ∪ suffixed {12..19} = 1..20

Resolution: explicit `mv` to canonical names, then re-ingested.
Final dataset: shape (700, 41648, 5), 35 conditions × 20 replicates.

Note: bash/zsh parameter expansion (`${var/ (1)/}`) failed to match the
suffix pattern despite glob `*(1)*` matching files — likely Unicode whitespace
mismatch (NBSP U+00A0 vs ASCII space U+0020). Single-quoted `mv` worked
because single-quotes treat the string as literal bytes.

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

### Decision: CSV manifest instead of PostgreSQL

The supplementary document (Załącznik, section 2.1) recommends PostgreSQL
for metadata storage citing leakage control and reproducibility. We adopt
a CSV-based manifest with the following justification:

1. **Reproducibility**: provided by Git + deterministic ingest; does not require
   ACID transactions.
2. **Leakage control**: implemented at split-construction time in code, not
   through database constraints.
3. **Single-writer pattern**: manifest is generated by `01_build_manifest.py`
   and consumed read-only — no concurrent writes.
4. **Query patterns**: filtering by (T, RH, replicate) is trivial in pandas.
5. **Onboarding cost**: zero-config CSV vs PostgreSQL setup per team member.

Migration to SQLite or PostgreSQL is one-script transformation if needs change:
```python
pandas.read_csv('manifest.csv').to_sql('experiments', sqlite_conn)
```

The conceptual schema (Experiments / Regime_Labels / Validation_Targets) from
the recommendation guides our CSV design:
- `data/manifest.csv` ≡ Experiments table (current)
- `data/regime_labels.csv` (TODO)
- `data/validation_targets.csv` (TODO)

## Anti-patterns

- **Never commit raw data** — gitignored; lives on HF Hub
- **Never commit data_processed/dataset.zarr** — gitignored; lives on HF Hub
- **No HF tokens in code or commits** — always via `os.environ['HF_TOKEN']` or `--token` arg
- **No hardcoded paths** — use `src/fiber_ml/utils/paths.py`
- **No `pd.read_csv` without explicit dtypes** at ingest boundaries
- **No feature engineering on Spectral Shift** until Open Question #1 is resolved
- **No God classes** — each module has one responsibility

## How to Give an LLM Context

When asking an LLM for help with this project, always include:
1. This `CONTEXT.md` file
2. The relevant source module (e.g. `src/fiber_ml/ingest/parser.py`)
3. A sample data file if the question is format-related (`data/sample/T35_RH20/Pomiar1.txt`)
4. The specific test that is failing (if applicable)

**Do not paste raw 41648-row data files.** Use `head -20` or the parsed DataFrame summary.