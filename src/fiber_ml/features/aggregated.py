"""Aggregated features for the static dataset (Level 1 of project section 4).

For each measurement file we compute a fixed-size feature vector summarising
the spatial profile of both channels. These features are the input to baseline
regressors (Ridge, RandomForest, GradientBoosting) that predict (T, RH).

## Feature taxonomy

Per channel (CH1_REF_T and CH2_PI_TRH separately):

* Distribution: ``mean``, ``std``, ``min``, ``max``, ``median``, ``p25``, ``p75``, ``range``.
* Spatial dynamics: ``grad_mean_abs`` — mean absolute first difference along
  position, captures local spatial variability of the spectral shift.
* Quality: ``q_mean`` (mean of ``spectral_shift_quality``),
  ``q_frac_low`` (fraction of positions with quality < dropout threshold).

Cross-channel features:

* ``diff_mean`` = mean(CH2) - mean(CH1) — primary humidity signal candidate
  (CH1 is reference, CH2 is polyimide; the residual is the H2O response).
* ``diff_median`` — robust version of the same.
* ``ratio_std`` = std(CH2) / std(CH1) — relative spatial heterogeneity.

Total: 22 numeric features + 4 metadata columns (``experiment_id``, ``T``,
``RH``, ``replicate``).

## Why these features

The project description (section 4) requires Level-1 features that are
invariant to position-by-position noise but capture the average response and
its spatial structure. The grouping into per-channel + cross-channel allows
a Ridge / Lasso baseline to learn separable terms while still letting tree
ensembles exploit interactions.

For sequential models (Level 2, used in dynamic and spatio-temporal tasks),
do NOT use this module — feed the raw paired ``(y1, y2)`` profile from
:func:`fiber_ml.preprocessing.channels.extract_channels` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fiber_ml.preprocessing.channels import (
    DROPOUT_QUALITY_THRESHOLD,
    PairedChannels,
    extract_channels,
)

if TYPE_CHECKING:
    import xarray as xr


# Stable ordering — keeps DataFrame columns deterministic across runs and
# makes feature importance plots reproducible.
PER_CHANNEL_FEATURES: tuple[str, ...] = (
    "mean", "std", "min", "max", "median", "p25", "p75", "range",
    "grad_mean_abs", "q_mean", "q_frac_low",
)

CROSS_CHANNEL_FEATURES: tuple[str, ...] = (
    "diff_mean", "diff_median", "ratio_std",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "experiment_id", "T", "RH", "replicate",
)


def feature_columns() -> list[str]:
    """Return the canonical feature column order (without metadata).

    Useful for ``X = df[feature_columns()]`` to avoid implicit reliance
    on insertion order.
    """
    cols: list[str] = []
    for ch in ("ch1", "ch2"):
        for f in PER_CHANNEL_FEATURES:
            cols.append(f"{ch}_{f}")
    cols.extend(CROSS_CHANNEL_FEATURES)
    return cols


def _per_channel_stats(
    shift: pd.Series, quality: pd.Series, prefix: str,
    dropout_threshold: float,
) -> dict[str, float]:
    """Compute the 11 per-channel features for one channel."""
    s = shift.dropna()
    q = quality.dropna()

    if len(s) < 2:
        # Degenerate channel — return NaNs so it surfaces clearly in modeling
        return {f"{prefix}_{f}": float("nan") for f in PER_CHANNEL_FEATURES}

    grad = np.diff(s.to_numpy())

    return {
        f"{prefix}_mean":          float(s.mean()),
        f"{prefix}_std":           float(s.std()),
        f"{prefix}_min":           float(s.min()),
        f"{prefix}_max":           float(s.max()),
        f"{prefix}_median":        float(s.median()),
        f"{prefix}_p25":           float(s.quantile(0.25)),
        f"{prefix}_p75":           float(s.quantile(0.75)),
        f"{prefix}_range":         float(s.max() - s.min()),
        f"{prefix}_grad_mean_abs": float(np.mean(np.abs(grad))),
        f"{prefix}_q_mean":        float(q.mean()) if len(q) else float("nan"),
        f"{prefix}_q_frac_low":    float((q < dropout_threshold).mean()) if len(q) else float("nan"),
    }


def aggregate_one(
    paired: PairedChannels,
    dropout_threshold: float = DROPOUT_QUALITY_THRESHOLD,
) -> dict[str, float]:
    """Compute the full feature vector for one pre-extracted PairedChannels."""
    feats: dict[str, float] = {}
    feats.update(_per_channel_stats(
        paired.ch1["spectral_shift_ghz"],
        paired.ch1["spectral_shift_quality"],
        "ch1", dropout_threshold,
    ))
    feats.update(_per_channel_stats(
        paired.ch2["spectral_shift_ghz"],
        paired.ch2["spectral_shift_quality"],
        "ch2", dropout_threshold,
    ))

    # Cross-channel features — guard against degenerate cases
    ch1_mean = feats["ch1_mean"]
    ch2_mean = feats["ch2_mean"]
    ch1_med = feats["ch1_median"]
    ch2_med = feats["ch2_median"]
    ch1_std = feats["ch1_std"]
    ch2_std = feats["ch2_std"]

    feats["diff_mean"] = ch2_mean - ch1_mean
    feats["diff_median"] = ch2_med - ch1_med
    feats["ratio_std"] = ch2_std / ch1_std if ch1_std and ch1_std > 1e-12 else float("nan")

    return feats


def aggregate_from_zarr(
    ds: "xr.Dataset",
    dropout_threshold: float = DROPOUT_QUALITY_THRESHOLD,
) -> pd.DataFrame:
    """Build the aggregated feature DataFrame from a Zarr dataset.

    Args:
        ds: xarray Dataset opened from ``data_processed/dataset.zarr``,
            with dims ``(experiment, position, channel)``.
        dropout_threshold: Quality threshold below which a position counts
            as a dropout for the ``q_frac_low`` feature.

    Returns:
        DataFrame with one row per experiment. Columns: ``METADATA_COLUMNS``
        followed by ``feature_columns()``. The returned column order is
        canonical and stable.

    Raises:
        ValueError: If the dataset is missing required variables.
    """
    required = {"data", "T", "RH", "replicate", "experiment_id"}
    missing = required - set(ds.data_vars) - set(ds.coords)
    if missing:
        raise ValueError(f"Zarr dataset missing required variables: {missing}")

    rows: list[dict[str, object]] = []
    n_exp = ds.sizes["experiment"]
    channel_names = ds["channel"].values.tolist()

    for i in range(n_exp):
        arr = ds["data"].isel(experiment=i).values
        raw = pd.DataFrame(arr, columns=channel_names)
        paired = extract_channels(raw)

        row: dict[str, object] = {
            "experiment_id": str(ds["experiment_id"].values[i]),
            "T": int(ds["T"].values[i]),
            "RH": int(ds["RH"].values[i]),
            "replicate": int(ds["replicate"].values[i]),
        }
        row.update(aggregate_one(paired, dropout_threshold=dropout_threshold))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[list(METADATA_COLUMNS) + feature_columns()]


def aggregate_from_files(
    file_paths: list,
    metadata_lookup: dict[str, dict[str, int]],
    dropout_threshold: float = DROPOUT_QUALITY_THRESHOLD,
) -> pd.DataFrame:
    """Build the feature DataFrame from .txt files directly (no Zarr).

    Useful for sample-data pipelines and CI tests — keeps the test suite
    independent from data_processed/dataset.zarr being present.

    Args:
        file_paths: List of paths to ``Pomiar*.txt`` files.
        metadata_lookup: Mapping from file name (e.g. ``"Pomiar1.txt"``) to
            ``{"T": int, "RH": int, "replicate": int}``. Caller is
            responsible for resolving these from folder names / manifest.
        dropout_threshold: As in :func:`aggregate_from_zarr`.

    Returns:
        Same schema as :func:`aggregate_from_zarr`.
    """
    from pathlib import Path

    from fiber_ml.ingest.parser import parse_file

    rows: list[dict[str, object]] = []
    for fp in file_paths:
        fp = Path(fp)
        meta = metadata_lookup[fp.name]
        mf = parse_file(fp)
        paired = extract_channels(mf.data)

        row: dict[str, object] = {
            "experiment_id": f"T{meta['T']}_RH{meta['RH']}_R{meta['replicate']:02d}",
            "T": meta["T"],
            "RH": meta["RH"],
            "replicate": meta["replicate"],
        }
        row.update(aggregate_one(paired, dropout_threshold=dropout_threshold))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[list(METADATA_COLUMNS) + feature_columns()]
