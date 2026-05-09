"""Regression metrics for tasks 1-2 (per-target + per-condition breakdowns).

Section 8 of the project description requires reporting metrics:

* Per target (T and RH separately, never averaged into a single number that
  hides which target is harder).
* Per condition (so we can see whether errors are uniform across the (T, RH)
  grid or concentrated in a few states).
* In appropriate physical units (°C for T, %RH for RH).

This module computes all three views from a (model, split) pair.

## Output schema

:func:`per_target_metrics` returns:

    target  mae   rmse  r2    max_abs
    T       0.83  1.04  0.97  3.21
    RH      2.11  2.85  0.94  8.40

:func:`per_condition_metrics` returns one row per (T, RH) tested:

    T   RH  target  mae   rmse  max_abs  n_samples
    20  30  T       0.5   0.6   1.1      3
    20  30  RH      1.8   2.0   3.5      3
    ...

Both schemas are flat DataFrames so they're easy to dump to CSV / read with
pandas / plot with seaborn / aggregate further.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _per_target_row(
    y_true: np.ndarray, y_pred: np.ndarray, target_name: str,
) -> dict[str, float | str]:
    """One row of per-target metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    max_abs = float(np.max(np.abs(y_true - y_pred))) if len(y_true) else float("nan")
    return {
        "target": target_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "max_abs": max_abs,
        "n_samples": int(len(y_true)),
    }


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] = ("T", "RH"),
) -> pd.DataFrame:
    """Compute MAE / RMSE / R² / max_abs separately for each output target.

    Args:
        y_true: Array of shape (n_samples, n_targets).
        y_pred: Array of shape (n_samples, n_targets).
        target_names: Names of the columns in y_true / y_pred.

    Returns:
        DataFrame with one row per target.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} shape mismatch"
        )
    if y_true.shape[1] != len(target_names):
        raise ValueError(
            f"y_true has {y_true.shape[1]} columns but {len(target_names)} "
            f"target names were provided: {target_names}"
        )

    rows = [
        _per_target_row(y_true[:, i], y_pred[:, i], target_names[i])
        for i in range(len(target_names))
    ]
    return pd.DataFrame(rows)


def per_condition_metrics(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] = ("T", "RH"),
) -> pd.DataFrame:
    """Per-(T,RH) metric breakdown — flags conditions where the model struggles.

    Args:
        df_test: The test slice of the feature DataFrame, with ``T`` and ``RH``
            columns. Row order must match ``y_true`` / ``y_pred``.
        y_true: True targets, shape (n_samples, n_targets).
        y_pred: Predictions, same shape.
        target_names: Column names corresponding to y_true / y_pred axis 1.

    Returns:
        DataFrame indexed by (T, RH, target).
    """
    if len(df_test) != len(y_true):
        raise ValueError("df_test and y_true must have same number of rows")

    rows: list[dict[str, float | str | int]] = []
    grouped = df_test.groupby(["T", "RH"], sort=True).indices

    for (T, RH), idx in grouped.items():
        for j, name in enumerate(target_names):
            yt = y_true[idx, j]
            yp = y_pred[idx, j]
            mae = float(mean_absolute_error(yt, yp)) if len(yt) else float("nan")
            rmse = (
                float(np.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan")
            )
            max_abs = float(np.max(np.abs(yt - yp))) if len(yt) else float("nan")
            rows.append(
                {
                    "T": int(T),
                    "RH": int(RH),
                    "target": name,
                    "mae": mae,
                    "rmse": rmse,
                    "max_abs": max_abs,
                    "n_samples": int(len(yt)),
                }
            )

    return pd.DataFrame(rows).sort_values(["T", "RH", "target"]).reset_index(drop=True)


def summarise_loco(
    fold_results: list[dict[str, object]],
) -> pd.DataFrame:
    """Aggregate per-fold metrics from a Leave-One-Condition-Out CV run.

    Expected input shape — list of dicts with keys: ``T``, ``RH``,
    ``per_target`` (DataFrame from :func:`per_target_metrics`).

    Returns: DataFrame with mean / std / max of each metric across folds,
    one row per target. This is the headline "how well do we generalise to
    unseen conditions" number for the report.
    """
    if not fold_results:
        return pd.DataFrame(
            columns=["target", "mae_mean", "mae_std", "mae_max", "rmse_mean", "rmse_std"]
        )

    # Stack all per-target frames into one with a fold key
    frames = []
    for fr in fold_results:
        f = fr["per_target"].copy()  # type: ignore[assignment, union-attr]
        assert isinstance(f, pd.DataFrame)
        f["T_holdout"] = fr["T"]
        f["RH_holdout"] = fr["RH"]
        frames.append(f)
    big = pd.concat(frames, ignore_index=True)

    return (
        big.groupby("target")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_max=("mae", "max"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_folds=("mae", "count"),
        )
        .reset_index()
    )
