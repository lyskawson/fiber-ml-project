"""Train all baseline models and dump per-target + LOCO-CV metrics.

Usage:
    uv run python scripts/04_train_baseline.py \
        --features data_processed/aggregated.parquet \
        --output-dir reports/metrics

Or build features on the fly from Zarr:

    uv run python scripts/04_train_baseline.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/metrics

Produces:

* ``baseline_per_target.csv`` — held-out test MAE/RMSE/R² for the
  default replicate split (train 1-14, val 15-17, test 18-20).
* ``baseline_per_condition.csv`` — same metrics broken down by (T, RH).
* ``loco_summary.csv`` — Leave-One-Condition-Out CV: mean / std / max
  MAE across all 35 held-out conditions.
* ``loco_folds.csv`` — full per-fold per-target detail.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fiber_ml.eval.metrics import (
    per_condition_metrics,
    per_target_metrics,
    summarise_loco,
)
from fiber_ml.features.aggregated import (
    aggregate_from_zarr,
    feature_columns,
)
from fiber_ml.models.baseline import TARGETS, fit_all_baselines, fit_baseline
from fiber_ml.models.splits import loco_cv, replicate_split

logger = logging.getLogger(__name__)


def _load_features(args: argparse.Namespace) -> pd.DataFrame:
    if args.features:
        logger.info("Loading aggregated features from %s", args.features)
        return pd.read_parquet(args.features)
    if args.zarr:
        import xarray as xr

        logger.info("Building features from Zarr at %s", args.zarr)
        ds = xr.open_zarr(str(args.zarr))
        return aggregate_from_zarr(ds)
    raise SystemExit("Pass either --features or --zarr.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        help="Path to aggregated features parquet.")
    parser.add_argument("--zarr", type=Path,
                        help="Path to Zarr dataset (used if --features absent).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/metrics"),
                        help="Where to write the CSV outputs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_features(args)
    feature_cols = feature_columns()
    logger.info("Loaded %d rows × %d features", len(df), len(feature_cols))

    # ---- Replicate split: standard test report ----
    split = replicate_split(df)
    logger.info(
        "Replicate split: train=%d, val=%d, test=%d",
        len(split.train), len(split.val), len(split.test),
    )

    models = fit_all_baselines(df, feature_cols, split.train, seed=args.seed)

    per_target_rows: list[pd.DataFrame] = []
    per_cond_rows: list[pd.DataFrame] = []
    for name, model in models.items():
        df_test = df.iloc[split.test].reset_index(drop=True)
        y_true = df_test[list(TARGETS)].to_numpy()
        y_pred = model.predict(df_test)

        pt = per_target_metrics(y_true, y_pred)
        pt["model"] = name
        per_target_rows.append(pt)

        pc = per_condition_metrics(df_test, y_true, y_pred)
        pc["model"] = name
        per_cond_rows.append(pc)

    pd.concat(per_target_rows, ignore_index=True).to_csv(
        args.output_dir / "baseline_per_target.csv", index=False,
    )
    pd.concat(per_cond_rows, ignore_index=True).to_csv(
        args.output_dir / "baseline_per_condition.csv", index=False,
    )
    logger.info("Wrote baseline_per_target.csv and baseline_per_condition.csv")

    # ---- LOCO CV: section-9 generalisation report (Ridge only by default) ----
    logger.info("Running Leave-One-Condition-Out CV with Ridge ...")
    fold_results = []
    for (T, RH), fold_split in loco_cv(df):
        model = fit_baseline("ridge", df, feature_cols, fold_split.train,
                             seed=args.seed)
        df_test = df.iloc[fold_split.test].reset_index(drop=True)
        y_true = df_test[list(TARGETS)].to_numpy()
        y_pred = model.predict(df_test)
        pt = per_target_metrics(y_true, y_pred)
        fold_results.append({"T": T, "RH": RH, "per_target": pt})

    folds_long: list[pd.DataFrame] = []
    for fr in fold_results:
        f = fr["per_target"].copy()
        f["T_holdout"] = fr["T"]
        f["RH_holdout"] = fr["RH"]
        folds_long.append(f)
    pd.concat(folds_long, ignore_index=True).to_csv(
        args.output_dir / "loco_folds.csv", index=False,
    )
    summarise_loco(fold_results).to_csv(
        args.output_dir / "loco_summary.csv", index=False,
    )
    logger.info("Wrote loco_folds.csv and loco_summary.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()
