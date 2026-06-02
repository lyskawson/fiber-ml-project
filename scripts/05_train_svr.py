"""Train SVR model with GridSearchCV hyperparameter tuning.

Strojenie hiperparametrow przez GridSearchCV na zbiorze treningowym
(3-fold CV). Najlepszy model jest potem oceniany na zbiorze testowym
oraz w LOCO CV.

Usage:
    uv run python scripts/05_train_svr.py \
        --features data_processed/aggregated.parquet \
        --output-dir reports/metrics

Or build features on the fly from Zarr:

    uv run python scripts/05_train_svr.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/metrics

Produces:

* ``svr_per_target.csv`` — held-out test MAE/RMSE/R² for the
  tuned SVR (best params from grid search).
* ``svr_per_condition.csv`` — same metrics broken down by (T, RH).
* ``svr_loco_summary.csv`` — Leave-One-Condition-Out CV with tuned SVR.
* ``svr_loco_folds.csv`` — full per-fold per-target detail.
* ``svr_best_params.txt`` — wybrane hiperparametry (do raportu).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from fiber_ml.eval.metrics import (
    per_condition_metrics,
    per_target_metrics,
    summarise_loco,
)
from fiber_ml.features.aggregated import (
    aggregate_from_zarr,
    feature_columns,
)
from fiber_ml.models.baseline import TARGETS
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


def make_svr_pipeline(C: float = 10.0, gamma="scale", kernel: str = "rbf") -> Pipeline:
    """Tworzy nowy, niewytrenowany SVR pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(
            SVR(kernel=kernel, C=C, gamma=gamma)
        )),
    ])


def tune_svr(X_train, y_train, seed: int = 42) -> tuple[Pipeline, dict, float]:
    """Strojenie hiperparametrow przez GridSearchCV.

    Zwraca: (najlepszy_pipeline, najlepsze_params, najlepszy_score)
    """
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(SVR())),
    ])

    # Siatka parametrow - przeszukamy 36 kombinacji
    # Nazwy parametrow: <step_name>__<estimator>__<parameter>
    param_grid = {
        "svr__estimator__C": [1.0, 10.0, 100.0, 1000.0],
        "svr__estimator__gamma": ["scale", "auto", 0.01, 0.1],
        "svr__estimator__kernel": ["rbf"],          # zostawiamy RBF (najczesciej najlepsze)
        "svr__estimator__epsilon": [0.01, 0.1],     # tolerancja regresji
    }

    logger.info("Grid search: %d combinations × 3-fold CV = %d fits",
                4 * 4 * 1 * 2,
                4 * 4 * 1 * 2 * 3)

    grid = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,         # wszystkie dostepne rdzenie
        verbose=1,
        refit=True,        # po znalezieniu najlepszego, fituj na calym train
    )

    grid.fit(X_train, y_train)

    best_score = -grid.best_score_  # negate bo neg_mean_absolute_error
    logger.info("Best CV MAE: %.4f", best_score)
    logger.info("Best params: %s", grid.best_params_)

    return grid.best_estimator_, grid.best_params_, best_score


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

    # ---- Load features ----
    df = _load_features(args)
    feature_cols = feature_columns()
    logger.info("Loaded %d rows × %d features", len(df), len(feature_cols))

    # ===================================================================
    # CZĘŚĆ 1: Replicate split z GridSearchCV
    # ===================================================================
    split = replicate_split(df)
    logger.info(
        "Replicate split: train=%d, val=%d, test=%d",
        len(split.train), len(split.val), len(split.test),
    )

    df_train = df.iloc[split.train].reset_index(drop=True)
    df_test = df.iloc[split.test].reset_index(drop=True)

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[list(TARGETS)].to_numpy()
    X_test = df_test[feature_cols].to_numpy()
    y_true = df_test[list(TARGETS)].to_numpy()

    # Strojenie hiperparametrow - tylko na zbiorze treningowym!
    logger.info("Tuning SVR hyperparameters via GridSearchCV ...")
    best_model, best_params, best_cv_score = tune_svr(X_train, y_train, seed=args.seed)

    # Zapisz najlepsze parametry do pliku tekstowego
    best_params_path = args.output_dir / "svr_best_params.txt"
    with open(best_params_path, "w") as f:
        f.write("SVR best hyperparameters (GridSearchCV, 3-fold CV)\n")
        f.write("=" * 50 + "\n\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest CV MAE: {best_cv_score:.4f}\n")
    logger.info("Wrote %s", best_params_path)

    # Predykcja na zbiorze testowym (nie widziany podczas strojenia)
    y_pred = best_model.predict(X_test)

    # Metryki
    pt = per_target_metrics(y_true, y_pred)
    pt["model"] = "svr_rbf_tuned"
    logger.info("SVR tuned test metrics:\n%s", pt.to_string(index=False))

    pc = per_condition_metrics(df_test, y_true, y_pred)
    pc["model"] = "svr_rbf_tuned"

    pt.to_csv(args.output_dir / "svr_per_target.csv", index=False)
    pc.to_csv(args.output_dir / "svr_per_condition.csv", index=False)
    logger.info("Wrote svr_per_target.csv and svr_per_condition.csv")

    # ===================================================================
    # CZĘŚĆ 2: LOCO CV z tymi samymi parametrami co znalezione w gridzie
    # ===================================================================
    # Wyciagamy parametry znalezione przez grid search
    tuned_C = best_params["svr__estimator__C"]
    tuned_gamma = best_params["svr__estimator__gamma"]
    tuned_kernel = best_params["svr__estimator__kernel"]
    tuned_epsilon = best_params["svr__estimator__epsilon"]

    logger.info("Running LOCO CV with tuned SVR (C=%s, gamma=%s) ...",
                tuned_C, tuned_gamma)
    logger.info("This will take a few minutes — be patient.")

    fold_results = []
    for fold_idx, ((T, RH), fold_split) in enumerate(loco_cv(df), start=1):
        df_fold_train = df.iloc[fold_split.train].reset_index(drop=True)
        df_fold_test = df.iloc[fold_split.test].reset_index(drop=True)

        X_fold_train = df_fold_train[feature_cols].to_numpy()
        y_fold_train = df_fold_train[list(TARGETS)].to_numpy()
        X_fold_test = df_fold_test[feature_cols].to_numpy()
        y_fold_true = df_fold_test[list(TARGETS)].to_numpy()

        # Swiezy pipeline z najlepszymi parametrami
        fold_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", MultiOutputRegressor(
                SVR(kernel=tuned_kernel, C=tuned_C, gamma=tuned_gamma,
                    epsilon=tuned_epsilon)
            )),
        ])
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(X_fold_test)

        pt_fold = per_target_metrics(y_fold_true, y_fold_pred)
        fold_results.append({"T": T, "RH": RH, "per_target": pt_fold})

        logger.info(
            "Fold %2d/35  T=%s RH=%s  MAE T=%.3f  MAE RH=%.3f",
            fold_idx, T, RH,
            pt_fold[pt_fold["target"] == "T"]["mae"].iloc[0],
            pt_fold[pt_fold["target"] == "RH"]["mae"].iloc[0],
        )

    folds_long: list[pd.DataFrame] = []
    for fr in fold_results:
        f = fr["per_target"].copy()
        f["T_holdout"] = fr["T"]
        f["RH_holdout"] = fr["RH"]
        folds_long.append(f)
    pd.concat(folds_long, ignore_index=True).to_csv(
        args.output_dir / "svr_loco_folds.csv", index=False,
    )
    summarise_loco(fold_results).to_csv(
        args.output_dir / "svr_loco_summary.csv", index=False,
    )
    logger.info("Wrote svr_loco_folds.csv and svr_loco_summary.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()