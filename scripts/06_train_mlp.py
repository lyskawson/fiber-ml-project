"""Train MLP (sklearn) with extended GridSearchCV — Adam + L-BFGS solvers.

Sklearn MLPRegressor — gesta siec neuronowa z wbudowanym multi-output.
Strojenie przez GridSearchCV na zbiorze treningowym z dwoma solverami:
- Adam (z early stopping)
- L-BFGS (lepszy dla malych datasetow)

Usage:
    uv run python scripts/06_train_mlp.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/metrics

Produces:

* ``mlp_per_target.csv`` — held-out test MAE/RMSE/R² for the tuned MLP.
* ``mlp_per_condition.csv`` — same metrics broken down by (T, RH).
* ``mlp_loco_summary.csv`` — Leave-One-Condition-Out CV with tuned MLP.
* ``mlp_loco_folds.csv`` — full per-fold per-target detail.
* ``mlp_best_params.txt`` — wybrane hiperparametry.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Tlumimy ostrzezenia o niezbieznosci - grid search czasem znajdzie
# kombinacje ktora nie zbiega, to normalne, GridSearchCV i tak wybierze
# najlepsza z wszystkich
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def build_mlp_from_params(best_params: dict, seed: int = 42) -> Pipeline:
    """Buduje swiezy MLP pipeline na podstawie parametrow z GridSearch.

    Obsluguje rozne solvery (Adam, L-BFGS) - L-BFGS nie ma niektorych
    parametrow ktore ma Adam.
    """
    mlp_params = {
        "activation": "relu",
        "max_iter": 5000,
        "random_state": seed,
    }
    # Przepisz wszystkie parametry z prefiksem 'mlp__'
    for k, v in best_params.items():
        if k.startswith("mlp__"):
            mlp_params[k.replace("mlp__", "")] = v

    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(**mlp_params)),
    ])


def tune_mlp(X_train, y_train, seed: int = 42) -> tuple[Pipeline, dict, float]:
    """Strojenie hiperparametrow MLP przez GridSearchCV.

    Testuje dwa solvery:
    - Adam: 5 architektur × 4 alpha × 4 learning_rate = 80 kombinacji
    - L-BFGS: 5 architektur × 4 alpha = 20 kombinacji

    Razem 100 kombinacji × 3-fold CV = 300 trenowan.
    """
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            activation="relu",
            max_iter=5000,
            random_state=seed,
        )),
    ])

    # Lista slownikow - kazdy slownik to osobna siatka dla innego solvera.
    # GridSearchCV przetestuje wszystkie kombinacje z kazdej siatki.
    param_grid = [
        # Wariant 1: Adam z early stopping
        {
            "mlp__solver": ["adam"],
            "mlp__hidden_layer_sizes": [
                (64,),
                (64, 32),
                (64, 64),
                (128, 64),
                (128, 128),
            ],
            "mlp__alpha": [1e-6, 1e-5, 1e-4, 1e-3],
            "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
            "mlp__early_stopping": [True],
            "mlp__validation_fraction": [0.15],
            "mlp__n_iter_no_change": [30],
        },
        # Wariant 2: L-BFGS (lepszy dla malych datasetow)
        # L-BFGS nie uzywa learning_rate ani early_stopping
        {
            "mlp__solver": ["lbfgs"],
            "mlp__hidden_layer_sizes": [
                (64,),
                (64, 32),
                (64, 64),
                (128, 64),
                (128, 128),
            ],
            "mlp__alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        },
    ]

    adam_combos = 5 * 4 * 4
    lbfgs_combos = 5 * 4
    total_combos = adam_combos + lbfgs_combos
    logger.info("Grid search: %d Adam + %d L-BFGS = %d combinations × 3-fold CV = %d fits",
                adam_combos, lbfgs_combos, total_combos, total_combos * 3)

    grid = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    best_score = -grid.best_score_
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
    logger.info("Tuning MLP hyperparameters via GridSearchCV (Adam + L-BFGS) ...")
    best_model, best_params, best_cv_score = tune_mlp(X_train, y_train, seed=args.seed)

    # Zapisz parametry do raportu
    best_params_path = args.output_dir / "mlp_best_params.txt"
    with open(best_params_path, "w") as f:
        f.write("MLP best hyperparameters (GridSearchCV, 3-fold CV)\n")
        f.write("=" * 50 + "\n\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest CV MAE: {best_cv_score:.4f}\n")
    logger.info("Wrote %s", best_params_path)

    # Predykcja na zbiorze testowym
    y_pred = best_model.predict(X_test)

    pt = per_target_metrics(y_true, y_pred)
    pt["model"] = "mlp_tuned"
    logger.info("MLP tuned test metrics:\n%s", pt.to_string(index=False))

    pc = per_condition_metrics(df_test, y_true, y_pred)
    pc["model"] = "mlp_tuned"

    pt.to_csv(args.output_dir / "mlp_per_target.csv", index=False)
    pc.to_csv(args.output_dir / "mlp_per_condition.csv", index=False)
    logger.info("Wrote mlp_per_target.csv and mlp_per_condition.csv")

    # ===================================================================
    # CZĘŚĆ 2: LOCO CV z najlepszymi parametrami
    # ===================================================================
    logger.info("Running LOCO CV with tuned MLP (35 folds) ...")
    logger.info("Solver: %s, hidden: %s",
                best_params.get("mlp__solver"),
                best_params.get("mlp__hidden_layer_sizes"))

    fold_results = []
    for fold_idx, ((T, RH), fold_split) in enumerate(loco_cv(df), start=1):
        df_fold_train = df.iloc[fold_split.train].reset_index(drop=True)
        df_fold_test = df.iloc[fold_split.test].reset_index(drop=True)

        X_fold_train = df_fold_train[feature_cols].to_numpy()
        y_fold_train = df_fold_train[list(TARGETS)].to_numpy()
        X_fold_test = df_fold_test[feature_cols].to_numpy()
        y_fold_true = df_fold_test[list(TARGETS)].to_numpy()

        # Swiezy pipeline z najlepszymi parametrami
        fold_model = build_mlp_from_params(best_params, seed=args.seed)
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
        args.output_dir / "mlp_loco_folds.csv", index=False,
    )
    summarise_loco(fold_results).to_csv(
        args.output_dir / "mlp_loco_summary.csv", index=False,
    )
    logger.info("Wrote mlp_loco_folds.csv and mlp_loco_summary.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()