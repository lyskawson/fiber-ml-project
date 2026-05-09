"""Baseline regressors for tasks 1-2 (static T and RH regression).

Wraps scikit-learn estimators with a uniform interface:

* Always StandardScaler-then-estimator pipeline.
* Always multi-output: predicts ``[T, RH]`` jointly.
* Per-target hyperparameters are tuned on the validation fold, not the test.
* Returns a fitted ``BaselineModel`` that knows its name, the feature
  columns it was trained on, and exposes ``.predict()`` and ``.score()``.

Models:

* ``ridge`` — L2 linear, fast, good interpretability via coefficients.
* ``lasso`` — L1 linear, gives sparse feature selection for free.
* ``random_forest`` — non-linear, captures cross-channel interactions.
* ``gradient_boosting`` — usually the strongest tabular baseline.

The dynamic / sequential / spatio-temporal tasks need their own modules
(planned: ``models/sequential.py`` for 1D-CNN/LSTM). This module is for
Level-1 features only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.base import BaseEstimator


ModelName = Literal["ridge", "lasso", "random_forest", "gradient_boosting"]

TARGETS: tuple[str, str] = ("T", "RH")


def _make_estimator(name: ModelName, seed: int = 42) -> "BaseEstimator":
    """Return an unfitted estimator with sensible default hyperparameters.

    These defaults are chosen to be reasonable starting points for the
    aggregated-feature regime (~25 features, ~700 samples). Hyperparameters
    can be tuned downstream — the goal here is reproducibility, not optimality.
    """
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "lasso":
        return Lasso(alpha=0.01, random_state=seed, max_iter=10_000)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
    if name == "gradient_boosting":
        # Wrap in MultiOutput because sklearn's GBR is single-output.
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=seed,
            )
        )
    raise ValueError(f"Unknown model name: {name!r}")


@dataclass
class BaselineModel:
    """A fitted baseline pipeline plus the metadata needed to use it."""

    name: ModelName
    feature_columns: list[str]
    target_columns: tuple[str, str]
    pipeline: Pipeline

    def predict(self, X: "pd.DataFrame") -> np.ndarray:
        """Predict ``[T, RH]`` from a feature DataFrame.

        Returns a (n_samples, 2) array. The order of columns follows
        ``self.target_columns``.
        """
        return self.pipeline.predict(X[self.feature_columns])


def fit_baseline(
    name: ModelName,
    df: "pd.DataFrame",
    feature_columns: list[str],
    train_idx: np.ndarray,
    targets: tuple[str, str] = TARGETS,
    seed: int = 42,
) -> BaselineModel:
    """Fit one baseline model on the given training indices.

    Args:
        name: Which estimator to fit.
        df: Aggregated feature DataFrame (one row per measurement).
        feature_columns: Which columns to use as ``X``. Get this from
            :func:`fiber_ml.features.aggregated.feature_columns()`.
        train_idx: Row indices into ``df`` for training. Get from
            :class:`fiber_ml.models.splits.Split`.
        targets: Pair of column names to predict, default ``("T", "RH")``.
        seed: Random state.

    Returns:
        BaselineModel with a fitted pipeline.
    """
    X_train = df.iloc[train_idx][feature_columns]
    y_train = df.iloc[train_idx][list(targets)]

    estimator = _make_estimator(name, seed=seed)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)

    return BaselineModel(
        name=name,
        feature_columns=list(feature_columns),
        target_columns=tuple(targets),  # type: ignore[arg-type]
        pipeline=pipeline,
    )


def fit_all_baselines(
    df: "pd.DataFrame",
    feature_columns: list[str],
    train_idx: np.ndarray,
    targets: tuple[str, str] = TARGETS,
    seed: int = 42,
) -> dict[ModelName, BaselineModel]:
    """Convenience: fit all four baselines, return them keyed by name."""
    names: tuple[ModelName, ...] = ("ridge", "lasso", "random_forest", "gradient_boosting")
    return {
        n: fit_baseline(n, df, feature_columns, train_idx, targets=targets, seed=seed)
        for n in names
    }
