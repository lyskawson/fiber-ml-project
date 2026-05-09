"""Tests for fiber_ml.models.baseline and fiber_ml.eval.metrics.

Synthetic toy dataset is used so tests run in milliseconds without needing
the full Zarr file. The point is to verify shapes, schemas, and that the
pipeline runs end-to-end — not to measure model quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fiber_ml.eval.metrics import (
    per_condition_metrics,
    per_target_metrics,
    summarise_loco,
)
from fiber_ml.models.baseline import (
    TARGETS,
    fit_all_baselines,
    fit_baseline,
)
from fiber_ml.models.splits import replicate_split


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Toy dataset where T is a noisy linear function of features."""
    rng = np.random.default_rng(seed)
    rows = []
    for T in (20, 25, 30):
        for RH in (30, 50, 70):
            for r in range(1, 21):
                # Simple signal: ch1_mean ~ T, ch2_mean ~ T + RH
                ch1 = float(T) + rng.normal(0, 0.1)
                ch2 = float(T) + 0.05 * RH + rng.normal(0, 0.1)
                rows.append(
                    {
                        "experiment_id": f"T{T}_RH{RH}_R{r:02d}",
                        "T": T,
                        "RH": RH,
                        "replicate": r,
                        "ch1_mean": ch1,
                        "ch2_mean": ch2,
                        "diff_mean": ch2 - ch1,
                        # Pad to a few features
                        "ch1_std": rng.normal(1, 0.1),
                        "ch2_std": rng.normal(1, 0.1),
                    }
                )
    return pd.DataFrame(rows)


FEATURE_COLS = ["ch1_mean", "ch2_mean", "diff_mean", "ch1_std", "ch2_std"]


class TestFitBaseline:
    def test_ridge_fits_and_predicts(self) -> None:
        df = _synthetic_df()
        split = replicate_split(df)
        model = fit_baseline("ridge", df, FEATURE_COLS, split.train)
        preds = model.predict(df.iloc[split.test])
        assert preds.shape == (len(split.test), 2)

    @pytest.mark.parametrize(
        "name", ["ridge", "lasso", "random_forest", "gradient_boosting"]
    )
    def test_all_models_fit_and_predict(self, name: str) -> None:
        df = _synthetic_df()
        split = replicate_split(df)
        model = fit_baseline(name, df, FEATURE_COLS, split.train)  # type: ignore[arg-type]
        preds = model.predict(df.iloc[split.test])
        assert preds.shape == (len(split.test), 2)
        # Predictions for synthetic linear data should beat the mean baseline
        y_true = df.iloc[split.test][list(TARGETS)].to_numpy()
        baseline_mae = np.abs(y_true - y_true.mean(axis=0)).mean()
        model_mae = np.abs(y_true - preds).mean()
        assert model_mae < baseline_mae, (
            f"{name} did worse than predicting the mean — "
            f"got MAE {model_mae:.3f} vs baseline {baseline_mae:.3f}"
        )

    def test_unknown_model_raises(self) -> None:
        df = _synthetic_df()
        split = replicate_split(df)
        with pytest.raises(ValueError, match="Unknown model"):
            fit_baseline("xgboost", df, FEATURE_COLS, split.train)  # type: ignore[arg-type]

    def test_fit_all_baselines_returns_dict(self) -> None:
        df = _synthetic_df()
        split = replicate_split(df)
        models = fit_all_baselines(df, FEATURE_COLS, split.train)
        assert set(models.keys()) == {
            "ridge", "lasso", "random_forest", "gradient_boosting"
        }


class TestPerTargetMetrics:
    def test_perfect_prediction_zero_error(self) -> None:
        y = np.array([[20, 30], [25, 50], [30, 70]], dtype=float)
        out = per_target_metrics(y, y)
        assert (out["mae"] == 0).all()
        assert (out["rmse"] == 0).all()
        assert (out["max_abs"] == 0).all()

    def test_constant_offset(self) -> None:
        y_true = np.array([[20.0, 30.0], [25.0, 50.0], [30.0, 70.0]])
        y_pred = y_true + np.array([1.0, -2.0])
        out = per_target_metrics(y_true, y_pred)
        # Target T has +1 offset → MAE = 1; target RH has -2 offset → MAE = 2
        assert out.loc[out["target"] == "T", "mae"].iloc[0] == pytest.approx(1.0)
        assert out.loc[out["target"] == "RH", "mae"].iloc[0] == pytest.approx(2.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            per_target_metrics(np.zeros((3, 2)), np.zeros((4, 2)))


class TestPerConditionMetrics:
    def test_one_row_per_condition_per_target(self) -> None:
        df = _synthetic_df()
        split = replicate_split(df)
        df_test = df.iloc[split.test].reset_index(drop=True)
        y_true = df_test[list(TARGETS)].to_numpy()
        y_pred = y_true + np.random.default_rng(0).normal(0, 0.5, size=y_true.shape)
        out = per_condition_metrics(df_test, y_true, y_pred)
        # 3 T × 3 RH × 2 targets = 18 rows
        assert len(out) == 18
        assert set(out["target"].unique()) == {"T", "RH"}


class TestSummariseLoco:
    def test_aggregates_per_target(self) -> None:
        # Manual fold results
        f1 = pd.DataFrame(
            [
                {"target": "T", "mae": 1.0, "rmse": 1.2, "r2": 0.9, "max_abs": 2.0, "n_samples": 20},
                {"target": "RH", "mae": 2.0, "rmse": 2.5, "r2": 0.8, "max_abs": 5.0, "n_samples": 20},
            ]
        )
        f2 = pd.DataFrame(
            [
                {"target": "T", "mae": 0.8, "rmse": 1.0, "r2": 0.92, "max_abs": 1.8, "n_samples": 20},
                {"target": "RH", "mae": 2.4, "rmse": 2.9, "r2": 0.75, "max_abs": 6.0, "n_samples": 20},
            ]
        )
        out = summarise_loco([
            {"T": 20, "RH": 30, "per_target": f1},
            {"T": 25, "RH": 50, "per_target": f2},
        ])
        assert set(out["target"]) == {"T", "RH"}
        # T mean MAE = (1.0 + 0.8) / 2 = 0.9
        assert out.loc[out["target"] == "T", "mae_mean"].iloc[0] == pytest.approx(0.9)
        # RH max MAE = max(2.0, 2.4) = 2.4
        assert out.loc[out["target"] == "RH", "mae_max"].iloc[0] == pytest.approx(2.4)
