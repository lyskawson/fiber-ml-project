"""Tests for fiber_ml.models.splits.

The most important test in this file is :class:`TestNoLeakage` — it locks in
the section-7 requirement that the same experiment never appears in two
folds. If a future refactor introduces leakage, those tests fail loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fiber_ml.models.splits import (
    Split,
    leave_one_condition_out,
    leave_one_session_out,
    loco_cv,
    replicate_split,
)


def _toy_df(conditions: list[tuple[int, int]] | None = None,
            n_replicates: int = 20) -> pd.DataFrame:
    """Build a minimal DataFrame mimicking the aggregated feature schema."""
    if conditions is None:
        conditions = [(t, rh) for t in (20, 25, 30) for rh in (30, 50, 70)]
    rows = []
    for T, RH in conditions:
        for r in range(1, n_replicates + 1):
            rows.append(
                {
                    "experiment_id": f"T{T}_RH{RH}_R{r:02d}",
                    "T": T,
                    "RH": RH,
                    "replicate": r,
                    # one dummy feature so callers can do X[features]
                    "ch1_mean": float(T) + 0.1 * r,
                }
            )
    return pd.DataFrame(rows)


class TestReplicateSplit:
    def test_default_split_sizes(self) -> None:
        df = _toy_df()
        split = replicate_split(df)
        # 9 conditions × 14 train + 3 val + 3 test replicates
        assert len(split.train) == 9 * 14
        assert len(split.val) == 9 * 3
        assert len(split.test) == 9 * 3

    def test_indices_disjoint(self) -> None:
        df = _toy_df()
        split = replicate_split(df)
        split.assert_disjoint()

    def test_all_conditions_in_each_fold(self) -> None:
        """Default replicate split keeps full (T, RH) coverage in train/val/test."""
        df = _toy_df()
        split = replicate_split(df)
        for idx in (split.train, split.val, split.test):
            sub = df.iloc[idx]
            assert sub[["T", "RH"]].drop_duplicates().shape[0] == 9

    def test_overlapping_replicates_raises(self) -> None:
        df = _toy_df()
        with pytest.raises(ValueError, match="disjoint"):
            replicate_split(df, train_replicates=(1, 2, 3),
                            val_replicates=(3, 4), test_replicates=(5, 6))

    def test_missing_replicate_raises(self) -> None:
        df = _toy_df(n_replicates=10)
        with pytest.raises(ValueError, match="not present"):
            replicate_split(df, train_replicates=(1, 2),
                            val_replicates=(3,), test_replicates=(50,))


class TestLeaveOneConditionOut:
    def test_holdout_fully_isolated(self) -> None:
        df = _toy_df()
        split = leave_one_condition_out(df, holdout_condition=(25, 50))
        # Test set: ALL replicates of (25, 50) — that's 20 rows.
        test_rows = df.iloc[split.test]
        assert (test_rows["T"] == 25).all()
        assert (test_rows["RH"] == 50).all()
        assert len(test_rows) == 20

    def test_train_does_not_contain_holdout(self) -> None:
        df = _toy_df()
        split = leave_one_condition_out(df, holdout_condition=(20, 30))
        train_rows = df.iloc[split.train]
        assert not ((train_rows["T"] == 20) & (train_rows["RH"] == 30)).any()

    def test_unknown_condition_raises(self) -> None:
        df = _toy_df()
        with pytest.raises(ValueError, match="No rows match"):
            leave_one_condition_out(df, holdout_condition=(999, 999))


class TestLOCO_CV:
    def test_iterates_over_all_conditions(self) -> None:
        df = _toy_df()
        all_folds = list(loco_cv(df))
        # 3 × 3 = 9 conditions
        assert len(all_folds) == 9
        seen = {cond for cond, _ in all_folds}
        assert seen == {(t, rh) for t in (20, 25, 30) for rh in (30, 50, 70)}

    def test_each_fold_disjoint(self) -> None:
        df = _toy_df()
        for _, split in loco_cv(df):
            split.assert_disjoint()


class TestNoLeakage:
    """Section-7 anti-leakage invariants. Critical."""

    def test_replicate_split_no_experiment_in_two_folds(self) -> None:
        df = _toy_df()
        split = replicate_split(df)
        train_ids = set(df.iloc[split.train]["experiment_id"].tolist())
        val_ids = set(df.iloc[split.val]["experiment_id"].tolist())
        test_ids = set(df.iloc[split.test]["experiment_id"].tolist())
        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()

    def test_loco_no_holdout_in_train(self) -> None:
        df = _toy_df()
        for (T, RH), split in loco_cv(df):
            train_rows = df.iloc[split.train]
            mask = (train_rows["T"] == T) & (train_rows["RH"] == RH)
            assert not mask.any(), (
                f"LOCO leakage: holdout ({T},{RH}) appears in train fold"
            )


class TestLeaveOneSessionOut:
    def test_raises_until_session_id_implemented(self) -> None:
        df = _toy_df()
        with pytest.raises(NotImplementedError, match="session_id"):
            next(leave_one_session_out(df))


class TestSplitDataclass:
    def test_assert_disjoint_catches_overlap(self) -> None:
        bad = Split(
            train=np.array([0, 1, 2]),
            val=np.array([2, 3]),
            test=np.array([4, 5]),
        )
        with pytest.raises(ValueError, match="overlapping"):
            bad.assert_disjoint()
