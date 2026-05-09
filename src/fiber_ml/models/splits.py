"""Train/val/test splitters that respect the leakage requirements of section 7.

The project specification (section 7) is explicit: data must be split by
**experiment / session / condition**, never by individual rows. Replicates
of the same (T, RH) state are time-correlated within a session and therefore
cannot appear in both training and test sets.

This module provides three split strategies, in increasing order of
strictness:

1. :func:`replicate_split` — same condition appears in train and test, but
   different replicates. Default for baseline development. Closest analogue
   to the project's "validation set on remaining replicas" wording.

2. :func:`leave_one_condition_out` — one (T, RH) state is held out entirely.
   Cross-validation generator over all 35 conditions. Required by section 9
   ("scenarios where whole levels of T or RH are excluded from training").

3. :func:`leave_one_session_out` — placeholder for the session-level CV the
   project also requires. Not implementable until the manifest carries a
   ``session_id`` column. Currently raises NotImplementedError with a clear
   message so callers fail fast rather than silently using a wrong split.

All splitters return numpy index arrays that can be used directly with
``X.iloc[idx]`` / ``y.iloc[idx]`` on a DataFrame whose row order matches
the input.

## Anti-leakage tests

The companion test file ``tests/test_splits.py`` enforces the following
invariants for every splitter:

* No experiment_id appears in more than one of train/val/test.
* For ``leave_one_condition_out`` the held-out (T, RH) does not appear in
  the training fold.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class Split:
    """Indices for one train/val/test split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def assert_disjoint(self) -> None:
        """Verify no index appears in more than one fold (defensive check)."""
        sets = [set(self.train.tolist()), set(self.val.tolist()), set(self.test.tolist())]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                overlap = sets[i] & sets[j]
                if overlap:
                    names = ["train", "val", "test"]
                    raise ValueError(
                        f"Split has overlapping indices between {names[i]} and "
                        f"{names[j]}: {sorted(overlap)[:5]}..."
                    )


def replicate_split(
    df: "pd.DataFrame",
    train_replicates: tuple[int, ...] = tuple(range(1, 15)),
    val_replicates: tuple[int, ...] = tuple(range(15, 18)),
    test_replicates: tuple[int, ...] = tuple(range(18, 21)),
) -> Split:
    """Split by replicate number, keeping (T, RH) coverage in all folds.

    Default 14/3/3 split out of 20 replicates per condition. All 35 (T, RH)
    states appear in train, val, and test — but never the same replicate.
    Useful as a baseline regression target where you want to measure
    how well the model generalises across nominally identical sessions.

    Args:
        df: DataFrame with a ``replicate`` column.
        train_replicates: Replicate indices to use for training.
        val_replicates: Replicate indices for validation.
        test_replicates: Replicate indices for test.

    Returns:
        Split with row indices into ``df``.

    Raises:
        ValueError: If the three replicate sets overlap, or if any of the
            specified replicates is not present in the DataFrame.
    """
    if "replicate" not in df.columns:
        raise ValueError("DataFrame must have a 'replicate' column")

    all_specs = set(train_replicates) | set(val_replicates) | set(test_replicates)
    if len(all_specs) != len(train_replicates) + len(val_replicates) + len(test_replicates):
        raise ValueError("Replicate sets for train/val/test must be disjoint")

    available = set(df["replicate"].unique().tolist())
    missing = all_specs - available
    if missing:
        raise ValueError(f"Replicates not present in DataFrame: {sorted(missing)}")

    train = np.where(df["replicate"].isin(train_replicates))[0]
    val = np.where(df["replicate"].isin(val_replicates))[0]
    test = np.where(df["replicate"].isin(test_replicates))[0]

    split = Split(train=train, val=val, test=test)
    split.assert_disjoint()
    return split


def leave_one_condition_out(
    df: "pd.DataFrame",
    holdout_condition: tuple[int, int],
    val_replicates: tuple[int, ...] = (15, 16, 17),
) -> Split:
    """Hold out one (T, RH) condition entirely from training.

    Train: all replicates of all conditions except holdout, except those in
        val_replicates.
    Val: replicates listed in ``val_replicates`` for non-holdout conditions.
    Test: ALL replicates of the held-out (T, RH) condition.

    This is the strictest single-fold split: tests whether the model
    extrapolates to a fully unseen condition without ever having seen any
    of its replicates. Required by section 9 of the project description.

    Args:
        df: DataFrame with ``T``, ``RH``, ``replicate`` columns.
        holdout_condition: ``(T, RH)`` to hold out as test.
        val_replicates: Within non-holdout conditions, which replicates go
            to validation.

    Returns:
        Split.
    """
    for col in ("T", "RH", "replicate"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have a '{col}' column")

    T_test, RH_test = holdout_condition
    is_holdout = (df["T"] == T_test) & (df["RH"] == RH_test)
    if not is_holdout.any():
        raise ValueError(
            f"No rows match holdout condition T={T_test}, RH={RH_test}"
        )

    test = np.where(is_holdout)[0]
    in_val_rep = df["replicate"].isin(val_replicates)
    val = np.where(~is_holdout & in_val_rep)[0]
    train = np.where(~is_holdout & ~in_val_rep)[0]

    split = Split(train=train, val=val, test=test)
    split.assert_disjoint()
    return split


def loco_cv(df: "pd.DataFrame") -> Iterator[tuple[tuple[int, int], Split]]:
    """Iterator: leave-one-condition-out cross-validation over all conditions.

    Yields ``(condition, split)`` pairs where ``condition`` is the
    held-out ``(T, RH)``.

    Use for the section-9 cross-validation required by the project:

    >>> from sklearn.linear_model import Ridge
    >>> for cond, split in loco_cv(features):
    ...     model = Ridge().fit(X.iloc[split.train], y.iloc[split.train])
    ...     pred = model.predict(X.iloc[split.test])
    ...     # ... record metrics keyed by cond
    """
    conditions = (
        df[["T", "RH"]].drop_duplicates().sort_values(["T", "RH"]).itertuples(index=False)
    )
    for cond in conditions:
        T, RH = int(cond.T), int(cond.RH)
        yield (T, RH), leave_one_condition_out(df, holdout_condition=(T, RH))


def leave_one_session_out(df: "pd.DataFrame") -> Iterator[tuple[str, Split]]:
    """Placeholder — requires session_id in the manifest.

    Section 9 requires "leave-one-session-out" CV to test inter-session
    stability. This requires the manifest to carry a ``session_id`` column
    that groups replicates by acquisition session (e.g. one calendar day,
    one continuous campaign). The current manifest only has T/RH/replicate.

    To implement: extend ``scripts/01_build_manifest.py`` to assign session
    IDs (e.g. by ``acquired_at`` clustering with a one-hour gap threshold),
    then implement this function as a sibling of :func:`loco_cv` keyed on
    ``session_id``.
    """
    raise NotImplementedError(
        "leave_one_session_out requires 'session_id' in the manifest; "
        "extend scripts/01_build_manifest.py first. See module docstring."
    )
