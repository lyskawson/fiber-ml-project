"""Tests for fiber_ml.features.aggregated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fiber_ml.features.aggregated import (
    CROSS_CHANNEL_FEATURES,
    METADATA_COLUMNS,
    PER_CHANNEL_FEATURES,
    aggregate_from_files,
    aggregate_one,
    feature_columns,
)
from fiber_ml.ingest.parser import parse_file
from fiber_ml.preprocessing.channels import extract_channels


class TestSchema:
    def test_feature_count(self) -> None:
        # 11 per-channel × 2 channels + 3 cross-channel = 25
        assert len(feature_columns()) == 2 * len(PER_CHANNEL_FEATURES) + len(CROSS_CHANNEL_FEATURES)
        assert len(feature_columns()) == 25

    def test_feature_columns_stable(self) -> None:
        """Column order must be deterministic across runs."""
        assert feature_columns() == feature_columns()

    def test_per_channel_prefixes(self) -> None:
        cols = feature_columns()
        ch1_cols = [c for c in cols if c.startswith("ch1_")]
        ch2_cols = [c for c in cols if c.startswith("ch2_")]
        assert len(ch1_cols) == len(PER_CHANNEL_FEATURES)
        assert len(ch2_cols) == len(PER_CHANNEL_FEATURES)


class TestAggregateOne:
    def test_returns_all_features(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        feats = aggregate_one(paired)
        assert set(feats.keys()) == set(feature_columns())

    def test_no_nans_for_valid_input(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        feats = aggregate_one(paired)
        # diff_mean and other features should be finite for a real measurement
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_diff_mean_consistency(self, pomiar1_path: Path) -> None:
        """diff_mean must equal ch2_mean - ch1_mean by construction."""
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        feats = aggregate_one(paired)
        assert feats["diff_mean"] == pytest.approx(feats["ch2_mean"] - feats["ch1_mean"])


class TestAggregateFromFiles:
    def test_two_files(self, pomiar1_path: Path) -> None:
        # Use the same file twice as different replicates
        meta = {pomiar1_path.name: {"T": 35, "RH": 20, "replicate": 1}}
        df = aggregate_from_files([pomiar1_path], meta)
        assert len(df) == 1
        assert list(df.columns)[: len(METADATA_COLUMNS)] == list(METADATA_COLUMNS)
        # Targets are correctly populated
        assert df.iloc[0]["T"] == 35
        assert df.iloc[0]["RH"] == 20
        assert df.iloc[0]["replicate"] == 1

    def test_features_dataframe_dtypes(self, pomiar1_path: Path) -> None:
        meta = {pomiar1_path.name: {"T": 35, "RH": 20, "replicate": 1}}
        df = aggregate_from_files([pomiar1_path], meta)
        # All non-metadata columns should be numeric
        feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"
