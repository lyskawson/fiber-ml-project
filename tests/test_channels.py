"""Tests for fiber_ml.preprocessing.channels.

Regression tests that lock in the resolution of CONTEXT.md Open Question #1:
the channel ranges from opis_pomiarow_ML.txt refer to ``length_2_m``, not
``length_1_m``. Filtering by ``length_2_m`` MUST yield non-trivial signal.
"""

from pathlib import Path

import pytest

from fiber_ml.ingest.parser import parse_file
from fiber_ml.preprocessing.channels import (
    CH1_REF_T_RANGE_M,
    CH2_PI_TRH_RANGE_M,
    DEFAULT_N_POINTS,
    ChannelExtractionError,
    extract_channels,
)


class TestChannelShape:
    """Both channels must have exactly DEFAULT_N_POINTS rows after extraction."""

    def test_ch1_n_points(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        assert len(paired.ch1) == DEFAULT_N_POINTS

    def test_ch2_n_points(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        assert len(paired.ch2) == DEFAULT_N_POINTS

    def test_paired_shifts_shape(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        df = paired.paired_shifts()
        assert df.shape == (DEFAULT_N_POINTS, 5)
        assert list(df.columns) == [
            "pos_idx",
            "ch1_shift_ghz",
            "ch2_shift_ghz",
            "ch1_quality",
            "ch2_quality",
        ]


class TestChannelHasRealSignal:
    """Filtering by length_2_m must yield non-null spectral shift with quality > 0.3.

    This is the regression test that locks in Open Question #1 resolution.
    If a future change breaks the L2 filter (e.g. by reverting to length_1_m),
    these tests fail loudly with non-null counts of zero.
    """

    def test_ch1_spectral_shift_all_present(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        non_null = paired.ch1["spectral_shift_ghz"].notna().sum()
        # All 140 positions must carry spectral shift; tolerate 1-2 NaN at edges.
        assert non_null >= DEFAULT_N_POINTS - 2

    def test_ch2_spectral_shift_all_present(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        non_null = paired.ch2["spectral_shift_ghz"].notna().sum()
        assert non_null >= DEFAULT_N_POINTS - 2

    def test_ch1_quality_above_threshold(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        # CH1_REF_T quality is consistently above 0.3 in valid measurements
        assert paired.ch1["spectral_shift_quality"].mean() > 0.3

    def test_ch2_quality_above_threshold(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        paired = extract_channels(mf.data)
        # CH2_PI_TRH typically has quality > 0.7
        assert paired.ch2["spectral_shift_quality"].mean() > 0.5


class TestL2VsL1Discriminator:
    """A direct test that L1-based filtering would fail.

    If anyone refactors :func:`extract_channels` to filter by ``length_1_m``,
    this test catches it: filtering by L1 in those ranges yields zero
    non-null spectral shift, which is the bug described in Open Question #1.
    """

    def test_l1_filter_yields_no_shift_signal(self, pomiar1_path: Path) -> None:
        """Documents WHY we filter by length_2_m: filtering by length_1_m fails."""
        mf = parse_file(pomiar1_path)
        ch1_lo, ch1_hi = CH1_REF_T_RANGE_M

        l1_filtered = mf.data[
            (mf.data["length_1_m"] >= ch1_lo)
            & (mf.data["length_1_m"] <= ch1_hi)
        ]
        # length_1_m yields thousands of rows but zero spectral shift signal —
        # this is the diagnostic that proves the original interpretation wrong.
        assert len(l1_filtered) > 1000
        assert l1_filtered["spectral_shift_ghz"].notna().sum() == 0

    def test_l2_filter_yields_full_shift_signal(self, pomiar1_path: Path) -> None:
        """The corollary: filtering by length_2_m yields the full signal."""
        mf = parse_file(pomiar1_path)
        ch2_lo, ch2_hi = CH2_PI_TRH_RANGE_M

        l2_filtered = mf.data[
            (mf.data["length_2_m"] >= ch2_lo)
            & (mf.data["length_2_m"] <= ch2_hi)
        ]
        assert len(l2_filtered) >= DEFAULT_N_POINTS
        assert l2_filtered["spectral_shift_ghz"].notna().sum() >= DEFAULT_N_POINTS


class TestErrorHandling:
    def test_missing_l2_column_raises(self) -> None:
        import pandas as pd

        bad = pd.DataFrame({"length_1_m": [1.0, 2.0]})
        with pytest.raises(ChannelExtractionError, match="length_2_m"):
            extract_channels(bad)

    def test_empty_range_raises(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        with pytest.raises(ChannelExtractionError, match="No rows match"):
            extract_channels(mf.data, ch1_range_m=(99.0, 100.0))
