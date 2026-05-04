"""Tests for fiber_ml.ingest.parser."""

import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from fiber_ml.ingest.parser import COLUMN_NAMES, ParseError, parse_file


class TestParsePomiar1Metadata:
    def test_n_points(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert mf.n_points == 41648

    def test_acquired_at_is_datetime(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert isinstance(mf.acquired_at, datetime)

    def test_acquired_at_value(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert mf.acquired_at == datetime(2026, 3, 23, 20, 45, 11)

    def test_required_metadata_keys(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        required = [
            "Acquired on",
            "Calibrated on",
            "Spatial Resolution (mm)",
            "Gage length",
            "Sensor spacing",
            "Number of Data Points (in this file)",
        ]
        for key in required:
            assert key in mf.metadata, f"Missing metadata key: {key!r}"


class TestParsePomiar1DataShape:
    def test_row_count(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert len(mf.data) == 41648

    def test_column_count(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert mf.data.shape[1] == 5

    def test_column_names(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert list(mf.data.columns) == COLUMN_NAMES

    def test_dtype_float(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        assert all(mf.data[col].dtype.kind == "f" for col in mf.data.columns)


class TestParseValidation:
    def test_n_points_mismatch_raises(self, tmp_path: Path) -> None:
        """A file declaring N points but containing N+1 rows raises ParseError."""
        # Construct a minimal valid header declaring 2 points but with 3 data rows
        content = textwrap.dedent("""\
            Acquired on 1/1/2025 at 10:00:00
            Calibrated on 1/1/2023 at 00:00:00
            Device Descriptor:  [none]
            Scan Range (nm):  1545.000  --  1588.000
            Measurement Type:  Reflection
            Group Index:  1.500
            Gain:  24 dB
            Domain:  Time
            Spatial Resolution (mm):  0.100
            Frequency domain window was not applied to measurement data.
            Gage length:  1.000 cm
            Sensor spacing:  0.100 cm
            Number of Data Points (in this file):  2
            NOTE: Data stored in this file is not decimated.

            Length (m)\tLength (m)\tAmplitude (dB/mm)\tSpectral Shift (GHz)\tSpectral Shift Quality\t
            1.0\t2.0\t-100.0\t-1.0\t0.5\t
            2.0\t3.0\t-101.0\t-2.0\t0.6\t
            3.0\t4.0\t-102.0\t-3.0\t0.7\t
        """)
        f = tmp_path / "bad.txt"
        f.write_text(content, encoding="utf-8")

        with pytest.raises(ParseError, match="expected 2 data points.*found 3"):
            parse_file(f)


class TestSpectralShiftSparsity:
    """Regression test: documents expected sparsity of Spectral Shift column.

    The first ~808 rows contain valid Spectral Shift values; the rest are NaN.
    This test will break if the file format changes in a future measurement campaign.
    See CONTEXT.md open_questions for details.
    """

    def test_spectral_shift_non_null_count(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        non_null = mf.data["spectral_shift_ghz"].notna().sum()
        # Allow ±50 around expected ~808 to accommodate minor file variation
        assert 700 < non_null < 900, (
            f"Unexpected non-null Spectral Shift count: {non_null}. "
            "File format may have changed — verify with supervisor."
        )

    def test_spectral_shift_quality_non_null_count(self, pomiar1_path: Path) -> None:
        mf = parse_file(pomiar1_path)
        non_null = (mf.data["spectral_shift_quality"] != 0).sum()
        # Quality is stored as 0.0 (not NaN) for the sparse region
        assert non_null > 700
