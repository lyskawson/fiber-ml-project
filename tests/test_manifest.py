"""Tests for fiber_ml.ingest.manifest."""

from pathlib import Path

import pytest

from fiber_ml.ingest.manifest import MANIFEST_COLUMNS, build_manifest


class TestManifestColumns:
    def test_all_required_columns_present(self, sample_dir: Path) -> None:
        df = build_manifest(sample_dir.parent)
        for col in MANIFEST_COLUMNS:
            assert col in df.columns, f"Missing column: {col!r}"

    def test_sample_produces_two_rows(self, sample_dir: Path) -> None:
        df = build_manifest(sample_dir.parent)
        assert len(df) == 2

    def test_sample_t35_rh20(self, sample_dir: Path) -> None:
        df = build_manifest(sample_dir.parent)
        assert (df["T_celsius"] == 35).all()
        assert (df["RH_percent"] == 20).all()


class TestDuplicateMarkerDetection:
    def test_duplicate_marker_flagged(self, tmp_path: Path) -> None:
        """File named 'Pomiar10 (1).txt' must have has_duplicate_marker=True."""
        cond_dir = tmp_path / "T45_RH30"
        cond_dir.mkdir()

        # Minimal valid content for a 1-point measurement
        import textwrap

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
            Number of Data Points (in this file):  1
            NOTE: Data stored in this file is not decimated.

            Length (m)\tLength (m)\tAmplitude (dB/mm)\tSpectral Shift (GHz)\tSpectral Shift Quality\t
            1.0\t2.0\t-100.0\t-1.0\t0.5\t
        """)
        (cond_dir / "Pomiar10 (1).txt").write_text(content, encoding="utf-8")

        df = build_manifest(tmp_path)
        row = df[df["file_path"].str.contains("Pomiar10")]
        assert len(row) == 1
        assert bool(row.iloc[0]["has_duplicate_marker"]) is True

    def test_canonical_file_not_flagged(self, sample_dir: Path) -> None:
        df = build_manifest(sample_dir.parent)
        assert not df["has_duplicate_marker"].any()


class TestSkipNonMeasurementFile:
    def test_tif_in_manifest_with_note(self, tmp_path: Path) -> None:
        """A .tif file in a condition dir appears in manifest with a note, but is flagged."""
        cond_dir = tmp_path / "T75_RH40"
        cond_dir.mkdir()
        (cond_dir / "zakres_odpowiedzi_czujnika.tif").write_bytes(b"fake tif")

        df = build_manifest(tmp_path)
        tif_rows = df[df["file_path"].str.endswith(".tif")]
        assert len(tif_rows) == 1
        assert "non-measurement file" in tif_rows.iloc[0]["notes"]
        assert tif_rows.iloc[0]["experiment_id"] is None
