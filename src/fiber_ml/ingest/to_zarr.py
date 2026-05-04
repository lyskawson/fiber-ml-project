"""Batch ingest: .txt measurement files → Zarr v2 dataset via xarray.

Schema:
    Dimensions: experiment (N), position (41648), channel (5)
    Variables:
        data(experiment, position, channel): float32
        T(experiment): int8
        RH(experiment): int8
        replicate(experiment): int8
        acquired_at(experiment): datetime64[ns]
        experiment_id(experiment): <U16
    Coordinates:
        channel: ['length_1_m', 'length_2_m', 'amplitude_db_mm',
                  'spectral_shift_ghz', 'spectral_shift_quality']
        position: 0..N_POINTS-1
    Root attrs: project, sensor_model, source, created_at, n_files, open_questions

Example:
    >>> from pathlib import Path
    >>> from fiber_ml.ingest.to_zarr import ingest_to_zarr
    >>> ingest_to_zarr(
    ...     manifest_path=Path("data/manifest_sample.csv"),
    ...     output_path=Path("/tmp/sample.zarr"),
    ...     sample_only=True,
    ... )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

from fiber_ml.ingest.parser import parse_file
from fiber_ml.utils.paths import INGEST_REPORT_PATH

logger = logging.getLogger(__name__)

CHANNEL_NAMES = [
    "length_1_m",
    "length_2_m",
    "amplitude_db_mm",
    "spectral_shift_ghz",
    "spectral_shift_quality",
]

COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def ingest_to_zarr(
    manifest_path: Path,
    output_path: Path,
    sample_only: bool = False,
) -> None:
    """Ingest .txt files listed in manifest into a Zarr v2 dataset.

    Args:
        manifest_path: Path to manifest.csv produced by build_manifest.
        output_path: Destination path for the .zarr store.
        sample_only: If True, only ingest rows from data/sample/ subdirectories.
    """
    manifest = pd.read_csv(manifest_path, parse_dates=["acquired_at"])

    # Drop non-measurement files
    rows = manifest[manifest["experiment_id"].notna()].copy()

    if sample_only:
        rows = rows[rows["file_path"].str.contains("/sample/")]
        logger.info("--sample mode: ingesting %d files from data/sample/", len(rows))

    # Skip duplicate-marked files with a warning
    dupes = rows[rows["has_duplicate_marker"]]
    if not dupes.empty:
        logger.warning(
            "%d files have duplicate markers and will be SKIPPED during ingest. "
            "Verify against canonical files: %s",
            len(dupes),
            dupes["file_path"].tolist(),
        )
        rows = rows[~rows["has_duplicate_marker"]]

    rows = rows.sort_values(["T_celsius", "RH_percent", "replicate"]).reset_index(drop=True)
    n_exp = len(rows)
    logger.info("Ingesting %d experiments to %s", n_exp, output_path)

    # Determine n_points from first file
    first_file = parse_file(Path(rows.iloc[0]["file_path"]))
    n_points = first_file.n_points
    n_channels = len(CHANNEL_NAMES)

    # Pre-allocate array — float32 to keep memory reasonable
    data_arr = np.full((n_exp, n_points, n_channels), np.nan, dtype=np.float32)

    acquired_at_arr = np.empty(n_exp, dtype="datetime64[ns]")
    t_arr = rows["T_celsius"].to_numpy(dtype=np.int8)
    rh_arr = rows["RH_percent"].to_numpy(dtype=np.int8)
    rep_arr = rows["replicate"].to_numpy(dtype=np.int8)
    exp_ids = rows["experiment_id"].to_numpy()

    total_bytes = 0

    for i, (_, row) in enumerate(rows.iterrows()):
        fpath = Path(row["file_path"])
        mf = parse_file(fpath)
        data_arr[i] = mf.data.to_numpy(dtype=np.float32)
        acquired_at_arr[i] = np.datetime64(mf.acquired_at, "ns")
        total_bytes += fpath.stat().st_size

        if (i + 1) % 50 == 0 or (i + 1) == n_exp:
            logger.info("  Parsed %d / %d files", i + 1, n_exp)

    ds = xr.Dataset(
        {
            "data": xr.DataArray(
                data_arr,
                dims=["experiment", "position", "channel"],
                attrs={"units": "mixed", "long_name": "raw sensor data"},
            ),
            "T": xr.DataArray(t_arr, dims=["experiment"], attrs={"units": "celsius"}),
            "RH": xr.DataArray(rh_arr, dims=["experiment"], attrs={"units": "percent"}),
            "replicate": xr.DataArray(rep_arr, dims=["experiment"]),
            "acquired_at": xr.DataArray(acquired_at_arr, dims=["experiment"]),
            "experiment_id": xr.DataArray(exp_ids, dims=["experiment"]),
        },
        coords={
            "channel": CHANNEL_NAMES,
            "position": np.arange(n_points, dtype=np.int32),
        },
        attrs={
            "project": "fiber-ml-project",
            "sensor_model": "Luna OBR-4600",
            "source": str(manifest_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_files": n_exp,
            "open_questions": (
                "Spectral Shift (GHz) and second Length (m) are non-null only for the "
                "first ~808 positions (2.592-2.607 m range). "
                "The opis_pomiarow_ML.txt specifies feature ranges 2.65999-2.80083 and "
                "3.22034-3.36018 m, where Spectral Shift is empty. "
                "Awaiting clarification from supervisor before feature engineering."
            ),
        },
    )

    encoding = {
        "data": {
            "compressor": COMPRESSOR,
            "chunks": [1, n_points, n_channels],
            "dtype": "float32",
        }
    }

    ds.to_zarr(str(output_path), mode="w", encoding=encoding)
    logger.info("Zarr written: %s", output_path)

    _write_ingest_report(ds, rows, total_bytes)


def _write_ingest_report(
    ds: xr.Dataset, rows: pd.DataFrame, total_bytes: int
) -> None:
    """Write a text summary report of the ingest run."""
    INGEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data_arr = ds["data"].values
    n_exp, n_points, n_channels = data_arr.shape

    lines: list[str] = [
        "=== Ingest Report ===",
        f"Created:        {datetime.now(timezone.utc).isoformat()}",
        f"Files ingested: {n_exp}",
        f"Total bytes:    {total_bytes:,}",
        f"Dataset shape:  {data_arr.shape}  (experiment × position × channel)",
        "",
        "--- Conditions ---",
    ]

    for (t, rh), grp in rows.groupby(["T_celsius", "RH_percent"]):
        lines.append(f"  T={t:3d}°C  RH={rh:3d}%  → {len(grp):3d} files")

    lines += ["", "--- NaN counts per channel (global) ---"]
    channel_names = CHANNEL_NAMES
    for c_idx, ch_name in enumerate(channel_names):
        channel_data = data_arr[:, :, c_idx]
        nan_count = int(np.isnan(channel_data).sum())
        nan_pct = 100.0 * nan_count / channel_data.size
        lines.append(f"  {ch_name:<30s}  {nan_count:>12,}  ({nan_pct:.2f}%)")

    lines += ["", "--- NaN counts per experiment (first 10) ---"]
    for i in range(min(10, n_exp)):
        exp_id = rows.iloc[i]["experiment_id"]
        nan_count = int(np.isnan(data_arr[i]).sum())
        lines.append(f"  {exp_id:<20s}  {nan_count:>8,} NaN")

    report_text = "\n".join(lines) + "\n"
    INGEST_REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Ingest report written to %s", INGEST_REPORT_PATH)
