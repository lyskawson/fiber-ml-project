"""Parser for Luna OBR-4600 .txt measurement files.

Each file has 14 header lines, 1 blank line, 1 column-header line, then 41648 data rows.
Columns: Length (m), Length (m), Amplitude (dB/mm), Spectral Shift (GHz), Spectral Shift Quality.

Spectral Shift and the second Length column are populated only for the first ~808 rows
(positions 2.592–2.607 m). Remaining rows have empty/NaN values in those columns.
See CONTEXT.md open_questions for details.

Example:
    >>> from pathlib import Path
    >>> from fiber_ml.ingest.parser import parse_file
    >>> mf = parse_file(Path("data/sample/T35_RH20/Pomiar1.txt"))
    >>> mf.n_points
    41648
    >>> mf.data.shape
    (41648, 5)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HEADER_LINES = 14
SKIP_ROWS = HEADER_LINES + 1  # 14 header + 1 blank line; row 16 (index 15) is the column header

COLUMN_NAMES = [
    "length_1_m",
    "length_2_m",
    "amplitude_db_mm",
    "spectral_shift_ghz",
    "spectral_shift_quality",
]

# Header keys that may span multiple spaces: "Key:  value" or "Key: value"
_KV_RE = re.compile(r"^(.+?):\s{2,}(.+)$")
# Fallback single-space key-value
_KV_SINGLE_RE = re.compile(r"^(.+?):\s(.+)$")
# "Acquired on DATE at TIME" / "Calibrated on DATE at TIME" — no colon
_TIMESTAMP_LINE_RE = re.compile(r"^(Acquired on|Calibrated on)\s+(.+)$")

_ACQUIRED_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{4})\s+at\s+(\d{1,2}:\d{2}:\d{2})$"
)


class ParseError(ValueError):
    """Raised when a measurement file fails validation."""


@dataclass
class MeasurementFile:
    """Parsed measurement file from Luna OBR-4600.

    Attributes:
        metadata: Raw key-value pairs from the file header.
        data: DataFrame with 5 columns and n_points rows. Sparse columns contain NaN.
        path: Source file path.
        n_points: Number of data rows (validated against header value).
        acquired_at: Measurement timestamp parsed from header.
    """

    metadata: dict[str, str]
    data: pd.DataFrame
    path: Path
    n_points: int
    acquired_at: datetime


def _parse_header(lines: list[str]) -> dict[str, str]:
    """Parse first HEADER_LINES lines into a metadata dict.

    Non-key-value lines (e.g. NOTE:, free-form text) are logged at DEBUG
    and skipped — not validated structurally.
    """
    metadata: dict[str, str] = {}
    for raw in lines:
        line = raw.rstrip("\r\n")
        m = _TIMESTAMP_LINE_RE.match(line) or _KV_RE.match(line) or _KV_SINGLE_RE.match(line)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            metadata[key] = value
        else:
            logger.debug("Header line not in key-value format (skipped): %r", line)
    return metadata


def _parse_acquired_at(metadata: dict[str, str]) -> datetime:
    """Extract acquisition datetime from metadata dict."""
    raw = metadata.get("Acquired on", "")
    m = _ACQUIRED_RE.match(raw)
    if not m:
        raise ParseError(f"Cannot parse 'Acquired on' timestamp: {raw!r}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%m/%d/%Y %H:%M:%S")


def parse_file(path: Path) -> MeasurementFile:
    """Parse a single Luna OBR-4600 .txt measurement file.

    Args:
        path: Path to the .txt file.

    Returns:
        MeasurementFile with validated metadata and raw data DataFrame.

    Raises:
        ParseError: If the number of data rows does not match the header declaration.
        FileNotFoundError: If path does not exist.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header_lines = [next(fh) for _ in range(HEADER_LINES)]

    metadata = _parse_header(header_lines)

    expected_n_points = int(metadata.get("Number of Data Points (in this file)", 0))

    # Read the data table: skip header block + blank line, row at SKIP_ROWS is column header
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=SKIP_ROWS,
        header=0,
        names=COLUMN_NAMES,
        engine="c",
        dtype=float,
        na_values=["", " "],
        usecols=range(5),
    )

    # Strip trailing tab that produces an unnamed 6th column in some rows
    if df.shape[1] > 5:
        df = df.iloc[:, :5]
    df.columns = COLUMN_NAMES

    n_points = len(df)
    if n_points != expected_n_points:
        raise ParseError(
            f"{path.name}: expected {expected_n_points} data points (from header) "
            f"but found {n_points} rows in the data table."
        )

    acquired_at = _parse_acquired_at(metadata)

    logger.debug(
        "Parsed %s: %d points, acquired %s, NaN counts: %s",
        path.name,
        n_points,
        acquired_at,
        df.isna().sum().to_dict(),
    )

    return MeasurementFile(
        metadata=metadata,
        data=df,
        path=path,
        n_points=n_points,
        acquired_at=acquired_at,
    )
