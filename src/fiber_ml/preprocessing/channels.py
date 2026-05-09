"""Channel extraction from Luna OBR-4600 raw measurement data.

This module resolves the ambiguity documented in CONTEXT.md Open Question #1:
how to map the two length ranges from ``opis_pomiarow_ML.txt`` onto the raw
.txt data, given that ``spectral_shift_ghz`` is non-null only for the first
~808 rows.

## Resolution

The two ``Length (m)`` columns in the OBR-4600 export represent different
quantities:

* ``length_1_m`` — **optical length along the fiber**, including the lead-in
  cable. Spans the full file (~2.59 m to ~3.39 m, monotonic, ~41 648 points
  at 0.1 mm spacing).
* ``length_2_m`` — **physical position on the sensor surface**, measured from
  the sensor's start. Populated only where the OBR has a coherent reflection
  signal — typically the first ~808 rows of the file. Spans 0 m to ~3.4 m.

The ranges given in ``opis_pomiarow_ML.txt`` (``2.65999–2.80083 m`` and
``3.22034–3.36018 m``) are sensor-coordinate ranges and refer to
``length_2_m``, not ``length_1_m``. Filtering by ``length_2_m`` recovers
exactly 141 and 140 points respectively, all with non-null spectral shift
and ``spectral_shift_quality`` between 0.43 and 0.85 — i.e. a real signal.
Filtering by ``length_1_m`` returns thousands of rows but zero non-null
spectral shift values.

This is consistent with how ODiSI / OBR systems present data: the first
``Length (m)`` column is the optical-domain coordinate (where the
backscatter sample lives in the fiber), and the second is the
spatial-domain coordinate after correlation against the reference scan.

## Channel definition (per opis_pomiarow_ML.txt)

* CH1_REF_T  — reference channel, dominant temperature sensitivity,
  ``length_2_m ∈ [2.65999, 2.80083]``, ~141 sensor points.
* CH2_PI_TRH — polyimide channel, T + RH sensitivity,
  ``length_2_m ∈ [3.22034, 3.36018]``, ~140 sensor points.

The pair ``(y1, y2)`` referenced in opis_pomiarow_ML.txt is constructed
position-wise: ``y1 = ch1.spectral_shift_ghz[i]``,
``y2 = ch2.spectral_shift_ghz[i]`` for ``i ∈ [0, n_points)``. Channels are
trimmed to a common length (``DEFAULT_N_POINTS = 140``) so this pairing is
1:1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Channel sensor-coordinate ranges, taken verbatim from opis_pomiarow_ML.txt
CH1_REF_T_RANGE_M: tuple[float, float] = (2.65999, 2.80083)
CH2_PI_TRH_RANGE_M: tuple[float, float] = (3.22034, 3.36018)

# Both channels yield 140-141 points at 1 mm spacing; trimming to 140 gives
# a 1:1 position-wise pairing across files.
DEFAULT_N_POINTS: int = 140

# Quality threshold below which a position is treated as a dropout candidate.
# Empirically, valid OBR readings have shift_quality > 0.3. This constant is
# used by the anomaly detection task and is exposed here for reuse.
DROPOUT_QUALITY_THRESHOLD: float = 0.15


class ChannelExtractionError(ValueError):
    """Raised when channel extraction cannot satisfy the requested layout."""


@dataclass(frozen=True)
class PairedChannels:
    """Two sensor channels paired position-wise, ready for feature engineering.

    Attributes:
        ch1: DataFrame with ``DEFAULT_N_POINTS`` rows holding CH1_REF_T data.
            Columns: ``length_2_m``, ``spectral_shift_ghz``,
            ``spectral_shift_quality``, ``amplitude_db_mm``.
        ch2: DataFrame with same shape and columns for CH2_PI_TRH.
        n_points: Number of paired positions (default 140).
    """

    ch1: pd.DataFrame
    ch2: pd.DataFrame
    n_points: int

    def paired_shifts(self) -> pd.DataFrame:
        """Return a DataFrame with one row per sensor position.

        Columns: ``pos_idx``, ``ch1_shift_ghz``, ``ch2_shift_ghz``,
        ``ch1_quality``, ``ch2_quality``. This is the
        ``(y1, y2)`` representation referenced in opis_pomiarow_ML.txt.
        """
        return pd.DataFrame(
            {
                "pos_idx": np.arange(self.n_points, dtype=np.int16),
                "ch1_shift_ghz": self.ch1["spectral_shift_ghz"].to_numpy(),
                "ch2_shift_ghz": self.ch2["spectral_shift_ghz"].to_numpy(),
                "ch1_quality": self.ch1["spectral_shift_quality"].to_numpy(),
                "ch2_quality": self.ch2["spectral_shift_quality"].to_numpy(),
            }
        )


def _slice_channel(
    raw: pd.DataFrame, l2_range: tuple[float, float], n_points: int
) -> pd.DataFrame:
    """Take a slice of raw rows where ``length_2_m`` lies in the given range.

    The result is trimmed (or NaN-padded) to exactly ``n_points`` rows so that
    paired channels can be aligned by position index across files.

    Raises:
        ChannelExtractionError: If the slice is empty (typically a malformed
            file or a wrong length range).
    """
    lo, hi = l2_range
    mask = raw["length_2_m"].between(lo, hi)
    seg = raw.loc[mask, ["length_2_m", "spectral_shift_ghz",
                         "spectral_shift_quality", "amplitude_db_mm"]].reset_index(drop=True)

    if len(seg) == 0:
        raise ChannelExtractionError(
            f"No rows match length_2_m in {l2_range}. "
            f"Verify that 'length_2_m' is populated and that the input is from "
            f"a Luna OBR-4600 measurement file (not pre-trimmed)."
        )

    if len(seg) >= n_points:
        return seg.iloc[:n_points].copy()

    pad = pd.DataFrame(
        np.nan,
        index=range(n_points - len(seg)),
        columns=seg.columns,
    )
    logger.warning(
        "Channel slice yielded only %d points for range %s; padded to %d with NaN.",
        len(seg), l2_range, n_points,
    )
    return pd.concat([seg, pad], ignore_index=True)


def extract_channels(
    raw: pd.DataFrame,
    n_points: int = DEFAULT_N_POINTS,
    ch1_range_m: tuple[float, float] = CH1_REF_T_RANGE_M,
    ch2_range_m: tuple[float, float] = CH2_PI_TRH_RANGE_M,
) -> PairedChannels:
    """Extract CH1_REF_T and CH2_PI_TRH from a parsed measurement DataFrame.

    Args:
        raw: DataFrame produced by :func:`fiber_ml.ingest.parser.parse_file`,
            i.e. with five columns including ``length_2_m`` and
            ``spectral_shift_ghz``.
        n_points: Common length to trim/pad both channels to. Default 140.
        ch1_range_m: ``length_2_m`` range for CH1_REF_T.
        ch2_range_m: ``length_2_m`` range for CH2_PI_TRH.

    Returns:
        PairedChannels with two ``n_points``-row DataFrames.

    Raises:
        ChannelExtractionError: If either channel slice is empty, which
            indicates the data was filtered or transformed before being
            passed in.
    """
    if "length_2_m" not in raw.columns:
        raise ChannelExtractionError(
            "Input DataFrame must contain 'length_2_m' column. Got: "
            f"{list(raw.columns)}"
        )

    ch1 = _slice_channel(raw, ch1_range_m, n_points)
    ch2 = _slice_channel(raw, ch2_range_m, n_points)

    return PairedChannels(ch1=ch1, ch2=ch2, n_points=n_points)
