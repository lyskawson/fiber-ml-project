"""Build manifest.csv from a directory tree of OBR-4600 measurement files.

Scans T{T}_RH{RH}/ subdirectories, extracts condition labels and replicate numbers,
computes SHA-256 hashes, and flags anomalies (duplicate markers, non-measurement files).

Example:
    >>> from pathlib import Path
    >>> from fiber_ml.ingest.manifest import build_manifest
    >>> df = build_manifest(Path("data/raw"))
    >>> df.columns.tolist()
    ['experiment_id', 'file_path', 'T_celsius', 'RH_percent', 'replicate', ...]
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from fiber_ml.ingest.parser import ParseError, parse_file

logger = logging.getLogger(__name__)

# Matches T{int}_RH{int} directory names
_CONDITION_RE = re.compile(r"^T(\d+)_RH(\d+)$")
# Matches Pomiar{int} with optional duplicate marker e.g. "Pomiar10 (1).txt"
_POMIAR_RE = re.compile(r"^Pomiar(\d+)(\s*\(\d+\))?\.txt$", re.IGNORECASE)

MANIFEST_COLUMNS = [
    "experiment_id",
    "file_path",
    "T_celsius",
    "RH_percent",
    "replicate",
    "acquired_at",
    "n_points",
    "file_size_bytes",
    "sha256",
    "has_duplicate_marker",
    "notes",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_dir: Path) -> pd.DataFrame:
    """Scan data_dir for T{T}_RH{RH}/ subdirectories and build a manifest DataFrame.

    Args:
        data_dir: Root directory containing T{T}_RH{RH}/ condition subdirectories.

    Returns:
        DataFrame with MANIFEST_COLUMNS. Includes non-measurement files with a note.
        Files with duplicate markers are flagged but not removed.
    """
    rows: list[dict] = []

    condition_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and _CONDITION_RE.match(d.name)
    )

    if not condition_dirs:
        logger.warning("No T{T}_RH{RH} directories found in %s", data_dir)

    for cond_dir in condition_dirs:
        m_cond = _CONDITION_RE.match(cond_dir.name)
        assert m_cond  # guaranteed by filter above
        t_celsius = int(m_cond.group(1))
        rh_percent = int(m_cond.group(2))

        for fpath in sorted(cond_dir.iterdir()):
            if not fpath.is_file():
                continue

            m_pomiar = _POMIAR_RE.match(fpath.name)

            if m_pomiar is None:
                # Non-measurement file (e.g. .tif, other formats)
                logger.debug("Non-measurement file, flagging in manifest: %s", fpath)
                rows.append(
                    {
                        "experiment_id": None,
                        "file_path": str(fpath),
                        "T_celsius": t_celsius,
                        "RH_percent": rh_percent,
                        "replicate": None,
                        "acquired_at": None,
                        "n_points": None,
                        "file_size_bytes": fpath.stat().st_size,
                        "sha256": _sha256(fpath),
                        "has_duplicate_marker": False,
                        "notes": "non-measurement file, skipped",
                    }
                )
                continue

            replicate = int(m_pomiar.group(1))
            has_duplicate_marker = m_pomiar.group(2) is not None

            if has_duplicate_marker:
                logger.warning(
                    "Duplicate marker detected in filename: %s — verify against canonical file",
                    fpath,
                )

            experiment_id = f"T{t_celsius}_RH{rh_percent}_R{replicate:02d}"

            acquired_at = None
            n_points = None
            notes = ""
            try:
                mf = parse_file(fpath)
                acquired_at = mf.acquired_at
                n_points = mf.n_points
            except ParseError as exc:
                notes = f"parse_error: {exc}"
                logger.error("ParseError for %s: %s", fpath, exc)
            except Exception as exc:  # noqa: BLE001
                notes = f"unexpected_error: {exc}"
                logger.error("Unexpected error for %s: %s", fpath, exc)

            rows.append(
                {
                    "experiment_id": experiment_id,
                    "file_path": str(fpath),
                    "T_celsius": t_celsius,
                    "RH_percent": rh_percent,
                    "replicate": replicate,
                    "acquired_at": acquired_at,
                    "n_points": n_points,
                    "file_size_bytes": fpath.stat().st_size,
                    "sha256": _sha256(fpath),
                    "has_duplicate_marker": has_duplicate_marker,
                    "notes": notes,
                }
            )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info(
        "Manifest built: %d measurement files, %d non-measurement, %d duplicate markers",
        df["experiment_id"].notna().sum(),
        (df["notes"] == "non-measurement file, skipped").sum(),
        df["has_duplicate_marker"].sum(),
    )
    return df
