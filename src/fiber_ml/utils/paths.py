"""Central path definitions — all project paths go through here."""

from pathlib import Path

# Repo root is two levels above this file: src/fiber_ml/utils/paths.py
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

DATA_DIR: Path = REPO_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_SAMPLE_DIR: Path = DATA_DIR / "sample"
DATA_PROCESSED_DIR: Path = REPO_ROOT / "data_processed"

MANIFEST_PATH: Path = DATA_DIR / "manifest.csv"
REPORTS_DIR: Path = REPO_ROOT / "reports"
INGEST_REPORT_PATH: Path = REPORTS_DIR / "metrics" / "ingest_report.txt"
