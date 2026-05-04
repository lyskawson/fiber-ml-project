"""CLI: scan data directory and build manifest.csv.

Usage:
    uv run python scripts/01_build_manifest.py --data-dir data/raw --output data/manifest.csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest CSV from raw data directory.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root data directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest CSV path.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.data_dir.exists():
        logging.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    from fiber_ml.ingest.manifest import build_manifest

    df = build_manifest(args.data_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info("Manifest saved to %s (%d rows)", args.output, len(df))


if __name__ == "__main__":
    main()
