"""CLI: ingest .txt files from manifest into a Zarr dataset.

Usage:
    uv run python scripts/02_ingest_to_zarr.py \\
        --manifest data/manifest.csv \\
        --output data_processed/dataset.zarr

    # Sample-only run (uses data/sample/ files):
    uv run python scripts/02_ingest_to_zarr.py \\
        --manifest data/manifest_sample.csv \\
        --output /tmp/sample.zarr \\
        --sample
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest .txt files to Zarr dataset.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest.csv.")
    parser.add_argument("--output", required=True, type=Path, help="Output .zarr path.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Only ingest files from data/sample/ (for testing).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.manifest.exists():
        logging.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    from fiber_ml.ingest.to_zarr import ingest_to_zarr

    ingest_to_zarr(
        manifest_path=args.manifest,
        output_path=args.output,
        sample_only=args.sample,
    )


if __name__ == "__main__":
    main()
