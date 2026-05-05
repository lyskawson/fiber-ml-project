"""Download dataset from Hugging Face Hub.

Run this after `git clone && uv sync` to get raw measurements
and processed Zarr from HF Hub.

Requires HF_TOKEN env var or --token argument with at least 'read' scope.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "lyskawson/fiber-ml-luna-obr-4600"
REPO_TYPE = "dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--what",
        choices=["raw", "processed", "all"],
        default="all",
    )
    args = parser.parse_args()

    if not args.token:
        print("ERROR: provide --token or set HF_TOKEN env var", file=sys.stderr)
        print("Get token from https://huggingface.co/settings/tokens", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent

    allow_patterns: list[str] = []
    if args.what in ("raw", "all"):
        allow_patterns.append("raw/**")
    if args.what in ("processed", "all"):
        allow_patterns.append("processed/**")

    print(f"Downloading from {REPO_ID} (patterns: {allow_patterns}) ...")

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=args.token,
        allow_patterns=allow_patterns,
        local_dir=str(repo_root / ".hf_cache"),
    )

    cache_path = Path(local_dir)
    print(f"  Downloaded to {cache_path}")

    if "raw/**" in allow_patterns:
        src = cache_path / "raw"
        dst = repo_root / "data" / "raw"
        if src.is_dir():
            _sync_dir(src, dst)
            print(f"  Synced {src} -> {dst}")

    if "processed/**" in allow_patterns:
        src = cache_path / "processed"
        dst = repo_root / "data_processed"
        if src.is_dir():
            for item in src.iterdir():
                _sync_dir(item, dst / item.name)
            print(f"  Synced {src} -> {dst}")

    print("\nDone. Validation:")
    n_raw = len(list((repo_root / "data" / "raw").rglob("*.txt")))
    print(f"  Raw .txt files: {n_raw} (expected: 700)")
    zarr = repo_root / "data_processed" / "dataset.zarr"
    print(f"  Zarr exists: {zarr.is_dir()}")

    return 0


def _sync_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


if __name__ == "__main__":
    sys.exit(main())
