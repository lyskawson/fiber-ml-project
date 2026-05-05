
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "lyskawson/fiber-ml-luna-obr-4600"
REPO_TYPE = "dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token with write access (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--what",
        choices=["raw", "processed", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: provide --token or set HF_TOKEN env var", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    api = HfApi(token=args.token)

    print(f"Ensuring repo {REPO_ID} exists ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=True,
        exist_ok=True,
    )

    if args.dry_run:
        _print_plan(repo_root, args.what)
        return 0

    if args.what in ("processed", "all"):
        _upload_processed(api, repo_root)

    if args.what in ("raw", "all"):
        _upload_raw_per_condition(api, repo_root)

    print("\nAll uploads complete.")
    print(f"View: https://huggingface.co/datasets/{REPO_ID}/tree/main")
    return 0


def _print_plan(repo_root: Path, what: str) -> None:
    print(f"\nPlan: upload to {REPO_ID}")
    if what in ("processed", "all"):
        zarr_dir = repo_root / "data_processed" / "dataset.zarr"
        if zarr_dir.is_dir():
            n = sum(1 for _ in zarr_dir.rglob("*") if _.is_file())
            mb = _size_mb(zarr_dir)
            print(f"  processed/dataset.zarr   <- {zarr_dir}  ({n} files, {mb:.1f} MB)")
    if what in ("raw", "all"):
        raw_dir = repo_root / "data" / "raw"
        if raw_dir.is_dir():
            conditions = sorted(d for d in raw_dir.iterdir() if d.is_dir())
            print(f"  raw/  ({len(conditions)} condition folders)")
            for cond in conditions[:3]:
                n = sum(1 for _ in cond.iterdir() if _.is_file())
                mb = _size_mb(cond)
                print(f"    raw/{cond.name}  ({n} files, {mb:.1f} MB)")
            if len(conditions) > 3:
                print(f"    ... and {len(conditions) - 3} more")


def _upload_processed(api: HfApi, repo_root: Path) -> None:
    src = repo_root / "data_processed" / "dataset.zarr"
    if not src.is_dir():
        print(f"WARNING: {src} does not exist, skipping processed")
        return

    print(f"\n[processed] Uploading {src} -> processed/dataset.zarr ...")
    api.upload_folder(
        folder_path=str(src),
        path_in_repo="processed/dataset.zarr",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="upload processed/dataset.zarr",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
    )
    print("[processed] Done.")


def _upload_raw_per_condition(api: HfApi, repo_root: Path) -> None:
    raw_dir = repo_root / "data" / "raw"
    if not raw_dir.is_dir():
        print(f"WARNING: {raw_dir} does not exist, skipping raw")
        return

    conditions = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    n_conditions = len(conditions)

    print(f"\n[raw] Uploading {n_conditions} condition folders ...")
    for i, cond in enumerate(conditions, start=1):
        print(f"  [{i}/{n_conditions}] {cond.name} ...")
        api.upload_folder(
            folder_path=str(cond),
            path_in_repo=f"raw/{cond.name}",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"upload raw/{cond.name}",
            ignore_patterns=[".DS_Store", "**/.DS_Store"],
        )
    print("[raw] Done.")


def _size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


if __name__ == "__main__":
    sys.exit(main())
