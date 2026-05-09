"""Generate the EDA plot set from a Zarr dataset.

Usage:
    uv run python scripts/03_eda_report.py \
        --zarr data_processed/dataset.zarr \
        --output-dir reports/figures/eda

Produces five PNGs (state_coverage, channel_profiles, response_surface,
replicate_variance, quality_distribution) plus a text summary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import xarray as xr

from fiber_ml.eda.plots import (
    aggregate_from_zarr,
    plot_channel_profiles,
    plot_quality_distribution,
    plot_replicate_variance,
    plot_response_surface,
    plot_state_coverage,
)
from fiber_ml.utils.paths import DATA_PROCESSED_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr",
        type=Path,
        default=DATA_PROCESSED_DIR / "dataset.zarr",
        help="Path to Zarr dataset (default: data_processed/dataset.zarr).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "figures" / "eda",
        help="Where to save PNGs (default: reports/figures/eda/).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=50,
        help="Sample this many experiments for the quality histogram.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening Zarr dataset: %s", args.zarr)
    ds = xr.open_zarr(str(args.zarr))

    logger.info(
        "Loaded dataset: %d experiments × %d positions × %d channels",
        ds.sizes["experiment"], ds.sizes["position"], ds.sizes["channel"],
    )

    logger.info("Aggregating per-experiment features ...")
    agg = aggregate_from_zarr(ds)
    agg_path = args.output_dir / "aggregated.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Wrote %s", agg_path)

    targets = {
        "state_coverage.png": plot_state_coverage(agg),
        "channel_profiles.png": plot_channel_profiles(ds),
        "response_surface.png": plot_response_surface(agg),
        "replicate_variance.png": plot_replicate_variance(agg),
        "quality_distribution.png": plot_quality_distribution(
            ds, sample_n=args.sample_n
        ),
    }
    for name, fig in targets.items():
        out = args.output_dir / name
        fig.savefig(out, dpi=130, bbox_inches="tight")
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
