"""Quick sanity check: parse sample files and print summary stats."""
import logging

from fiber_ml.ingest.parser import parse_file
from fiber_ml.utils.paths import DATA_SAMPLE_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    sample_files = sorted(DATA_SAMPLE_DIR.rglob("*.txt"))
    if not sample_files:
        logging.error("No .txt files found in %s", DATA_SAMPLE_DIR)
        return
    for fpath in sample_files:
        mf = parse_file(fpath)
        nan_counts = mf.data.isna().sum()
        print(
            f"
{fpath.name}
"
            f"  acquired_at : {mf.acquired_at}
"
            f"  n_points    : {mf.n_points}
"
            f"  data.shape  : {mf.data.shape}
"
            f"  NaN counts  : {nan_counts.to_dict()}"
        )


if __name__ == "__main__":
    main()
