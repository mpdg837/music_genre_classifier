from argparse import ArgumentParser
import logging
from pathlib import Path

from midi_xai.interpretability.tcav.reports import write_tcav_aggregates

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="Aggregate TCAV summary scores into per-class tables and plots."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("checkpoints/tcav_scores/tcav_summary.csv"),
        help="Path to tcav_summary.csv produced by scripts/run_tcav_muserenet.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/tcav_scores/aggregated"),
        help="Directory for aggregated CSV tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures/tcav"),
        help="Directory for TCAV heatmaps and per-class bar charts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of strongest concepts to show on each per-class bar chart.",
    )
    parser.add_argument(
        "--significant-only",
        action="store_true",
        help="Exclude non-significant rows from the ranked CSV table.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV aggregates only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    outputs = write_tcav_aggregates(
        summary_path=args.summary_csv,
        output_dir=args.output_dir,
        figures_dir=None if args.no_plots else args.figures_dir,
        top_k=args.top_k,
        include_non_significant=not args.significant_only,
    )

    logger.info("Wrote annotated TCAV summary to %s", outputs["annotated"])
    logger.info("Wrote per-class TCAV ranking to %s", outputs["ranked"])
    logger.info("Wrote layer-aware TCAV matrix to %s", outputs["class_layer_matrix"])
    logger.info("Wrote all-layer TCAV matrix to %s", outputs["class_matrix"])
    if outputs["figures"]:
        logger.info("Wrote %s TCAV figures to %s", len(outputs["figures"]), args.figures_dir)


if __name__ == "__main__":
    main()
