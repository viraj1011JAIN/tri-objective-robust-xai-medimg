"""
Phase 5.5 Orthogonality Analysis - CLI Wrapper

Command-line interface for running RQ1 orthogonality analysis.
Uses production-ready OrthogonalityAnalyzer from src.evaluation.orthogonality.

Example Usage:
    # Basic analysis
    python scripts/run_phase_5_5_analysis.py \\
        --dataset isic2018 \\
        --results-dir results/phase_5_baselines/isic2018 \\
        --output-dir results/phase_5_5_analysis

    # With custom seeds
    python scripts/run_phase_5_5_analysis.py \\
        --dataset isic2018 \\
        --results-dir results/phase_5_baselines/isic2018 \\
        --seeds 42 123 456 789 \\
        --output-dir results/phase_5_5_analysis

    # Skip LaTeX generation
    python scripts/run_phase_5_5_analysis.py \\
        --dataset isic2018 \\
        --results-dir results/phase_5_baselines/isic2018 \\
        --no-latex

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.orthogonality import OrthogonalityAnalyzer, OrthogonalityConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 5.5: RQ1 Orthogonality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., isic2018, derm7pt)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing model results (baseline, pgd_at, trades)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase_5_5_analysis"),
        help="Directory to save analysis outputs (default: results/phase_5_5_analysis)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds used for training (default: 42 123 456)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["baseline", "pgd_at", "trades"],
        help="Models to compare (default: baseline pgd_at trades)",
    )

    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="Statistical significance threshold (default: 0.05)",
    )

    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX table generation",
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )

    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Format for saved figures (default: pdf)",
    )

    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Phase 5.5: RQ1 Orthogonality Analysis")
    logger.info("=" * 80)

    # Create configuration
    config = OrthogonalityConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        dataset=args.dataset,
        models=args.models,
        significance_level=args.significance_level,
        generate_latex=not args.no_latex,
        save_figures=not args.no_figures,
        figure_format=args.figure_format,
        figure_dpi=args.figure_dpi,
    )

    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Results directory: {config.results_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(f"Models: {config.models}")

    # Run analysis
    analyzer = OrthogonalityAnalyzer(config)

    try:
        results = analyzer.run_analysis()

        logger.info("=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Orthogonality Confirmed: {results.is_orthogonal}")
        logger.info(f"Summary: {results.summary}")
        logger.info("")
        logger.info("Comparison Table:")
        logger.info(results.comparison_table.to_string(index=False))
        logger.info("")
        logger.info("Statistical Tests:")
        for test in results.statistical_tests:
            logger.info(f"  - {test.interpretation}")

        logger.info("=" * 80)
        logger.info(f"✓ Analysis complete. Results saved to: {config.output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
