#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.train.triobj_training import (  # and TrainingConfig if you need it
    TriObjectiveTrainer,
)
from src.utils.config import (
    ExperimentConfig,
    get_config_hash,
    load_experiment_config,
    save_resolved_config,
)
from src.utils.reproducibility import (
    reproducibility_header,
    set_global_seed,
    summarise_reproducibility_state,
)

LOGGER = logging.getLogger("train_tri_objective")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tri-objective robust XAI training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        nargs="+",
        required=True,
        help=(
            "Config YAML paths (in order). Later files override earlier ones. "
            "Typical pattern: base, dataset, model, experiment."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for reproducibility.seed in the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for training.device (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--save-config-to",
        type=str,
        default=None,
        help="Optional path to save the fully-resolved config YAML used for this run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, run in debug mode (trainer decides what this means).",
    )
    return parser.parse_args(argv)


def _override_config_from_cli(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    # Override seed if requested
    if args.seed is not None:
        cfg.reproducibility.seed = args.seed

    # Override device if requested
    if args.device is not None:
        cfg.training.device = args.device

    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_args(argv)

    LOGGER.info("Loading experiment config from: %s", ", ".join(args.config))
    cfg = load_experiment_config(*args.config)
    cfg = _override_config_from_cli(cfg, args)

    # Set seeds and log reproducibility state
    state = set_global_seed(
        cfg.reproducibility.seed,
        deterministic=cfg.reproducibility.use_deterministic_algorithms,
    )

    LOGGER.info(
        "Reproducibility header: %s",
        reproducibility_header(state.seed, state.deterministic),
    )
    LOGGER.info("Reproducibility details:\n%s", summarise_reproducibility_state(state))

    if args.save_config_to:
        out_path = Path(args.save_config_to)
        LOGGER.info("Saving resolved config to %s", out_path)
        save_resolved_config(cfg, out_path)
        LOGGER.info("Config hash: %s", get_config_hash(cfg))

    # Kick off training (TriObjectiveTrainer is responsible for everything downstream)
    LOGGER.info("Initialising TriObjectiveTrainer...")
    trainer = TriObjectiveTrainer(cfg, debug=args.debug)  # adjust signature if needed

    LOGGER.info("Starting tri-objective training loop...")
    trainer.run()  # or trainer.fit(), depending on your trainer implementation
    LOGGER.info("Training completed.")


if __name__ == "__main__":  # pragma: no cover (script entry point)
    main()
