"""Command-line interface: ``kt <command>`` (also ``python -m knowledge_tracing``).

Commands:
    kt etl     extract, transform and load the data
    kt train   run the whole pipeline: ETL, quality gate, training, evaluation,
               drift monitoring, MLflow logging and reports

Heavy libraries are imported inside the commands, after the platform runtime
settings are applied, so ``kt --help`` is instant.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from pydantic import ValidationError

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config, quick_overlay_path
from .errors import DataQualityError
from .logging_setup import configure_logging
from .runtime import configure_runtime

LOG = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INVALID_CONFIG = 2
EXIT_DATA_QUALITY = 3


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="pipeline configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--data-source",
        choices=["sample", "full"],
        default="sample",
        help="committed sample (offline) or the full dataset (default: %(default)s)",
    )


def _etl(args: argparse.Namespace) -> int:
    from .etl.run import run_etl

    run_etl(load_config(args.config), args.data_source, sample_students=args.write_sample)
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    overlays = [quick_overlay_path(args.config)] if args.quick else []
    overlays += args.override
    run_pipeline(load_config(args.config, overlays), args.data_source, quick=args.quick)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``kt`` command."""
    parser = argparse.ArgumentParser(
        prog="kt",
        description="Knowledge tracing on ASSISTments 2009: ETL, training and evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosity of the package logs (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    etl = commands.add_parser("etl", help="extract, transform and load the data")
    _add_data_args(etl)
    etl.add_argument(
        "--write-sample",
        type=int,
        metavar="N_STUDENTS",
        help="also refresh the committed sample CSV with the first N students "
        "(use with --data-source full)",
    )
    etl.set_defaults(handler=_etl)

    train = commands.add_parser("train", help="run the whole training pipeline")
    _add_data_args(train)
    train.add_argument(
        "--quick",
        action="store_true",
        help="minimal training budgets from quick.yaml next to the config (smoke runs)",
    )
    train.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="YAML",
        help="extra configuration overlay merged on top (repeatable)",
    )
    train.set_defaults(handler=_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``kt`` command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_runtime()
    configure_logging(args.log_level)
    try:
        exit_code: int = args.handler(args)
    except ValidationError as exc:
        LOG.error("Invalid configuration: %s", exc)
        return EXIT_INVALID_CONFIG
    except DataQualityError as exc:
        LOG.error("Run stopped by the data-quality gate: %s", exc)
        return EXIT_DATA_QUALITY
    return exit_code
