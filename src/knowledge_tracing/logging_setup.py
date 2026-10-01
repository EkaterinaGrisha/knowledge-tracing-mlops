"""Logging configuration for the command-line entry points."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
PACKAGE_LOGGER = "knowledge_tracing"


def configure_logging(level: str | int = logging.INFO) -> None:
    """Send the package's log records to stderr.

    Library modules only create loggers (``logging.getLogger(__name__)``);
    handlers and the level are set once here, by the entry point. Third-party
    loggers keep their defaults, so their INFO chatter stays hidden.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
