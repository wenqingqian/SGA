"""SGA package logger.

Mirrors the API surface the launcher code expects from the original
``flagscale.logger`` (a ``logging.Logger``-compatible object with
info/warning/error/debug), so the deep-copied launcher files keep working
after the namespace surgery.
"""

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _build_logger(name: str = "sga") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("SGA_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


logger = _build_logger()
