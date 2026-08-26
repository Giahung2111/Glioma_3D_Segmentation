"""Consistent timestamped console and file logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    *, level: int | str = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """Configure the project logger idempotently for a single process."""

    logger = logging.getLogger("glioma_seg")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
