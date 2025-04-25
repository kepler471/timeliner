"""Centralised coloured logging."""
import logging
import sys
from typing import Final

_FMT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _configure_root(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format=_FMT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str, level: str | None = None) -> logging.Logger:  # noqa: D401
    if not logging.getLogger().handlers:
        _configure_root()
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level)
    return logger