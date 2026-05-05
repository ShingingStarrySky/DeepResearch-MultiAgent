from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings

_loggers: dict[str, logging.Logger] = {}
_logging_initialized: bool = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    global _logging_initialized
    if _logging_initialized:
        return

    settings = get_settings()

    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

    _logging_initialized = True


def get_logger(name: str = "DeepResearch") -> logging.Logger:
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
        if not _logging_initialized:
            settings = get_settings()
            setup_logging(settings.log_level)
    return _loggers[name]
