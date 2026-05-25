"""
Centralized logging configuration for the Sports Predictor system.

Provides a single setup_logger() factory that returns consistently
configured loggers with both console and file handlers.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import settings


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    log_file : str, optional
        Path to log file. Defaults to settings value.
    level : str, optional
        Logging level string. Defaults to settings value or env var.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_cfg = settings.get("logging", {})

    # Resolve level: param > env var > settings.yaml > INFO
    resolved_level = (
        level
        or os.environ.get("LOG_LEVEL")
        or log_cfg.get("level", "INFO")
    ).upper()

    resolved_file = log_file or log_cfg.get("file", "logs/sports_predictor.log")
    log_format = log_cfg.get(
        "format",
        "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    )

    # Parse rotation size (e.g. "10 MB" -> 10_485_760 bytes)
    rotation_str = log_cfg.get("rotation", "10 MB")
    rotation_bytes = _parse_rotation_size(rotation_str)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    formatter = logging.Formatter(log_format)

    # --- Console handler (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, resolved_level, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler with rotation ---
    try:
        log_path = Path(resolved_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=rotation_bytes,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, resolved_level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        logger.warning("Could not create file handler: %s. Logging to console only.", exc)

    return logger


def _parse_rotation_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Examples: '10 MB' -> 10_485_760, '500 KB' -> 512_000
    """
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }

    parts = size_str.strip().split()
    if len(parts) != 2:
        return 10 * 1024 * 1024  # default 10 MB

    try:
        value = float(parts[0])
        unit = parts[1].upper()
        return int(value * multipliers.get(unit, 1024 ** 2))
    except (ValueError, KeyError):
        return 10 * 1024 * 1024
