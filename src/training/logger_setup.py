"""Configure Python's logging module for training traces.

Python's logging system is itself an example of the Decorator/Chain-of-
Responsibility pattern: handlers and formatters are attached to loggers
at runtime without modifying the code that emits the log messages.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "trainer",
    level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    """Create and configure a logger with console and optional file output.

    Args:
        name: Logger name (appears in every log line).
        level: Minimum log level (DEBUG shows everything, INFO hides batch traces).
        log_file: If provided, also writes logs to this file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
