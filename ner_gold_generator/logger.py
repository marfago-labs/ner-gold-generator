"""Shared stderr logger setup for CLI debug output."""

from __future__ import annotations

import logging
import os
import sys


class _DynamicStderrHandler(logging.Handler):
    """Logging handler that always writes to the current ``sys.stderr``."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with file logging enabled by default.

    Defaults:
    - Write structured debug logs to ``gold.log`` in the current working dir.
    - Do not emit logger debug lines to stderr unless explicitly enabled.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # Defaults: structured logs to file, no debug logger noise on stderr.
    log_file = os.environ.get("NER_GOLD_LOG_FILE", "gold.log").strip()
    log_to_stderr = os.environ.get("NER_GOLD_LOG_STDERR", "0").strip() not in {
        "0",
        "false",
        "False",
        "no",
        "NO",
    }

    # CLI stderr: keep messages unadorned so progress / prompts look the same.
    if log_to_stderr:
        fmt_stderr = logging.Formatter("%(message)s")
        stderr_handler = _DynamicStderrHandler()
        stderr_handler.setFormatter(fmt_stderr)
        logger.addHandler(stderr_handler)

    if log_file:
        # File logs: structured format with timestamp, level, and logger name.
        fmt_file = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt_file)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
