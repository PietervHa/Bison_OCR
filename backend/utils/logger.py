"""
Logging Configuration

Sets up the application's logging system, providing console and file output
with HMI polling requests demoted to DEBUG. vision.log rotates at 5 MB x 5.
"""

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Endpoints the HMI polls every few hundred ms. Their access-log lines are
# demoted to DEBUG so they stay out of the console and vision.log.
_POLL_PATHS = frozenset({"/result", "/status", "/camera_status"})
_REQUEST_LINE = re.compile(r'"[A-Z]+ (/[^\s?"]*)')


class _DemotePollRequestsFilter(logging.Filter):
    """Lower uvicorn access-log records for HMI polling endpoints to DEBUG."""

    def filter(self, record):
        # uvicorn.access args: (client_addr, method, full_path, http_version, status)
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) >= 3 and isinstance(args[2], str):
            path = args[2].split("?", 1)[0]
        else:
            match = _REQUEST_LINE.search(record.getMessage())
            path = match.group(1) if match else None
        if path in _POLL_PATHS and record.levelno > logging.DEBUG:
            record.levelno = logging.DEBUG
            record.levelname = logging.getLevelName(logging.DEBUG)
        return True


_POLL_FILTER = _DemotePollRequestsFilter()

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5


# UVICORN_LOG_CONFIG

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {},
    "handlers": {},
    "loggers": {
        "uvicorn": {"level": "INFO", "propagate": True},
        "uvicorn.error": {"level": "INFO", "propagate": True},
        "uvicorn.access": {"level": "INFO", "propagate": True},
    },
}


def setup_logging():
    """Configure application logging for console and file output."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers when setup is called more than once.
    if getattr(root_logger, "_vision_logging_configured", False):
        return

    logs_dir = Path(__file__).resolve().parents[2] / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        logs_dir / "vision.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # A logger-level filter runs before the record reaches any handler, so the
    # demoted level is what the INFO handlers see. addFilter skips duplicates,
    # and uvicorn's dictConfig (UVICORN_LOG_CONFIG) does not clear filters.
    logging.getLogger("uvicorn.access").addFilter(_POLL_FILTER)

    root_logger._vision_logging_configured = True


def get_logger(name):
    """
    Returns a logger instance with the specified name.
    """
    return logging.getLogger(name)