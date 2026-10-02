"""
Centralized logging setup.

Uses the standard library only — no external dependency. Logs are structured
to stdout so a container runtime (Docker / k8s) can collect them.
"""

from __future__ import annotations

import logging
import sys
from logging.config import dictConfig

from app.core.config import settings

_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging() -> None:
    """Apply logging configuration. Call once at app startup."""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": _LOG_FORMAT,
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                },
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "default",
                    "level": settings.LOG_LEVEL,
                },
            },
            "loggers": {
                # App loggers
                "app": {
                    "handlers": ["stdout"],
                    "level": settings.LOG_LEVEL,
                    "propagate": False,
                },
                # Tame uvicorn double-logging
                "uvicorn": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
            },
            "root": {
                "handlers": ["stdout"],
                "level": settings.LOG_LEVEL,
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `app` tree."""
    if not name.startswith("app"):
        name = f"app.{name}"
    return logging.getLogger(name)
