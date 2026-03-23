"""daga.telemetry.logging

Central logging helpers for DAGA.

Goals:
- Make it easy to instrument granular pipeline stages without scattering
  logging configuration throughout the code.
- Keep logs off by default (library-friendly).
- Provide structured-ish fields via LoggerAdapter (`extra`).

Environment variables:
- DAGA_LOG_LEVEL:   default INFO
- DAGA_LOG_FORMAT:  'text' (default) | 'json'

Note: We intentionally keep JSON logging minimal and dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


_LOGGER_NAME = "daga"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include common structured fields if present
        for k in (
            "task_id",
            "plan_id",
            "trace_id",
            "stage",
            "role_id",
            "role_name",
            "model_id",
            "topology",
            "routing_source",
            "rule",
            "tool",
            "cmd",
            "path",
        ):
            if hasattr(record, k):
                base[k] = getattr(record, k)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def configure_logging(
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    stream=None,
) -> None:
    """Configure the `daga` root logger once.

    Safe to call multiple times.
    """
    level = (level or os.environ.get("DAGA_LOG_LEVEL") or "INFO").upper()
    fmt = (fmt or os.environ.get("DAGA_LOG_FORMAT") or "text").lower()

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(handler)


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Get a logger under the `daga` namespace."""
    return logging.getLogger(name)


def _safe_extra(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def log_kv(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Convenience helper to attach structured fields using `extra`."""
    extra = {k: _safe_extra(v) for k, v in fields.items() if v is not None}
    logger.log(level, msg, extra=extra)
