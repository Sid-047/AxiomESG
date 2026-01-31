from __future__ import annotations

import logging
from typing import Any, Dict


class ContextFilter(logging.Filter):
    def __init__(self, context: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str, **context: Any) -> logging.Logger:
    logger = logging.getLogger(name)
    if context:
        logger.addFilter(ContextFilter(context))
    return logger
