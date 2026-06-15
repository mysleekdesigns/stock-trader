"""Structlog configuration for the Stock Trader application.

Produces JSON logs in production and coloured console output in development.
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """Configure structlog and stdlib logging for the application.

    Args:
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: When set to ``"production"``, logs are rendered as JSON;
            otherwise coloured console output is used.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors applied to every log event.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if environment == "production":
        # JSON renderer for structured log ingestion.
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # Human-friendly coloured output for local development.
        renderer = structlog.dev.ConsoleRenderer()

    # Processors that run only within structlog's own pipeline.
    structlog_processors = [
        *shared_processors,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=structlog_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Format stdlib log records through structlog so that third-party
    # libraries (uvicorn, sqlalchemy, etc.) use the same output style.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "httpcore", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
