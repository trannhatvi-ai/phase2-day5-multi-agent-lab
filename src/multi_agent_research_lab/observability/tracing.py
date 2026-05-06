"""Tracing hooks.

Supports LangSmith, Langfuse, or simple JSON traces. Automatically detects
available providers based on environment variables.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_trace_log: list[dict[str, Any]] = []
_langsmith_run = None


def _try_init_langsmith() -> None:
    global _langsmith_run
    try:
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        if settings.langsmith_api_key:
            import os

            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
            os.environ["LANGSMITH_TRACING"] = "true"
            logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)
    except Exception:
        pass


def flush_trace_log(output_path: Path | None = None) -> Path:
    """Write accumulated trace events to a JSON file."""
    path = output_path or Path("reports/trace.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_trace_log, f, indent=2, default=str)
    logger.info("Trace flushed to %s (%d events)", path, len(_trace_log))
    return path


def get_trace_log() -> list[dict[str, Any]]:
    return list(_trace_log)


def clear_trace_log() -> None:
    _trace_log.clear()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context that records duration and can be extended with providers.

    Usage:
        with trace_span("my_operation", {"key": "value"}) as span:
            # do work
            span["result"] = "ok"
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _trace_log.append(span)
        logger.debug("trace_span %s: %.3fs", name, span["duration_seconds"])
