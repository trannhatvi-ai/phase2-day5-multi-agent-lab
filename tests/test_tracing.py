from __future__ import annotations

from pathlib import Path

from multi_agent_research_lab.observability.tracing import (
    clear_trace_log,
    flush_trace_log,
    get_trace_log,
    trace_span,
)


def test_trace_span_records_duration() -> None:
    clear_trace_log()
    with trace_span("test_op", {"key": "value"}) as span:
        span["result"] = "ok"
    assert span["duration_seconds"] is not None
    assert span["duration_seconds"] >= 0
    assert span["result"] == "ok"


def test_trace_span_captures_exceptions() -> None:
    clear_trace_log()
    try:
        with trace_span("failing_op"):
            raise ValueError("boom")
    except ValueError:
        pass
    log = get_trace_log()
    assert len(log) == 1
    assert "error" in log[0]


def test_flush_trace_log(tmp_path: Path) -> None:
    clear_trace_log()
    with trace_span("flush_test"):
        pass
    output = tmp_path / "trace.json"
    result = flush_trace_log(output)
    assert result == output
    assert output.exists()


def test_clear_trace_log() -> None:
    with trace_span("clear_test"):
        pass
    assert len(get_trace_log()) > 0
    clear_trace_log()
    assert len(get_trace_log()) == 0
