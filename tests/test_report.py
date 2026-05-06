from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "1.23" in report


def test_report_with_cost_and_quality() -> None:
    metrics = BenchmarkMetrics(
        run_name="test",
        latency_seconds=2.5,
        estimated_cost_usd=0.001,
        quality_score=7.5,
        notes="good",
    )
    report = render_markdown_report([metrics])
    assert "$0.0010" in report
    assert "7.5/10" in report


def test_report_comparison_section() -> None:
    single = BenchmarkMetrics(run_name="single-agent-baseline", latency_seconds=5.0, notes="ok")
    multi = BenchmarkMetrics(run_name="multi-agent", latency_seconds=2.5, notes="ok")
    report = render_markdown_report([single, multi])
    assert "Comparison" in report
    assert "2.0x" in report


def test_report_empty_list() -> None:
    report = render_markdown_report([])
    assert "Benchmark Report" in report
