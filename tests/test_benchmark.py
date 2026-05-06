from __future__ import annotations

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def _mock_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.final_answer = f"Answer to: {query}. Based on [1] and [2] sources."
    state.add_trace_event("mock", {"cost_usd": 0.001})
    return state


def test_benchmark_measures_latency() -> None:
    state, metrics = run_benchmark("test", "test query", _mock_runner)
    assert metrics.latency_seconds >= 0
    assert metrics.run_name == "test"
    assert state.final_answer is not None


def test_benchmark_estimates_cost() -> None:
    _, metrics = run_benchmark("test", "test query", _mock_runner)
    assert metrics.estimated_cost_usd is not None
    assert metrics.estimated_cost_usd > 0


def test_benchmark_detects_citations() -> None:
    _, metrics = run_benchmark("test", "test query", _mock_runner)
    assert "citations:" in metrics.notes
