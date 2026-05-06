from __future__ import annotations

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState


def test_state_records_route_and_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("route", {"next": "researcher"})
    assert state.iteration == 1
    assert state.route_history == ["researcher"]
    assert state.trace[0]["name"] == "route"


def test_state_multiple_routes() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.record_route("analyst")
    state.record_route("writer")
    state.record_route("done")
    assert state.iteration == 4
    assert state.route_history == ["researcher", "analyst", "writer", "done"]


def test_state_sources_and_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources.append(SourceDocument(title="Test", snippet="A snippet"))
    state.research_notes = "Some notes"
    state.analysis_notes = "Analysis"
    state.final_answer = "Answer"
    assert len(state.sources) == 1
    assert state.research_notes == "Some notes"


def test_state_agent_results() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.agent_results.append(AgentResult(agent=AgentName.RESEARCHER, content="done"))
    assert len(state.agent_results) == 1
    assert state.agent_results[0].agent == AgentName.RESEARCHER
