from __future__ import annotations

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


class TestSupervisorDeterministic:
    """Test deterministic routing without LLM."""

    def test_routes_to_researcher_first(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert result.route_history[-1] == "researcher"
        assert result.iteration == 1

    def test_routes_to_analyst_when_notes_exist(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.research_notes = "Some notes"
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert result.route_history[-1] == "analyst"

    def test_routes_to_writer_when_analysis_exists(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.research_notes = "Notes"
        state.analysis_notes = "Analysis"
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert result.route_history[-1] == "writer"

    def test_routes_to_done_when_answer_exists(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.research_notes = "Notes"
        state.analysis_notes = "Analysis"
        state.final_answer = "Answer"
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"

    def test_max_iterations_forces_done(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.iteration = 100
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"

    def test_trace_events_recorded(self) -> None:
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        supervisor = SupervisorAgent()
        result = supervisor.run(state)
        assert any(e["name"] == "supervisor" for e in result.trace)
