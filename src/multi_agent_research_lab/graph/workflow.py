"""LangGraph-based multi-agent workflow."""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Uses LangGraph if available, otherwise falls back to a simple loop.
    Keep orchestration here; keep agent internals in agents/.
    """

    def __init__(self) -> None:
        from multi_agent_research_lab.agents.analyst import AnalystAgent
        from multi_agent_research_lab.agents.researcher import ResearcherAgent
        from multi_agent_research_lab.agents.supervisor import SupervisorAgent
        from multi_agent_research_lab.agents.writer import WriterAgent

        self._llm = LLMClient()
        self._agents: dict[str, BaseAgent] = {
            "supervisor": SupervisorAgent(llm=self._llm),
            "researcher": ResearcherAgent(llm=self._llm),
            "analyst": AnalystAgent(llm=self._llm),
            "writer": WriterAgent(llm=self._llm),
        }

    def build(self) -> Any:
        """Create a LangGraph graph. Returns the compiled graph or None."""
        try:
            return self._build_langgraph()
        except ImportError:
            logger.info("LangGraph not available, using simple loop executor")
            return None

    def _build_langgraph(self) -> Any:
        from langgraph.graph import END, StateGraph

        def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
            rs = ResearchState(**state)
            rs = self._agents["supervisor"].run(rs)
            return rs.model_dump()

        def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
            rs = ResearchState(**state)
            rs = self._agents["researcher"].run(rs)
            return rs.model_dump()

        def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            rs = ResearchState(**state)
            rs = self._agents["analyst"].run(rs)
            return rs.model_dump()

        def writer_node(state: dict[str, Any]) -> dict[str, Any]:
            rs = ResearchState(**state)
            rs = self._agents["writer"].run(rs)
            return rs.model_dump()

        def route_after_supervisor(state: dict[str, Any]) -> str:
            rs = ResearchState(**state)
            if not rs.route_history:
                return END
            last_route = rs.route_history[-1]
            if last_route == "done":
                return END
            if last_route in ("researcher", "analyst", "writer"):
                return last_route
            return END

        graph: Any = StateGraph(dict[str, Any])
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", route_after_supervisor)
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state."""
        settings = get_settings()

        compiled = self.build()
        if compiled is not None:
            return self._run_langgraph(compiled, state)

        return self._run_loop(state, settings.max_iterations)

    def _run_langgraph(self, compiled: Any, state: ResearchState) -> ResearchState:
        try:
            result = compiled.invoke(state.model_dump())
            return ResearchState(**result)
        except Exception as exc:
            logger.error("LangGraph execution failed: %s", exc)
            raise AgentExecutionError(f"Workflow failed: {exc}") from exc

    def _run_loop(self, state: ResearchState, max_iterations: int) -> ResearchState:
        for _i in range(max_iterations):
            state = self._agents["supervisor"].run(state)
            last_route = state.route_history[-1] if state.route_history else "done"

            if last_route == "done":
                break

            agent = self._agents.get(last_route)
            if agent is None:
                logger.error("Unknown route: %s", last_route)
                state.errors.append(f"Unknown route: {last_route}")
                break

            try:
                state = agent.run(state)
            except Exception as exc:
                logger.error("Agent %s failed: %s", last_route, exc)
                state.errors.append(f"Agent {last_route} failed: {exc}")
                break

        return state
