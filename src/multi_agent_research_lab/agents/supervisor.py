"""Supervisor / router agent that decides which worker runs next."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

VALID_ROUTES = {"researcher", "analyst", "writer", "done"}


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        - If no research_notes -> route to researcher
        - If research_notes but no analysis_notes -> route to analyst
        - If analysis_notes but no final_answer -> route to writer
        - If final_answer is set -> done
        - Enforce max_iterations to prevent infinite loops
        """
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            logger.warning("Max iterations (%d) reached, forcing done", settings.max_iterations)
            state.record_route("done")
            state.add_trace_event("supervisor", {"decision": "done", "reason": "max_iterations"})
            return state

        decision = self._llm_route(state) if self._llm else self._deterministic_route(state)

        state.record_route(decision)
        state.add_trace_event("supervisor", {"decision": decision, "iteration": state.iteration})
        logger.info("Supervisor routed to: %s (iteration %d)", decision, state.iteration)
        return state

    def _deterministic_route(self, state: ResearchState) -> str:
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"

    def _llm_route(self, state: ResearchState) -> str:
        context = self._build_context(state)
        system_prompt = (
            "You are a supervisor routing agent. Based on the current workflow state, "
            "decide which agent should run next. "
            "Valid routes: researcher, analyst, writer, done.\n\n"
            "Routing rules:\n"
            "- If research_notes is missing or empty -> researcher\n"
            "- If research_notes exists but analysis_notes is missing -> analyst\n"
            "- If analysis_notes exists but final_answer is missing -> writer\n"
            "- If all outputs are present -> done\n\n"
            "Respond with ONLY the route name, nothing else."
        )
        try:
            assert self._llm is not None
            response = self._llm.complete(system_prompt=system_prompt, user_prompt=context)
            route = response.content.strip().lower()
            if route not in VALID_ROUTES:
                return self._deterministic_route(state)
            return route
        except Exception:
            logger.warning("LLM routing failed, falling back to deterministic")
            return self._deterministic_route(state)

    @staticmethod
    def _build_context(state: ResearchState) -> str:
        parts = [f"Query: {state.request.query}"]
        parts.append(f"Iteration: {state.iteration}")
        parts.append(f"Has research_notes: {bool(state.research_notes)}")
        parts.append(f"Has analysis_notes: {bool(state.analysis_notes)}")
        parts.append(f"Has final_answer: {bool(state.final_answer)}")
        parts.append(f"Route history: {state.route_history}")
        return "\n".join(parts)
