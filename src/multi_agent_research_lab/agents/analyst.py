"""Analyst agent that turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        state.add_trace_event("analyst_start", {"has_notes": bool(state.research_notes)})

        research_notes = state.research_notes or "No research notes available."
        system_prompt = (
            "You are an expert analyst. Given research notes from a researcher, produce "
            "structured analysis notes. Include:\n"
            "1. Key claims (numbered list)\n"
            "2. Points of agreement and disagreement across sources\n"
            "3. Strength of evidence for each claim (strong/moderate/weak)\n"
            "4. Gaps or areas needing more research\n"
            "5. Overall assessment (2-3 sentences)\n\n"
            "Be concise and precise. Use bullet points. Keep under 400 words."
        )
        user_prompt = f"Original query: {state.request.query}\n\nResearch notes:\n{research_notes}"

        response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content

        state.add_trace_event(
            "analyst_done",
            {
                "analysis_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Analyst: produced %d chars analysis", len(response.content))
        return state
