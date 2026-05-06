"""Writer agent that produces the final answer."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        state.add_trace_event("writer_start", {})

        research_notes = state.research_notes or "No research notes."
        analysis_notes = state.analysis_notes or "No analysis notes."

        source_refs = ""
        if state.sources:
            source_refs = "\n\nAvailable sources for citation:\n"
            for i, src in enumerate(state.sources, 1):
                url_part = f" ({src.url})" if src.url else ""
                source_refs += f"[{i}] {src.title}{url_part}\n"

        system_prompt = (
            f"You are an expert writer for {state.request.audience}. "
            "Given research notes and analysis, write a clear, well-structured final answer.\n\n"
            "Requirements:\n"
            "- Address the original query directly\n"
            "- Include key findings from the analysis\n"
            "- Cite sources using [1], [2] notation where appropriate\n"
            "- Use clear headings and paragraphs\n"
            "- Be concise but thorough (aim for 300-500 words)\n"
            "- End with a brief summary or conclusion"
        )
        user_prompt = (
            f"Original query: {state.request.query}\n\n"
            f"Research notes:\n{research_notes}\n\n"
            f"Analysis:\n{analysis_notes}"
            f"{source_refs}"
        )

        response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = response.content

        state.add_trace_event(
            "writer_done",
            {
                "answer_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Writer: produced %d chars final answer", len(response.content))
        return state
