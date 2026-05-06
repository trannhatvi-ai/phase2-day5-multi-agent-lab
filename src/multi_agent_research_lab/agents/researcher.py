"""Researcher agent that collects sources and creates research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        state.add_trace_event("researcher_start", {"query": query})

        sources = self._search.search(query, max_results=max_sources)
        state.sources = sources

        source_text = self._format_sources(sources)
        system_prompt = (
            "You are a research assistant. Given the user's query and a set of sources, "
            "create concise research notes. For each key finding, cite the source by title. "
            "Focus on facts, data points, and key arguments. Keep notes under 400 words."
        )
        user_prompt = f"Query: {query}\n\nSources:\n{source_text}"

        response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = response.content

        state.add_trace_event(
            "researcher_done",
            {
                "sources_count": len(sources),
                "notes_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Researcher: %d sources, %d chars notes", len(sources), len(response.content))
        return state

    @staticmethod
    def _format_sources(sources: list[SourceDocument]) -> str:
        parts = []
        for i, src in enumerate(sources, 1):
            url_part = f" ({src.url})" if src.url else ""
            parts.append(f"[{i}] {src.title}{url_part}: {src.snippet}")
        return "\n".join(parts) if parts else "No sources found."
