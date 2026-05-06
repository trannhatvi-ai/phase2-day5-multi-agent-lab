"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from openai import OpenAI

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Search client using OpenAI web search tool or Tavily fallback."""

    _openai_client: OpenAI | None

    def __init__(self) -> None:
        settings = get_settings()
        self._tavily_key = settings.tavily_api_key
        if settings.openai_api_key:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._openai_client = None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily_key:
            return self._search_tavily(query, max_results)
        return self._search_llm_fallback(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            from tavily import TavilyClient  # type: ignore[import-not-found]

            client = TavilyClient(api_key=self._tavily_key)
            response = client.search(query=query, max_results=max_results)
            results = []
            for item in response.get("results", []):
                results.append(
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("url"),
                        snippet=item.get("content", "")[:500],
                    )
                )
            return results
        except Exception:
            logger.warning("Tavily search failed, falling back to LLM fallback")
            return self._search_llm_fallback(query, max_results)

    def _search_llm_fallback(self, query: str, max_results: int) -> list[SourceDocument]:
        if not self._openai_client:
            logger.warning("No search provider available, returning empty results")
            return []

        response = self._openai_client.chat.completions.create(
            model=get_settings().openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Given the user's query, provide up to "
                        f"{max_results} relevant sources. For each source, return a JSON array "
                        "with objects containing 'title', 'url' (can be null), and 'snippet' "
                        "(a concise 1-2 sentence summary). Return ONLY valid JSON, no markdown."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        import json

        content = response.choices[0].message.content or "[]"
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed = parsed.get("sources", parsed.get("results", []))
            if not isinstance(parsed, list):
                parsed = []
        except json.JSONDecodeError:
            parsed = []

        results: list[SourceDocument] = []
        for item in parsed[:max_results]:
            if isinstance(item, dict):
                results.append(
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("url"),
                        snippet=item.get("snippet", "")[:500],
                    )
                )
        return results
