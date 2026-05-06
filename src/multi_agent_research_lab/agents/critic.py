"""Optional critic agent for fact-checking and safety review."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to state."""
        state.add_trace_event("critic_start", {})

        if not state.final_answer:
            state.errors.append("Critic: no final answer to review")
            state.add_trace_event("critic_skip", {"reason": "no final_answer"})
            return state

        source_text = ""
        if state.sources:
            for i, src in enumerate(state.sources, 1):
                url_part = f" ({src.url})" if src.url else ""
                source_text += f"[{i}] {src.title}{url_part}: {src.snippet}\n"

        system_prompt = (
            "You are a critical reviewer. Evaluate the final answer for:\n"
            "1. Factual accuracy — are claims supported by the provided sources?\n"
            "2. Citation coverage — are key claims properly cited?\n"
            "3. Hallucination risk — does the answer contain unsupported claims?\n"
            "4. Completeness — does the answer fully address the query?\n\n"
            "Provide your review in this format:\n"
            "VERDICT: PASS or FAIL\n"
            "ISSUES: (list any issues, or 'None')\n"
            "SUGGESTIONS: (list improvements)\n"
            "CONFIDENCE: 0-10"
        )
        user_prompt = (
            f"Original query: {state.request.query}\n\n"
            f"Final answer:\n{state.final_answer}\n\n"
            f"Sources:\n{source_text or 'No sources provided.'}"
        )

        try:
            response = self._llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            review = response.content

            state.add_trace_event(
                "critic_done",
                {
                    "review_length": len(review),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )

            if "VERDICT: FAIL" in review:
                state.errors.append(f"Critic flagged issues: {review}")

            logger.info("Critic: review complete (%d chars)", len(review))

        except Exception as exc:
            logger.warning("Critic failed: %s", exc)
            state.errors.append(f"Critic failed: {exc}")

        return state
