"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


COST_PER_1K_INPUT: dict[str, float] = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0005,
}

COST_PER_1K_OUTPUT: dict[str, float] = {
    "gpt-4o-mini": 0.0006,
    "gpt-4o": 0.015,
    "gpt-4-turbo": 0.03,
    "gpt-3.5-turbo": 0.0015,
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate = COST_PER_1K_INPUT.get(model, 0.001)
    output_rate = COST_PER_1K_OUTPUT.get(model, 0.002)
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


class LLMClient:
    """Provider-agnostic LLM client with retry and token tracking."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        resolved_key = api_key or settings.openai_api_key
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env or pass api_key.")
        self._client = OpenAI(api_key=resolved_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, timeout, and token logging."""
        settings = get_settings()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=settings.timeout_seconds,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = _estimate_cost(self.model, input_tokens, output_tokens)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost

        logger.debug(
            "LLM call: model=%s input_tokens=%s output_tokens=%s cost=%.6f",
            self.model,
            input_tokens,
            output_tokens,
            cost or 0.0,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
