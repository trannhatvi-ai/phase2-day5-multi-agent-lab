"""Benchmark runner for single-agent vs multi-agent comparison."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _count_citations(state: ResearchState) -> tuple[int, int]:
    """Count claims with citations vs total claims. Simple heuristic."""
    answer = state.final_answer or ""
    import re

    citation_pattern = re.compile(r"\[\d+\]")
    citations_found = len(citation_pattern.findall(answer))
    sentence_pattern = re.compile(r"[.!?]+")
    total_sentences = max(len(sentence_pattern.findall(answer)), 1)
    return citations_found, total_sentences


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate cost, and compute citation coverage."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    citations, total = _count_citations(state)
    round(citations / total, 2) if total > 0 else 0.0

    estimated_cost = 0.0
    for event in state.trace:
        payload = event.get("payload", {})
        cost = payload.get("cost_usd")
        if cost is not None:
            estimated_cost += cost

    has_error = bool(state.errors) or not state.final_answer

    notes_parts = []
    if state.errors:
        notes_parts.append(f"errors: {'; '.join(state.errors)}")
    if not state.final_answer:
        notes_parts.append("no final answer produced")
    notes_parts.append(f"citations: {citations}/{total}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(estimated_cost, 6) if estimated_cost > 0 else None,
        notes="; ".join(notes_parts),
    )

    logger.info(
        "Benchmark '%s': %.2fs, cost=$%.6f, citations=%d/%d, errors=%s",
        run_name,
        latency,
        estimated_cost,
        citations,
        total,
        has_error,
    )

    return state, metrics
