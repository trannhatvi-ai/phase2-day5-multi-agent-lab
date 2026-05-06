"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _baseline_runner(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call handles everything."""
    llm = LLMClient()
    state = ResearchState(request=ResearchQuery(query=query))
    system_prompt = (
        "You are a research assistant. Given the user's query, research the topic, "
        "analyze key findings, and write a clear, well-structured answer with citations "
        "where possible. Provide your response in under 500 words."
    )
    response = llm.complete(system_prompt=system_prompt, user_prompt=query)
    state.final_answer = response.content
    state.add_trace_event(
        "baseline_call",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline with real LLM call and benchmark."""
    _init()
    state, metrics = run_benchmark("single-agent-baseline", query, _baseline_runner)
    console.print(Panel.fit(state.final_answer or "(empty)", title="Single-Agent Baseline"))
    report = render_markdown_report([metrics])
    console.print(report)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()

    def _multi_runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        return workflow.run(s)

    result_state, metrics = run_benchmark("multi-agent", query, _multi_runner)
    console.print(result_state.model_dump_json(indent=2))
    report = render_markdown_report([metrics])
    console.print(report)


if __name__ == "__main__":
    app()
