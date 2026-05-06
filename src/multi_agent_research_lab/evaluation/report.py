"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a rich markdown report."""
    lines = [
        "# Benchmark Report",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "-" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "-" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        row = (
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )
        lines.append(row)

    if len(metrics) >= 2:
        lines.extend(["", "## Comparison", ""])
        single = next(
            (
                m
                for m in metrics
                if "baseline" in m.run_name.lower() or "single" in m.run_name.lower()
            ),
            None,
        )
        multi = next(
            (m for m in metrics if "multi" in m.run_name.lower()),
            None,
        )
        if single and multi:
            speedup = (
                single.latency_seconds / multi.latency_seconds if multi.latency_seconds > 0 else 0
            )
            faster = "faster" if speedup > 1 else "slower"
            lines.append(
                f"- **Latency ratio**: multi-agent is {speedup:.1f}x {faster} than baseline"
            )
            if single.estimated_cost_usd and multi.estimated_cost_usd:
                cost_ratio = multi.estimated_cost_usd / single.estimated_cost_usd
                lines.append(f"- **Cost ratio**: multi-agent costs {cost_ratio:.1f}x vs baseline")
            lines.append(f"- **Baseline notes**: {single.notes}")
            lines.append(f"- **Multi-agent notes**: {multi.notes}")

    lines.append("")
    return "\n".join(lines)
