# AGENTS.md — Project Guide for AI Coding Agents

## Project Overview

**multi-agent-research-lab** — A production-grade starter skeleton for building a multi-agent research system. Students implement Supervisor + Researcher + Analyst + Writer agents, then benchmark against a single-agent baseline.

- **Language**: Python 3.11+
- **Package layout**: `src/` layout with `multi_agent_research_lab` as the importable package
- **CLI entry**: `malab` (or `python -m multi_agent_research_lab.cli`)

## Architecture

```text
User Query
   |
   v
Supervisor / Router (routing policy)
   |------> Researcher Agent  -> state.research_notes + state.sources
   |------> Analyst Agent     -> state.analysis_notes
   |------> Writer Agent      -> state.final_answer
   |------> Critic Agent      -> (optional, bonus)
   |
   v
Trace + Benchmark Report
```

All agents share a single `ResearchState` (Pydantic model). Each agent reads from and writes to this shared state. The Supervisor decides routing via `state.route_history`.

## Directory Structure

```text
src/multi_agent_research_lab/
  __init__.py
  cli.py                  # Typer CLI entrypoint (baseline + multi-agent commands)
  agents/
    __init__.py            # Re-exports all agent classes
    base.py                # BaseAgent ABC — run(state) -> state
    supervisor.py          # Routing policy (TODO)
    researcher.py          # Search + research notes (TODO)
    analyst.py             # Structured insights from notes (TODO)
    writer.py              # Final answer synthesis (TODO)
    critic.py              # Optional fact-check / safety review (TODO)
  core/
    config.py              # Pydantic Settings from .env
    schemas.py             # ResearchQuery, AgentResult, SourceDocument, BenchmarkMetrics, AgentName
    state.py               # ResearchState — shared state model
    errors.py              # LabError, StudentTodoError, AgentExecutionError, ValidationError
  graph/
    workflow.py            # LangGraph workflow skeleton (TODO)
  services/
    llm_client.py          # LLMClient abstraction (TODO)
    search_client.py       # SearchClient abstraction (TODO)
    storage.py             # LocalArtifactStore for reports
  evaluation/
    benchmark.py           # run_benchmark() — latency + metrics
    report.py              # render_markdown_report()
  observability/
    logging.py             # configure_logging()
    tracing.py             # trace_span() context manager
  utils/
    timer.py               # elapsed_timer() context manager
configs/
  lab_default.yaml         # Agent model configs + benchmark queries
docs/
  lab_guide.md             # Lab instructions, milestones, exit ticket
  design_template.md       # Design doc template for students
  peer_review_rubric.md    # Peer review scoring rubric
tests/
  test_config.py           # Settings defaults test
  test_state.py            # ResearchState route/trace test
  test_report.py           # Markdown report rendering test
  test_agents_todo.py      # Verifies SupervisorAgent raises StudentTodoError
scripts/
  check_todos.sh           # grep for all TODO(student) markers
reports/
  .gitkeep                 # Output dir for benchmark reports (git-ignored)
```

## Key Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev,llm]"
cp .env.example .env

# Quality checks (run ALL before finishing any task)
make lint                       # ruff check src tests
make format                     # ruff format src tests
make typecheck                  # mypy src
make test                       # pytest

# Run
make run-baseline               # Single-agent baseline
make run-multi                  # Multi-agent workflow
python -m multi_agent_research_lab.cli baseline --query "..."
python -m multi_agent_research_lab.cli multi-agent --query "..."
```

## Code Style & Conventions

- **Line length**: 100 chars max (`ruff` enforced)
- **Type hints**: Required everywhere. `mypy --strict` is enabled.
- **Pydantic v2**: All data models use `BaseModel` with `Field()`. Use `model_dump()`, `model_dump_json()`.
- **Python 3.11+**: Use `str | None` union syntax, `StrEnum`, built-in generics (`list[str]`, `dict[str, Any]`).
- **No hard-coded secrets**: All API keys via `.env` / `pydantic-settings`.
- **Docstrings**: Module-level and class/function-level docstrings in English.
- **Imports**: Absolute imports only (`from multi_agent_research_lab.core.schemas import ...`).
- **No comments**: Do not add comments unless explicitly asked.
- **Errors**: Use domain errors from `core/errors.py`. Agents raise `StudentTodoError` for unimplemented parts.
- **Agents pattern**: Inherit `BaseAgent`, implement `run(state: ResearchState) -> ResearchState`.

## Dependencies

### Core (always installed)
- `pydantic>=2.7`, `pydantic-settings>=2.3` — Data models + config
- `typer>=0.12`, `rich>=13.7` — CLI + pretty output
- `python-dotenv>=1.0` — .env loading
- `PyYAML>=6.0` — YAML config parsing
- `tenacity>=8.3` — Retry logic

### Optional `[llm]`
- `openai>=1.40` — OpenAI API client
- `langgraph>=0.2`, `langchain-core>=0.2` — Workflow graph
- `langsmith>=0.1` — Tracing

### Optional `[dev]`
- `pytest>=8.2`, `pytest-cov>=5.0` — Testing
- `ruff>=0.5` — Linting + formatting
- `mypy>=1.10` — Type checking
- `pre-commit>=3.7` — Git hooks (runs ruff check --fix + ruff format)

## Shared State (ResearchState)

The central data model passed through all agents:

| Field | Type | Written by |
|---|---|---|
| `request` | `ResearchQuery` | CLI (input) |
| `iteration` | `int` | Supervisor |
| `route_history` | `list[str]` | Supervisor |
| `sources` | `list[SourceDocument]` | Researcher |
| `research_notes` | `str \| None` | Researcher |
| `analysis_notes` | `str \| None` | Analyst |
| `final_answer` | `str \| None` | Writer |
| `agent_results` | `list[AgentResult]` | All agents |
| `trace` | `list[dict]` | All agents |
| `errors` | `list[str]` | All agents |

## TODO Markers

All student-required implementations are marked with `TODO(student)`. Find them:

```bash
grep -R "TODO(student)" -n src tests docs
```

## Benchmark Metrics

| Metric | How to measure |
|---|---|
| Latency | `perf_counter()` wall-clock time |
| Cost | Token usage from LLM response |
| Quality | Rubric 0-10 (peer review) |
| Citation coverage | claims with source / total claims |
| Failure rate | failed queries / total queries |

## Environment Variables

See `.env.example`. Key variables: `OPENAI_API_KEY`, `OPENAI_MODEL`, `LANGSMITH_API_KEY`, `TAVILY_API_KEY`, `MAX_ITERATIONS`, `TIMEOUT_SECONDS`, `LOG_LEVEL`.

## Guardrails

- `max_iterations`: 6 (configurable, max 20)
- `timeout_seconds`: 60 (configurable, max 600)
- Retry logic via `tenacity`
- Validation via Pydantic schemas
- `StudentTodoError` for unimplemented code paths
