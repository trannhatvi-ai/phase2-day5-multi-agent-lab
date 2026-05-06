from __future__ import annotations

from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openai_model
    assert settings.max_iterations >= 1
    assert settings.timeout_seconds >= 5


def test_settings_max_iterations_bounds() -> None:
    settings = Settings()
    assert 1 <= settings.max_iterations <= 20
    assert 5 <= settings.timeout_seconds <= 600
