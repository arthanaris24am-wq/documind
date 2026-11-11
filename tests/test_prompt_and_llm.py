"""Tests for grounded prompt formatting and Gemini LLM retry/backoff logic."""

from __future__ import annotations

import pytest

from app.config import LLMConfig
from app.llm import GeminiLLMClient, LLMConfigurationError, LLMError, LLMRateLimitError
from app.prompts import INSUFFICIENT_CONTEXT_RESPONSE, build_grounded_prompt
from app.vectorstore import RetrievedChunk


def test_grounded_prompt_includes_instructions_sources_and_fallback() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="cid-1",
            doc_id="did-1",
            filename="guide.pdf",
            page=2,
            chunk_index=0,
            text="Qdrant supports memory, docker, and cloud modes.",
            score=0.91,
        )
    ]
    prompt = build_grounded_prompt("What modes does Qdrant support?", chunks)
    assert "guide.pdf" in prompt
    assert "Page: 2" in prompt
    assert "cid-1" in prompt
    assert INSUFFICIENT_CONTEXT_RESPONSE in prompt
    assert "Question: What modes does Qdrant support?" in prompt


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FlakyModels:
    def __init__(self, failures_before_success: int, exc_factory) -> None:
        self.calls = 0
        self.failures_before_success = failures_before_success
        self.exc_factory = exc_factory

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise self.exc_factory()
        return _FakeResponse("Grounded response [guide.pdf, p.2].")


class _FakeGenAIClient:
    def __init__(self, models) -> None:
        self.models = models


def test_gemini_client_missing_api_key_raises() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiLLMClient(config=LLMConfig(api_key=None))


def test_gemini_client_retries_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    models = _FlakyModels(
        failures_before_success=2,
        exc_factory=lambda: RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded"),
    )
    client = GeminiLLMClient(
        config=LLMConfig(
            api_key="dummy",
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=8.0,
        ),
        client=_FakeGenAIClient(models),
        sleep_fn=sleeps.append,
    )

    answer = client.generate("hello")
    assert answer == "Grounded response [guide.pdf, p.2]."
    assert models.calls == 3
    assert len(sleeps) == 2
    assert sleeps[0] >= 1.0
    assert sleeps[1] >= 2.0


def test_gemini_client_exhausts_retries_and_raises_rate_limit_error() -> None:
    sleeps: list[float] = []
    models = _FlakyModels(
        failures_before_success=10,
        exc_factory=lambda: RuntimeError("429 rate limit"),
    )
    client = GeminiLLMClient(
        config=LLMConfig(api_key="dummy", max_retries=2, base_delay_seconds=0.5),
        client=_FakeGenAIClient(models),
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMRateLimitError, match="after 3 attempts"):
        client.generate("hello")
    assert models.calls == 3
    assert len(sleeps) == 2


def test_gemini_client_does_not_retry_non_retryable_error() -> None:
    sleeps: list[float] = []
    models = _FlakyModels(
        failures_before_success=5,
        exc_factory=lambda: ValueError("Invalid argument payload"),
    )
    client = GeminiLLMClient(
        config=LLMConfig(api_key="dummy", max_retries=3),
        client=_FakeGenAIClient(models),
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMError, match="Invalid argument payload"):
        client.generate("hello")
    assert models.calls == 1
    assert sleeps == []

