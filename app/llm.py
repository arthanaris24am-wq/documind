"""Abstract LLM client and Gemini API adapter with exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base error for LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when required LLM credentials or settings are missing."""


class LLMRateLimitError(LLMError):
    """Raised when rate-limit retries are exhausted."""


class LLMClient(ABC):
    """Abstract interface for language model text generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion from the given prompt."""


def _is_retryable_exception(exc: Exception) -> bool:
    """Determine if an exception is transient (rate limit or temporary server error)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code in {429, 500, 502, 503, 504}:
            return True
    message = str(exc).lower()
    retry_markers = ("429", "rate limit", "resource_exhausted", "quota", "unavailable", "503")
    return any(marker in message for marker in retry_markers)


class GeminiLLMClient(LLMClient):
    """Google Gemini LLM client with exponential backoff retry for rate limits."""

    def __init__(
        self,
        config: LLMConfig,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._sleep = sleep_fn
        if client is not None:
            self._client = client
        else:
            if not config.api_key:
                raise LLMConfigurationError(
                    "GEMINI_API_KEY is not configured. Set it in .env or environment variables."
                )
            self._client = genai.Client(api_key=config.api_key)

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def generate(self, prompt: str) -> str:
        max_attempts = self._config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=self._config.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.max_output_tokens,
                    ),
                )
                text = getattr(response, "text", None)
                if not text:
                    raise LLMError("Gemini returned an empty response")
                return str(text).strip()
            except Exception as exc:
                last_error = exc
                if not _is_retryable_exception(exc) or attempt >= max_attempts - 1:
                    if _is_retryable_exception(exc):
                        raise LLMRateLimitError(
                            f"Gemini rate limit/transient error after {max_attempts} attempts: "
                            f"{exc}"
                        ) from exc
                    raise LLMError(f"Gemini generation failed: {exc}") from exc

                delay = min(
                    self._config.base_delay_seconds * (2**attempt),
                    self._config.max_delay_seconds,
                )
                jitter = random.uniform(0.0, delay * 0.25)
                wait_seconds = delay + jitter
                logger.warning(
                    "Retryable Gemini API error on attempt %d/%d (%s). Backing off %.2fs.",
                    attempt + 1,
                    max_attempts,
                    exc,
                    wait_seconds,
                )
                self._sleep(wait_seconds)

        raise LLMRateLimitError(f"Gemini request failed: {last_error}")

