"""Deterministic Mock LLM Provider for ResearchLens AI."""
from typing import Any, Optional
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMConfigurationError,
    LLMContextLimitExceededError,
    LLMProviderError,
    LLMQuotaExceededError,
)


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for offline testing and development.

    Design & Architecture Principles:
    1. Deterministic & Fully Offline:
       Requires no internet connection, external API keys, or GPU resources.
    2. Explicit Non-Cognitive Nature:
       Produces predictable template-based text and does not claim semantic reasoning
       or actual LLM comprehension.
    3. Strict Citation Marker Attribution:
       When passages are provided, cites ONLY chunk IDs present in the request using
       the standard `[CHUNK:<chunk_id>]` syntax. Never invents or hallucinates IDs.
    4. Empty Context Fallback:
       Explicitly falls back to a standardized message when no context chunks are supplied.
    5. Testable Failure Injection:
       Accepts `simulate_error` parameter to test caller exception handling.
    """

    def __init__(
        self,
        model_name: str = "mock-model-v1",
        simulate_error: Optional[Any] = None,
    ):
        """Initializes the mock provider with configurable model name and error simulation.

        Args:
            model_name: Identifier for the simulated model.
            simulate_error: Optional error trigger ('quota', 'context_limit', 'provider',
                            'config', or an Exception instance/class).
        """
        self._model_name = model_name
        self.simulate_error = simulate_error

    @property
    def provider_name(self) -> str:
        """Returns canonical provider name."""
        return "mock"

    @property
    def model_name(self) -> str:
        """Returns active model name."""
        return self._model_name

    def validate_configuration(self) -> bool:
        """Validates provider configuration."""
        if self.simulate_error == "config":
            raise LLMConfigurationError("Mock provider simulated configuration failure.")
        return True

    def _handle_simulated_error(self) -> None:
        """Raises injected simulated exceptions if configured."""
        if not self.simulate_error:
            return

        if isinstance(self.simulate_error, type) and issubclass(self.simulate_error, Exception):
            raise self.simulate_error("Mock provider simulated exception.")
        if isinstance(self.simulate_error, Exception):
            raise self.simulate_error
        if self.simulate_error == "quota":
            raise LLMQuotaExceededError("Mock provider rate limit or quota exceeded.")
        if self.simulate_error == "context_limit":
            raise LLMContextLimitExceededError("Mock provider context window limit exceeded.")
        if self.simulate_error in ("provider", "error"):
            raise LLMProviderError("Mock provider generation failure.")
        if self.simulate_error == "config":
            raise LLMConfigurationError("Mock provider simulated configuration failure.")

        raise LLMProviderError(f"Mock provider simulated generic failure: {self.simulate_error}")

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generates a deterministic completion for the provided request.

        Args:
            request: Validated LLMRequest.

        Returns:
            LLMResponse containing deterministic text with raw `[CHUNK:<id>]` markers.

        Raises:
            LLMQuotaExceededError, LLMContextLimitExceededError, LLMProviderError,
            LLMConfigurationError if error simulation is triggered.
        """
        self._handle_simulated_error()

        # Handle empty context fallback
        if not request.context_chunks:
            fallback_text = (
                f"No relevant context was provided to answer the prompt: '{request.prompt}'. "
                "No source passages were supplied."
            )
            prompt_toks = max(1, len(request.prompt) // 4)
            completion_toks = max(1, len(fallback_text) // 4)
            return LLMResponse(
                content=fallback_text,
                model_name=self._model_name,
                provider_name=self.provider_name,
                finish_reason="stop",
                prompt_tokens=prompt_toks,
                completion_tokens=completion_toks,
                total_tokens=prompt_toks + completion_toks,
            )

        # Deterministic generation citing only supplied chunk IDs
        lines = [
            f"Mock synthesized response for query: '{request.prompt}'.",
            "Referenced evidence passages:",
        ]
        for idx, chunk in enumerate(request.context_chunks, start=1):
            snippet = chunk.content[:80].strip()
            lines.append(
                f"{idx}. From '{chunk.document_title}' (page {chunk.page_number}): "
                f"\"{snippet}\" [CHUNK:{chunk.chunk_id}]"
            )

        # Concluding summary tag list
        summary_tags = " ".join(f"[CHUNK:{c.chunk_id}]" for c in request.context_chunks)
        lines.append(f"Summary: synthesized from {len(request.context_chunks)} source passage(s) {summary_tags}.")

        content = "\n".join(lines)

        # Predictable token heuristic
        prompt_toks = max(
            1,
            (len(request.prompt) // 4)
            + sum(len(c.content) // 4 for c in request.context_chunks)
            + 10,
        )
        completion_toks = max(1, len(content) // 4)

        return LLMResponse(
            content=content,
            model_name=self._model_name,
            provider_name=self.provider_name,
            finish_reason="stop",
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            total_tokens=prompt_toks + completion_toks,
        )
