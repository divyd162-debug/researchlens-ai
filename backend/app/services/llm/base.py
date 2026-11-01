"""Abstract base class for LLM providers in ResearchLens AI."""
from abc import ABC, abstractmethod
from app.schemas.llm import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers in ResearchLens AI.

    Design & Architecture Principles:
    1. Synchronous Generation (Phase 4, Milestone 1):
       `generate()` is synchronous to match the current pipeline architecture.
    2. Explicit Error Propagation:
       Implementations must translate underlying provider/network errors into
       the standardized LLMError hierarchy (LLMProviderError, LLMQuotaExceededError,
       LLMContextLimitExceededError) instead of leaking raw vendor SDK exceptions.
    3. Citation Marker Contract:
       Providers must prompt/format answers using the standard citation marker syntax:
       `[CHUNK:<chunk_id>]`.
    4. Raw Provider Output:
       `generate()` returns raw text output with citation markers intact. Authoritative
       citation parsing and validation against retrieved chunk IDs is deferred to
       Phase 4, Milestone 2.
    """

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generates a text completion for the provided request.

        Args:
            request: Validated LLMRequest payload with prompt and optional context chunks.

        Returns:
            LLMResponse containing raw generated content and token metadata.

        Raises:
            LLMContextLimitExceededError: If the prompt/context exceeds the model context window.
            LLMQuotaExceededError: If rate limits or quota are exhausted.
            LLMProviderError: If the provider fails to generate a completion.
            LLMConfigurationError: If the provider is misconfigured or lacks valid credentials.
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validates that provider credentials and required settings are present.

        Returns:
            True if configuration is valid.

        Raises:
            LLMConfigurationError: If configuration is invalid.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical provider identifier (e.g. 'mock', 'omniroute')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name (e.g. 'mock-model-v1', 'gemini-fallback-combo')."""
        pass
