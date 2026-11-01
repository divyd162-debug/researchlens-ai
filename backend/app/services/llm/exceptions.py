"""Typed exception hierarchy for LLM providers in ResearchLens AI."""


class LLMError(Exception):
    """Base exception for all LLM provider and generation errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is misconfigured, unconfigured, or unknown."""
    pass


class LLMProviderError(LLMError):
    """Raised when an LLM provider encounters an upstream generation error or API failure."""
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when an LLM provider rate limit or token/call quota is exceeded."""
    pass


class LLMContextLimitExceededError(LLMProviderError):
    """Raised when input prompt or context exceeds the model's maximum context window."""
    pass
