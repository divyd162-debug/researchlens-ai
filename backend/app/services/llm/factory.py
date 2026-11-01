"""LLM provider factory and registry for ResearchLens AI."""
from typing import Any, Dict, Optional, Type
from app.core.config import settings
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMConfigurationError
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.omniroute_provider import OmniRouteLLMProvider

# Canonical registry of supported LLM provider implementations
_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "mock": MockLLMProvider,
    "omniroute": OmniRouteLLMProvider,
}


def get_llm_provider(
    provider_name: Optional[str] = None,
    **provider_kwargs: Any,
) -> BaseLLMProvider:
    """Resolves and instantiates an LLM provider based on configuration or explicit argument.

    Args:
        provider_name: Optional explicit provider identifier (e.g. 'mock').
                       If None, resolves from settings.LLM_PROVIDER.
        **provider_kwargs: Additional keyword arguments forwarded to provider constructor.

    Returns:
        Configured instance of BaseLLMProvider.

    Raises:
        LLMConfigurationError: If no provider is specified or the requested provider
                               is unknown/unregistered. Never silently falls back.
    """
    raw_name = provider_name if provider_name is not None else getattr(settings, "LLM_PROVIDER", None)
    if not raw_name or not str(raw_name).strip():
        raise LLMConfigurationError(
            "LLM provider is not configured. Set LLM_PROVIDER in your environment "
            "or provide an explicit provider name."
        )

    canonical_name = str(raw_name).strip().lower()

    provider_cls = _PROVIDER_REGISTRY.get(canonical_name)
    if provider_cls is None:
        supported = sorted(list(_PROVIDER_REGISTRY.keys()))
        raise LLMConfigurationError(
            f"Unsupported or unconfigured LLM provider: '{canonical_name}'. "
            f"Supported providers in current milestone: {supported}. "
            "Never silently falling back to a default provider."
        )

    provider = provider_cls(**provider_kwargs)
    provider.validate_configuration()
    return provider
