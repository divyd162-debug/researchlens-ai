"""Unit tests for LLM schemas, BaseLLMProvider contract, MockLLMProvider, and factory."""
import re
import pytest
from pydantic import ValidationError

from app.schemas.llm import LLMChunkContext, LLMRequest, LLMResponse
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMConfigurationError,
    LLMContextLimitExceededError,
    LLMError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.llm.factory import get_llm_provider
from app.services.llm.mock_provider import MockLLMProvider


class TestLLMSchemas:
    """Tests verifying strict Pydantic validation on LLM DTOs."""

    def test_valid_llm_chunk_context(self):
        """Verifies valid LLMChunkContext construction."""
        chunk = LLMChunkContext(
            chunk_id=42,
            content="Attention is all you need for transformer architectures.",
            page_number=1,
            document_title="Attention Paper",
        )
        assert chunk.chunk_id == 42
        assert chunk.page_number == 1
        assert chunk.document_title == "Attention Paper"

    def test_chunk_context_rejects_non_positive_chunk_id(self):
        """Verifies chunk_id must be strictly positive (> 0)."""
        with pytest.raises(ValidationError):
            LLMChunkContext(
                chunk_id=0,
                content="Text",
                page_number=1,
                document_title="Title",
            )
        with pytest.raises(ValidationError):
            LLMChunkContext(
                chunk_id=-5,
                content="Text",
                page_number=1,
                document_title="Title",
            )

    def test_chunk_context_rejects_non_positive_page_number(self):
        """Verifies page_number must be >= 1."""
        with pytest.raises(ValidationError):
            LLMChunkContext(
                chunk_id=1,
                content="Text",
                page_number=0,
                document_title="Title",
            )

    def test_chunk_context_rejects_whitespace_content_or_title(self):
        """Verifies content and document_title cannot be empty or pure whitespace."""
        with pytest.raises(ValidationError):
            LLMChunkContext(
                chunk_id=1,
                content="   ",
                page_number=1,
                document_title="Title",
            )
        with pytest.raises(ValidationError):
            LLMChunkContext(
                chunk_id=1,
                content="Valid content",
                page_number=1,
                document_title="  \t\n ",
            )

    def test_valid_llm_request_defaults(self):
        """Verifies valid LLMRequest with default parameters."""
        req = LLMRequest(prompt="What is self-attention?")
        assert req.prompt == "What is self-attention?"
        assert req.context_chunks == []
        assert req.system_instruction is None
        assert req.temperature == 0.0
        assert req.max_tokens is None

    def test_llm_request_rejects_empty_prompt(self):
        """Verifies prompt cannot be empty or whitespace."""
        with pytest.raises(ValidationError):
            LLMRequest(prompt="")
        with pytest.raises(ValidationError):
            LLMRequest(prompt="   \n\t  ")

    def test_llm_request_temperature_bounds(self):
        """Verifies temperature must be within [0.0, 2.0]."""
        # Valid bounds
        assert LLMRequest(prompt="test", temperature=0.0).temperature == 0.0
        assert LLMRequest(prompt="test", temperature=2.0).temperature == 2.0

        # Out of bounds
        with pytest.raises(ValidationError):
            LLMRequest(prompt="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMRequest(prompt="test", temperature=2.1)

    def test_llm_request_max_tokens_positive(self):
        """Verifies max_tokens must be strictly positive if provided."""
        assert LLMRequest(prompt="test", max_tokens=100).max_tokens == 100
        with pytest.raises(ValidationError):
            LLMRequest(prompt="test", max_tokens=0)
        with pytest.raises(ValidationError):
            LLMRequest(prompt="test", max_tokens=-10)

    def test_valid_llm_response(self):
        """Verifies valid LLMResponse construction."""
        resp = LLMResponse(
            content="Answer text [CHUNK:10]",
            model_name="mock-model-v1",
            provider_name="mock",
            finish_reason="stop",
            prompt_tokens=25,
            completion_tokens=10,
            total_tokens=35,
        )
        assert resp.content == "Answer text [CHUNK:10]"
        assert resp.prompt_tokens == 25
        assert resp.total_tokens == 35

    def test_llm_response_rejects_negative_token_usage(self):
        """Verifies token counts must be non-negative."""
        with pytest.raises(ValidationError):
            LLMResponse(
                content="text",
                model_name="m",
                provider_name="p",
                prompt_tokens=-1,
            )


class TestBaseLLMProviderContract:
    """Tests verifying abstract interface rules."""

    def test_cannot_instantiate_abstract_base_class(self):
        """Verifies BaseLLMProvider cannot be instantiated without implementations."""
        with pytest.raises(TypeError):
            BaseLLMProvider()


class TestMockLLMProvider:
    """Tests for MockLLMProvider deterministic generation and error simulation."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            LLMChunkContext(
                chunk_id=101,
                content="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
                page_number=4,
                document_title="Attention Is All You Need",
            ),
            LLMChunkContext(
                chunk_id=102,
                content="The Transformer follows an overall architecture using stacked self-attention and point-wise, fully connected layers.",
                page_number=3,
                document_title="Attention Is All You Need",
            ),
        ]

    def test_mock_provider_properties(self):
        """Verifies provider_name and model_name properties."""
        provider = MockLLMProvider(model_name="custom-mock-v2")
        assert provider.provider_name == "mock"
        assert provider.model_name == "custom-mock-v2"
        assert provider.validate_configuration() is True

    def test_deterministic_generation_with_context(self, sample_chunks):
        """Verifies deterministic output referencing all and only supplied chunk IDs."""
        provider = MockLLMProvider()
        req = LLMRequest(
            prompt="How does multi-head attention work?",
            context_chunks=sample_chunks,
        )

        resp1 = provider.generate(req)
        resp2 = provider.generate(req)

        # 1. Output is 100% deterministic
        assert resp1.content == resp2.content
        assert resp1.prompt_tokens == resp2.prompt_tokens
        assert resp1.completion_tokens == resp2.completion_tokens
        assert resp1.total_tokens == resp2.total_tokens

        # 2. Contains standard citation marker syntax [CHUNK:<id>]
        assert "[CHUNK:101]" in resp1.content
        assert "[CHUNK:102]" in resp1.content

        # 3. Only cites supplied IDs; no invented IDs
        found_ids = [int(m) for m in re.findall(r"\[CHUNK:(\d+)\]", resp1.content)]
        assert set(found_ids) == {101, 102}

        # 4. Finish reason and provider metadata
        assert resp1.finish_reason == "stop"
        assert resp1.provider_name == "mock"
        assert resp1.model_name == "mock-model-v1"
        assert resp1.prompt_tokens > 0
        assert resp1.completion_tokens > 0

    def test_empty_context_fallback(self):
        """Verifies explicit fallback when context_chunks is empty."""
        provider = MockLLMProvider()
        req = LLMRequest(prompt="What is quantum gravity?", context_chunks=[])

        resp = provider.generate(req)

        # Fallback message
        assert "No relevant context was provided" in resp.content
        assert "quantum gravity" in resp.content

        # Zero citation tags in empty context output
        found_markers = re.findall(r"\[CHUNK:\d+\]", resp.content)
        assert len(found_markers) == 0

    def test_error_simulation_quota_exceeded(self):
        """Verifies simulated quota error triggers LLMQuotaExceededError."""
        provider = MockLLMProvider(simulate_error="quota")
        req = LLMRequest(prompt="Test prompt")
        with pytest.raises(LLMQuotaExceededError, match="rate limit or quota exceeded"):
            provider.generate(req)

    def test_error_simulation_context_limit(self):
        """Verifies simulated context limit triggers LLMContextLimitExceededError."""
        provider = MockLLMProvider(simulate_error="context_limit")
        req = LLMRequest(prompt="Test prompt")
        with pytest.raises(LLMContextLimitExceededError, match="context window limit"):
            provider.generate(req)

    def test_error_simulation_provider_failure(self):
        """Verifies simulated generic provider failure triggers LLMProviderError."""
        provider = MockLLMProvider(simulate_error="provider")
        req = LLMRequest(prompt="Test prompt")
        with pytest.raises(LLMProviderError, match="generation failure"):
            provider.generate(req)

    def test_error_simulation_config_failure(self):
        """Verifies simulated configuration failure triggers LLMConfigurationError."""
        provider = MockLLMProvider(simulate_error="config")
        with pytest.raises(LLMConfigurationError, match="configuration failure"):
            provider.validate_configuration()


class TestLLMProviderFactory:
    """Tests for get_llm_provider factory resolution and validation."""

    def test_factory_resolves_mock_provider(self):
        """Verifies factory returns MockLLMProvider for 'mock'."""
        provider = get_llm_provider("mock")
        assert isinstance(provider, MockLLMProvider)
        assert provider.provider_name == "mock"

    def test_factory_resolves_case_insensitively(self):
        """Verifies provider name resolution is case-insensitive."""
        p1 = get_llm_provider("MOCK")
        p2 = get_llm_provider("  MoCk  ")
        assert isinstance(p1, MockLLMProvider)
        assert isinstance(p2, MockLLMProvider)

    def test_factory_forwards_constructor_kwargs(self):
        """Verifies factory passes custom kwargs to provider constructor."""
        provider = get_llm_provider("mock", model_name="custom-test-model")
        assert provider.model_name == "custom-test-model"

    def test_factory_defaults_to_settings_provider(self, monkeypatch):
        """Verifies factory resolves settings.LLM_PROVIDER when provider_name is None."""
        monkeypatch.setattr("app.core.config.settings.LLM_PROVIDER", "mock")
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_factory_rejects_unknown_provider_without_fallback(self):
        """Verifies unknown provider raises LLMConfigurationError and never silently falls back."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm_provider("unsupported-provider")
        assert "Unsupported or unconfigured LLM provider: 'unsupported-provider'" in str(exc_info.value)
        assert "['mock', 'omniroute']" in str(exc_info.value)

        with pytest.raises(LLMConfigurationError):
            get_llm_provider("openai")

    def test_factory_resolves_omniroute_provider_when_configured(self, monkeypatch):
        """Verifies factory resolves OmniRouteLLMProvider when configured."""
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_MODEL", "gemini-fallback-combo")
        from app.services.llm.omniroute_provider import OmniRouteLLMProvider
        provider = get_llm_provider("omniroute")
        assert isinstance(provider, OmniRouteLLMProvider)
        assert provider.provider_name == "omniroute"

    def test_factory_rejects_omniroute_provider_when_invalid_config(self, monkeypatch):
        """Verifies factory raises LLMConfigurationError and does not fallback when configuration is invalid."""
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_BASE_URL", "")
        with pytest.raises(LLMConfigurationError, match="OmniRoute base URL"):
            get_llm_provider("omniroute")

    def test_factory_rejects_unconfigured_or_empty_provider(self, monkeypatch):
        """Verifies empty or whitespace provider name raises LLMConfigurationError."""
        monkeypatch.setattr("app.core.config.settings.LLM_PROVIDER", "")
        with pytest.raises(LLMConfigurationError, match="LLM provider is not configured"):
            get_llm_provider()
