"""Unit tests for OmniRoute OpenAI-Compatible LLM Provider."""
import json
import logging
from unittest.mock import MagicMock
import httpx
import pytest

from app.core.config import load_app_env, settings
from app.schemas.chat import ChatScopeType
from app.schemas.llm import LLMChunkContext, LLMRequest
from app.schemas.rag import RetrievedContextChunk
from app.services.llm.exceptions import (
    LLMConfigurationError,
    LLMContextLimitExceededError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.llm.factory import get_llm_provider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.omniroute_provider import (
    DEFAULT_OMNIROUTE_BASE_URL,
    DEFAULT_OMNIROUTE_MODEL,
    OmniRouteLLMProvider,
    _sanitize_error_message,
    _strip_reasoning_tags,
)
from app.services.rag_service import RAGService


def _create_mock_response(
    content: str,
    status_code: int = 200,
    model: str = "gemini-fallback-combo",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    error_code: str = "",
    error_message: str = "",
) -> httpx.Response:
    """Helper creating a mocked httpx.Response following OpenAI chat completions schema."""
    if status_code == 200:
        body = {
            "id": "chatcmpl-test-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    else:
        body = {
            "error": {
                "code": error_code or "error",
                "message": error_message or "An error occurred",
                "type": "invalid_request_error" if status_code == 400 else "api_error",
            }
        }

    request = httpx.Request("POST", "http://localhost:20128/v1/chat/completions")
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode("utf-8"),
        request=request,
        headers={"Content-Type": "application/json"},
    )


class TestOmniRouteConfiguration:
    """Tests provider initialization, properties, and validation."""

    def test_provider_name_and_default_model(self):
        """Verifies provider identity and default combo configuration."""
        provider = OmniRouteLLMProvider(client=MagicMock())
        assert provider.provider_name == "omniroute"
        assert provider.model_name == DEFAULT_OMNIROUTE_MODEL
        assert provider.base_url == DEFAULT_OMNIROUTE_BASE_URL
        assert provider.timeout_seconds == 60.0
        assert provider.max_retries == 1

    def test_custom_configuration(self):
        """Verifies custom base URL, model override, timeout, and API key."""
        provider = OmniRouteLLMProvider(
            base_url="http://custom-proxy:8080/v1/",
            api_key="secret-gateway-key",
            model_name="claude-3-5-sonnet",
            timeout_seconds=45.0,
            max_retries=2,
            client=MagicMock(),
        )
        assert provider.base_url == "http://custom-proxy:8080/v1"
        assert provider.model_name == "claude-3-5-sonnet"
        assert provider.timeout_seconds == 45.0
        assert provider.max_retries == 2

    def test_validation_fails_on_empty_base_url(self):
        """Verifies missing or non-HTTP base URL raises LLMConfigurationError."""
        provider = OmniRouteLLMProvider(base_url="", model_name="auto")
        with pytest.raises(LLMConfigurationError, match="must be a valid HTTP or HTTPS URL"):
            provider.validate_configuration()

        provider2 = OmniRouteLLMProvider(base_url="ftp://invalid", model_name="auto")
        with pytest.raises(LLMConfigurationError, match="must be a valid HTTP or HTTPS URL"):
            provider2.validate_configuration()

    def test_validation_fails_on_empty_model_name(self):
        """Verifies empty model name raises LLMConfigurationError."""
        provider = OmniRouteLLMProvider(base_url="http://localhost:20128/v1", model_name="")
        with pytest.raises(LLMConfigurationError, match="model name cannot be empty"):
            provider.validate_configuration()

    def test_settings_validate_llm_settings(self, monkeypatch):
        """Verifies Settings.validate_llm_settings() validates omniroute configuration."""
        monkeypatch.setattr("app.core.config.settings.LLM_PROVIDER", "omniroute")
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_BASE_URL", "")
        with pytest.raises(LLMConfigurationError, match="OMNIROUTE_BASE_URL cannot be empty"):
            settings.validate_llm_settings()

        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_MODEL", "")
        with pytest.raises(LLMConfigurationError, match="OMNIROUTE_MODEL cannot be empty"):
            settings.validate_llm_settings()

        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_MODEL", "gemini-fallback-combo")
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_TIMEOUT_SECONDS", 0.0)
        with pytest.raises(LLMConfigurationError, match="OMNIROUTE_TIMEOUT_SECONDS must be positive"):
            settings.validate_llm_settings()

        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_TIMEOUT_SECONDS", 60.0)
        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_MAX_RETRIES", -1)
        with pytest.raises(LLMConfigurationError, match="OMNIROUTE_MAX_RETRIES must be non-negative"):
            settings.validate_llm_settings()

        monkeypatch.setattr("app.core.config.settings.OMNIROUTE_MAX_RETRIES", 1)
        # Should now pass without raising
        settings.validate_llm_settings()


class TestOmniRoutePromptFormatting:
    """Tests evidence context structuring and system prompt grounding directives."""

    def test_evidence_context_formatting(self):
        """Verifies chunks format into structured passages with [CHUNK:id] header."""
        chunks = [
            LLMChunkContext(
                chunk_id=101,
                content="Attention replaces recurrence in seq2seq models.",
                page_number=3,
                document_title="Attention Paper",
            ),
            LLMChunkContext(
                chunk_id=102,
                content="Self-attention yields faster training times.",
                page_number=5,
                document_title="Attention Paper",
            ),
        ]
        evidence = OmniRouteLLMProvider._build_evidence_context(chunks)
        assert '[CHUNK:101] From "Attention Paper" (Page 3):' in evidence
        assert "Attention replaces recurrence in seq2seq models." in evidence
        assert '[CHUNK:102] From "Attention Paper" (Page 5):' in evidence
        assert "---" in evidence

    def test_system_instruction_directives(self):
        """Verifies grounding directives, citation format rules, numerical fidelity, and experimental conditions."""
        instruction = OmniRouteLLMProvider._build_system_instruction()
        assert "ResearchLens AI" in instruction
        assert "[CHUNK:<id>]" in instruction
        assert "insufficient evidence" in instruction
        # Numerical fidelity and rounding prohibition
        assert "NUMERICAL & TABLE FIDELITY" in instruction
        assert "Never round, truncate, or substitute approximations" in instruction
        # Experimental condition and table disambiguation
        assert "EXPERIMENTAL CONDITIONS & TABLE DISAMBIGUATION" in instruction
        assert "SMOTE" in instruction
        # Ambiguity and conflict reporting
        assert "AMBIGUITY & INCONSISTENCY" in instruction

    def test_user_prompt_includes_precision_and_condition_directives(self):
        """Verifies user prompt delivered to OmniRoute contains immediate precision and condition directives."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="Answer quoting exact values [CHUNK:101].",
        )
        provider = OmniRouteLLMProvider(client=mock_client)
        chunks = [
            LLMChunkContext(
                chunk_id=101,
                content="Table 6: CICIDS2017 without SMOTE. Naive Bayes achieved Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
                page_number=7,
                document_title="NIDS Paper",
            )
        ]
        provider.generate(LLMRequest(
            prompt="What is the Naive Bayes Precision and Recall without SMOTE?",
            context_chunks=chunks,
        ))

        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args.kwargs["json"]
        user_message = payload["messages"][1]["content"]

        assert "GROUNDING & PRECISION DIRECTIVES:" in user_message
        assert "Quote exact numerical values and metrics with their full decimal precision" in user_message
        assert "with vs. without SMOTE" in user_message

    def test_empty_context_returns_insufficient_evidence_without_api_call(self):
        """Verifies empty context passages return early without invoking OmniRoute."""
        mock_client = MagicMock()
        provider = OmniRouteLLMProvider(client=mock_client)

        req = LLMRequest(prompt="What is self-attention?", context_chunks=[])
        resp = provider.generate(req)

        assert resp.provider_name == "omniroute"
        assert "No relevant context was provided" in resp.content
        mock_client.post.assert_not_called()


class TestOmniRouteGenerationAndCitations:
    """Tests response parsing, citation survival, and reasoning tag handling."""

    def test_successful_grounded_generation(self):
        """Verifies grounded answer with citations parses correctly."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="Self-attention computes representations without recurrence [CHUNK:101].",
            model="gemini-fallback-combo",
            prompt_tokens=150,
            completion_tokens=40,
        )

        provider = OmniRouteLLMProvider(client=mock_client, api_key="test-key")
        chunks = [
            LLMChunkContext(
                chunk_id=101,
                content="Attention mechanisms compute representations without recurrence.",
                page_number=1,
                document_title="Transformer Paper",
            )
        ]
        resp = provider.generate(LLMRequest(prompt="Explain self-attention", context_chunks=chunks))

        assert resp.provider_name == "omniroute"
        assert resp.model_name == "gemini-fallback-combo"
        assert "[CHUNK:101]" in resp.content
        assert resp.prompt_tokens == 150
        assert resp.completion_tokens == 40
        assert resp.total_tokens == 190

        # Verify outgoing payload
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://localhost:20128/v1/chat/completions"
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        payload = call_args.kwargs["json"]
        assert payload["model"] == "gemini-fallback-combo"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_strip_reasoning_tags(self):
        """Verifies <think>, <thought>, and <reasoning> blocks from reasoning models are stripped cleanly."""
        raw = "<think>Let me analyze the paper... [CHUNK:101] supports the claim.</think>Self-attention is parallelizable [CHUNK:101]."
        cleaned = _strip_reasoning_tags(raw)
        assert cleaned == "Self-attention is parallelizable [CHUNK:101]."
        assert "<think>" not in cleaned

        # Test <thought> variant
        raw_thought = "<thought>Examining table metrics...</thought>The FAR is 0.1820 [CHUNK:102]."
        assert _strip_reasoning_tags(raw_thought) == "The FAR is 0.1820 [CHUNK:102]."

        # Test <reasoning> variant
        raw_reasoning = "<reasoning>Step 1: Check CICIDS2017 without SMOTE</reasoning>Precision is 0.6364 [CHUNK:103]."
        assert _strip_reasoning_tags(raw_reasoning) == "Precision is 0.6364 [CHUNK:103]."

        # Test unclosed reasoning blocks with leading whitespace
        raw_unclosed = "   <think>Token limit hit during internal chain-of-thought..."
        assert _strip_reasoning_tags(raw_unclosed) == ""

    def test_safety_blocked_response_handling(self):
        """Verifies content filter finish reason produces safe message."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            finish_reason="content_filter",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        resp = provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

        assert "safety filters" in resp.content


class TestOmniRouteErrorHandlingAndRetries:
    """Tests granular error mapping, failure fast on 429/timeout, and transient retries."""

    def test_authentication_error_mapping(self):
        """Verifies HTTP 401 raises LLMConfigurationError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            status_code=401,
            error_message="Invalid gateway API key",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMConfigurationError, match="authentication failed"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

    def test_quota_exceeded_error_fails_fast_without_retry(self):
        """Verifies HTTP 429 raises LLMQuotaExceededError without secondary retries."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            status_code=429,
            error_code="rate_limit_exceeded",
            error_message="All combo providers exhausted quota",
        )

        provider = OmniRouteLLMProvider(client=mock_client, max_retries=2)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMQuotaExceededError, match="exhausted rate limits or quota"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

        # Must NOT retry 429
        assert mock_client.post.call_count == 1

    def test_context_window_error_mapping(self):
        """Verifies HTTP 400 with context_length_exceeded raises LLMContextLimitExceededError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            status_code=400,
            error_code="context_length_exceeded",
            error_message="Prompt exceeds model context window of 8192 tokens",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMContextLimitExceededError, match="exceed the model context length"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

    def test_model_not_found_error_mapping(self):
        """Verifies HTTP 400 with model_not_found raises LLMConfigurationError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            status_code=400,
            error_code="model_not_found",
            error_message="The combo 'invalid-combo' does not exist in OmniRoute",
        )

        provider = OmniRouteLLMProvider(client=mock_client, model_name="invalid-combo")
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMConfigurationError, match="was not found"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

    def test_generic_400_bad_request_mapping(self):
        """Verifies other HTTP 400 bad requests raise LLMProviderError."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="",
            status_code=400,
            error_code="invalid_payload",
            error_message="Temperature must be between 0 and 2",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMProviderError, match="rejected the request"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

    def test_gateway_unreachable_error_mapping(self):
        """Verifies httpx.ConnectError raises LLMProviderError with actionable startup message."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        provider = OmniRouteLLMProvider(client=mock_client, max_retries=1)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMProviderError, match="Ensure the OmniRoute service is running"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

        # Retried once on transport drop (1 original + 1 retry = 2 attempts)
        assert mock_client.post.call_count == 2

    def test_timeout_fails_fast_without_retry(self):
        """Verifies httpx.ReadTimeout raises LLMProviderError without compounding retries."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ReadTimeout("The read operation timed out")

        provider = OmniRouteLLMProvider(client=mock_client, timeout_seconds=30.0, max_retries=2)
        chunks = [LLMChunkContext(chunk_id=1, content="C", page_number=1, document_title="T")]
        with pytest.raises(LLMProviderError, match="timed out after 30.0s"):
            provider.generate(LLMRequest(prompt="Query", context_chunks=chunks))

        # Must NOT retry timeout
        assert mock_client.post.call_count == 1

    def test_secret_sanitization_in_error_messages(self):
        """Verifies sensitive API keys and Bearer tokens are redacted from error logs."""
        secret = "secret-omniroute-key-9999"
        raw_msg = f"Failed contacting http://localhost:20128/v1 with Authorization: Bearer {secret} and key={secret}"
        sanitized = _sanitize_error_message(raw_msg, secret)
        assert secret not in sanitized
        assert "Bearer [REDACTED]" in sanitized
        assert "key=[REDACTED]" in sanitized


class TestOmniRouteRAGServiceIntegration:
    """Tests RAGService orchestration using OmniRouteLLMProvider with primary and fallback responses."""

    def test_rag_service_with_omniroute_primary_generates_grounded_response(self):
        """Verifies RAGService coordinates retrieval, OmniRoute generation, and citation validation."""
        generated_answer = (
            "Machine learning evolution in IDS began with signature rules and transitioned to "
            "deep recurrent architectures [CHUNK:10]."
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content=generated_answer,
            model="gemini-fallback-combo",
            prompt_tokens=80,
            completion_tokens=25,
        )

        omniroute_provider = OmniRouteLLMProvider(client=mock_client)

        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=10,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Evolution of ML in IDS transitioned from signatures to deep learning.",
            page_number=1,
            document_title="IDS Survey Paper",
            similarity_score=0.92,
            char_start=0,
            char_end=70,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=omniroute_provider,
            retrieval_service=mock_retrieval,
        )

        db_mock = MagicMock()
        response = rag_service.answer_query(
            db=db_mock,
            query="explain Evolution of Machine Learning and Deep Learning for IDS",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        assert response.provider_name == "omniroute"
        assert response.model_name == "gemini-fallback-combo"
        assert response.answer == generated_answer
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 10
        assert response.citations[0].document_title == "IDS Survey Paper"

    def test_rag_service_with_omniroute_fallback_model_preserves_citations(self):
        """Verifies that when OmniRoute fails over to a secondary model, citations still pass validation."""
        # Simulated response from fallback provider (e.g. Groq/Llama or OpenAI)
        fallback_answer = (
            "According to the research survey, intrusion detection adopted deep neural "
            "networks for high-dimensional feature learning [CHUNK:10]."
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content=fallback_answer,
            model="llama-3.3-70b-versatile",  # Effective model returned by fallback step
            prompt_tokens=85,
            completion_tokens=30,
        )

        omniroute_provider = OmniRouteLLMProvider(client=mock_client)

        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=10,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Intrusion detection adopted deep neural networks for feature learning.",
            page_number=2,
            document_title="IDS Survey Paper",
            similarity_score=0.88,
            char_start=0,
            char_end=65,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=omniroute_provider,
            retrieval_service=mock_retrieval,
        )

        db_mock = MagicMock()
        response = rag_service.answer_query(
            db=db_mock,
            query="explain Deep Learning for IDS",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        assert response.provider_name == "omniroute"
        assert response.model_name == "llama-3.3-70b-versatile"
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 10
        assert response.citations[0].document_title == "IDS Survey Paper"


class TestConfigurationLoadingAndResolution:
    """Tests verifying environment loading, Settings reloading, and factory selection."""

    def test_load_app_env_loads_backend_env_when_run_from_workspace_root(self, tmp_path, monkeypatch):
        """Verifies load_app_env finds backend/.env regardless of current working directory."""
        b_dir = tmp_path / "backend"
        b_dir.mkdir()
        env_file = b_dir / ".env"
        env_file.write_text("LLM_PROVIDER=omniroute\nOMNIROUTE_MODEL=test-combo-123\n")

        monkeypatch.setenv("ENV_FILE", str(env_file))
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OMNIROUTE_MODEL", raising=False)

        load_app_env(override=True)
        settings.reload()

        assert settings.LLM_PROVIDER == "omniroute"
        assert settings.OMNIROUTE_MODEL == "test-combo-123"

    def test_settings_reload_updates_from_os_environ(self, monkeypatch):
        """Verifies settings.reload() dynamically reflects changes to os.environ."""
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        settings.reload()
        assert settings.LLM_PROVIDER == "mock"

        monkeypatch.setenv("LLM_PROVIDER", "omniroute")
        monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://gateway.test:20128/v1")
        monkeypatch.setenv("OMNIROUTE_MODEL", "custom-omni-combo")
        settings.reload()
        assert settings.LLM_PROVIDER == "omniroute"
        assert settings.OMNIROUTE_BASE_URL == "http://gateway.test:20128/v1"
        assert settings.OMNIROUTE_MODEL == "custom-omni-combo"

    def test_factory_selects_omniroute_when_env_is_omniroute(self, monkeypatch):
        """Verifies get_llm_provider() resolves OmniRouteLLMProvider when LLM_PROVIDER=omniroute."""
        monkeypatch.setattr(settings, "LLM_PROVIDER", "omniroute")
        monkeypatch.setattr(settings, "OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        monkeypatch.setattr(settings, "OMNIROUTE_MODEL", "gemini-fallback-combo")

        provider = get_llm_provider()
        assert isinstance(provider, OmniRouteLLMProvider)
        assert provider.provider_name == "omniroute"
        assert provider.model_name == "gemini-fallback-combo"

    def test_factory_selects_mock_when_env_is_mock(self, monkeypatch):
        """Verifies get_llm_provider() resolves MockLLMProvider when LLM_PROVIDER=mock."""
        monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)
        assert provider.provider_name == "mock"

    def test_factory_never_falls_back_to_mock_on_invalid_omniroute_config(self, monkeypatch):
        """Verifies that an invalid OmniRoute configuration raises LLMConfigurationError instead of falling back to mock."""
        monkeypatch.setattr(settings, "LLM_PROVIDER", "omniroute")
        monkeypatch.setattr(settings, "OMNIROUTE_BASE_URL", "")

        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm_provider()

        assert "OmniRoute base URL must be a valid HTTP or HTTPS URL" in str(exc_info.value)

    def test_settings_validate_llm_settings_rejects_non_http_omniroute_url(self, monkeypatch):
        """Verifies settings.validate_llm_settings() rejects non-HTTP(S) base URLs."""
        monkeypatch.setattr(settings, "LLM_PROVIDER", "omniroute")
        monkeypatch.setattr(settings, "OMNIROUTE_BASE_URL", "ftp://invalid-url")

        with pytest.raises(LLMConfigurationError, match="valid HTTP or HTTPS URL"):
            settings.validate_llm_settings()


class TestChatAndRAGRoutingToOmniRoute:
    """Tests verifying RAG and Chat workflows cleanly route to OmniRoute without mock bypass."""

    def test_rag_service_sends_configured_auth_and_endpoint(self):
        """Verifies that RAGService dispatches HTTP request to OmniRoute with configured base URL and bearer token."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="OmniRoute synthesized grounded answer [CHUNK:42].",
            model="gemini-fallback-combo",
        )

        provider = OmniRouteLLMProvider(
            base_url="http://custom-gw:20128/v1",
            api_key="secret-omniroute-token-1234",
            model_name="gemini-fallback-combo",
            client=mock_client,
        )

        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=42,
            document_id="00000000-0000-0000-0000-000000000001",
            content="High-dimensional neural features outperform manual features.",
            page_number=5,
            document_title="ML for IDS",
            similarity_score=0.91,
            char_start=0,
            char_end=60,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        response = rag_service.answer_query(
            db=MagicMock(),
            query="how does deep learning compare to manual features?",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        assert response.provider_name == "omniroute"
        assert response.model_name == "gemini-fallback-combo"
        assert "Mock synthesized response" not in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 42

        # Verify exact endpoint and authorization header sent to client.post
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://custom-gw:20128/v1/chat/completions"
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret-omniroute-token-1234"
        payload = call_args[1]["json"]
        assert payload["model"] == "gemini-fallback-combo"

    def test_rag_service_gateway_error_fails_without_falling_back_to_mock(self):
        """Verifies that an upstream gateway connection error raises LLMProviderError and never returns mock output."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused by gateway")

        provider = OmniRouteLLMProvider(
            base_url="http://localhost:20128/v1",
            model_name="gemini-fallback-combo",
            max_retries=0,
            client=mock_client,
        )

        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=1,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Sample passage",
            page_number=1,
            document_title="Sample Title",
            similarity_score=0.9,
            char_start=0,
            char_end=14,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        with pytest.raises(LLMProviderError) as exc_info:
            rag_service.answer_query(
                db=MagicMock(),
                query="test query",
                user_id="00000000-0000-0000-0000-000000000099",
                scope_type=ChatScopeType.GLOBAL,
            )

        assert "OmniRoute gateway is unreachable" in str(exc_info.value)

    def test_rag_service_preserves_exact_table_values_across_gemini_and_groq(self):
        """Verifies exact 4-decimal table values survive RAG synthesis without warnings and with empty unsupported_numbers."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content=(
                "According to Table 6, Naive Bayes without SMOTE achieved a Precision of 0.6364, "
                "Recall of 0.9742, and FAR of 0.1820 [CHUNK:10]."
            ),
            model="gemini-fallback-combo",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=10,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Table 6: CICIDS2017 without SMOTE. Naive Bayes achieved Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
            page_number=7,
            document_title="NIDS Paper",
            similarity_score=0.95,
            char_start=0,
            char_end=100,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        response = rag_service.answer_query(
            db=MagicMock(),
            query="What are the Precision, Recall, and FAR of Naive Bayes without SMOTE?",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        assert response.unsupported_numbers == []
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 10
        assert "0.6364" in response.answer
        assert "0.9742" in response.answer
        assert "0.1820" in response.answer

    def test_rag_service_flags_rounded_approximations_in_validation_result(self):
        """Verifies rounded approximations (e.g. 0.64 instead of 0.6364) are isolated in unsupported_numbers without failing response."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content=(
                "According to Table 6, Naive Bayes without SMOTE achieved Precision 0.64, "
                "Recall 0.97, and FAR 0.18 [CHUNK:10]."
            ),
            model="gemini-fallback-combo",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        mock_retrieval = MagicMock()
        retrieved_chunk = RetrievedContextChunk(
            chunk_id=10,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Table 6: CICIDS2017 without SMOTE. Naive Bayes achieved Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
            page_number=7,
            document_title="NIDS Paper",
            similarity_score=0.95,
            char_start=0,
            char_end=100,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [retrieved_chunk]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        response = rag_service.answer_query(
            db=MagicMock(),
            query="What are the Precision, Recall, and FAR of Naive Bayes without SMOTE?",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        # Non-destructive audit flags rounded approximations that were not present in source chunk
        assert response.unsupported_numbers == ["0.64", "0.97", "0.18"]
        # Response content and citations remain intact
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 10
        assert "0.64" in response.answer

    def test_rag_distinguishes_smote_conditions(self):
        """Verifies multiple experimental condition chunks (with vs without SMOTE) are delivered distinctly in prompt."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(
            content="Without SMOTE, Naive Bayes achieved Precision 0.6364 and Recall 0.9742 [CHUNK:10].",
            model="gemini-fallback-combo",
        )

        provider = OmniRouteLLMProvider(client=mock_client)
        mock_retrieval = MagicMock()
        chunk_no_smote = RetrievedContextChunk(
            chunk_id=10,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Table 6: CICIDS2017 without SMOTE. Naive Bayes: Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
            page_number=7,
            document_title="NIDS Paper",
            similarity_score=0.95,
            char_start=0,
            char_end=100,
        )
        chunk_with_smote = RetrievedContextChunk(
            chunk_id=11,
            document_id="00000000-0000-0000-0000-000000000001",
            content="Table 7: CICIDS2017 with SMOTE. Naive Bayes: Precision: 0.7241, Recall: 0.8850, FAR: 0.0910.",
            page_number=8,
            document_title="NIDS Paper",
            similarity_score=0.88,
            char_start=0,
            char_end=100,
        )
        mock_retrieval.retrieve_relevant_chunks.return_value = [chunk_no_smote, chunk_with_smote]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        response = rag_service.answer_query(
            db=MagicMock(),
            query="What is the Naive Bayes Precision without SMOTE?",
            user_id="00000000-0000-0000-0000-000000000099",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        assert response.citations[0].chunk_id == 10
        assert response.unsupported_numbers == []

        # Verify outgoing prompt delivered both condition chunks clearly
        payload = mock_client.post.call_args.kwargs["json"]
        user_prompt = payload["messages"][1]["content"]
        assert "[CHUNK:10]" in user_prompt
        assert "Table 6: CICIDS2017 without SMOTE" in user_prompt
        assert "[CHUNK:11]" in user_prompt
        assert "Table 7: CICIDS2017 with SMOTE" in user_prompt


class TestOmniRouteAdaptiveMarkdown:
    """Verifies adaptive Markdown instructions, prompt directives, and end-to-end table/citation preservation."""

    def test_system_instruction_contains_adaptive_markdown_rules(self):
        """Verifies system instructions mandate adaptive Markdown, direct answers, and outside-code citation tags."""
        instruction = OmniRouteLLMProvider._build_system_instruction()
        assert "ADAPTIVE MARKDOWN FORMATTING & PROFESSIONAL STRUCTURE" in instruction
        assert "Meaningful Markdown section headings (## or ###)" in instruction
        assert "Markdown tables for numerical comparisons" in instruction
        assert "outside of code blocks or inline code spans (never write `[CHUNK:1]`)" in instruction
        assert "Begin directly with the answer or core finding" in instruction
        assert "Avoid rigid template headings, decorative emojis" in instruction

    def test_user_prompt_contains_adaptive_markdown_directives(self):
        """Verifies prompt delivered to OmniRoute guides model toward adaptive Markdown with scannable tables."""
        mock_client = MagicMock()
        mock_client.post.return_value = _create_mock_response(content="Formatted text [CHUNK:101].")
        provider = OmniRouteLLMProvider(client=mock_client)

        chunks = [
            LLMChunkContext(
                chunk_id=101,
                content="Precision is 0.9542 and Recall is 0.9810.",
                page_number=1,
                document_title="Paper",
            )
        ]
        provider.generate(LLMRequest(prompt="Compare metrics", context_chunks=chunks))

        payload = mock_client.post.call_args.kwargs["json"]
        user_prompt = payload["messages"][1]["content"]

        assert "Use adaptive Markdown (headings, bullets, and tables for numerical comparisons)" in user_prompt
        assert "exact [CHUNK:<id>] tags in regular text (never inside backticks/code spans)" in user_prompt

    def test_rag_service_preserves_markdown_structure_and_validates_citations(self):
        """Verifies full Markdown answer with headings, table, bold terms, and [CHUNK:id] is preserved and validated."""
        mock_client = MagicMock()
        structured_markdown_answer = (
            "## Key Finding\n"
            "Naïve Bayes has high recall but lower precision on CICIDS2017 without SMOTE [CHUNK:10].\n\n"
            "## Reported Results\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| Precision | 0.6364 |\n"
            "| Recall | 0.9742 |\n"
            "| FAR | 0.1820 |\n\n"
            "## Interpretation\n"
            "The model achieves high Recall of 0.9742 but triggers a notable FAR of 0.1820 on benign traffic [CHUNK:10]."
        )
        mock_client.post.return_value = _create_mock_response(content=structured_markdown_answer)

        provider = OmniRouteLLMProvider(client=mock_client)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_relevant_chunks.return_value = [
            RetrievedContextChunk(
                chunk_id=10,
                document_id="00000000-0000-0000-0000-000000000001",
                content="Table 6: CICIDS2017 without SMOTE. Naive Bayes achieved Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
                page_number=7,
                document_title="NIDS Paper",
                similarity_score=0.95,
                char_start=0,
                char_end=100,
            )
        ]

        rag_service = RAGService(
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            llm_provider=provider,
            retrieval_service=mock_retrieval,
        )

        response = rag_service.answer_query(
            db=MagicMock(),
            query="Analyze Naive Bayes performance without SMOTE.",
            user_id="00000000-0000-0000-0000-000000000001",
            scope_type=ChatScopeType.GLOBAL,
        )

        assert response.status == "SUCCESS"
        # Markdown structure is preserved exactly without flattening
        assert "## Key Finding" in response.answer
        assert "| Precision | 0.6364 |" in response.answer
        assert "| Recall | 0.9742 |" in response.answer
        assert "| FAR | 0.1820 |" in response.answer
        assert "high Recall of 0.9742" in response.answer
        # Citations are properly parsed and snapshot-linked
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == 10
        assert response.citations[0].citation_frequency == 2
        # Exact decimals from evidence are recognized as valid
        assert response.unsupported_numbers == []



