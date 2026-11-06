"""OmniRoute OpenAI-Compatible LLM Provider adapter for ResearchLens AI."""
import logging
import re
import time
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.schemas.llm import ChatMessageItem, LLMChunkContext, LLMRequest, LLMResponse
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import (
    LLMConfigurationError,
    LLMContextLimitExceededError,
    LLMProviderError,
    LLMQuotaExceededError,
)

logger = logging.getLogger(__name__)

DEFAULT_OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
DEFAULT_OMNIROUTE_MODEL = "gemini-fallback-combo"


def _sanitize_error_message(msg: str, secret: Optional[str] = None) -> str:
    """Sanitizes sensitive tokens, authorization headers, and secrets from error strings.

    Args:
        msg: Raw error message string.
        secret: Optional secret/API key to specifically redact.

    Returns:
        Sanitized message safe for logging and client exposure.
    """
    if not msg:
        return ""

    sanitized = str(msg)
    if secret and str(secret).strip():
        sanitized = sanitized.replace(str(secret).strip(), "[REDACTED]")

    # Redact bearer tokens: Bearer <token> -> Bearer [REDACTED]
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    # Redact key=... query parameters
    sanitized = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", sanitized)
    # Redact x-api-key / api_key headers and params
    sanitized = re.sub(
        r"(?:api_key|x-api-key)[=:]\s*([A-Za-z0-9_\-\.]+)",
        "key=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def _strip_reasoning_tags(content: str) -> str:
    """Removes internal reasoning blocks (<think>, <thought>, <reasoning>) emitted by thinking models.

    Preserves citation markers and response text intact without mutating factual content.

    Args:
        content: Raw generated response content.

    Returns:
        Cleaned response content without thinking tags.
    """
    if not content:
        return ""
    # Remove balanced reasoning blocks: <think>...</think>, <thought>...</thought>, <reasoning>...</reasoning>
    cleaned = re.sub(r"<(think|thought|reasoning)>[\s\S]*?</\1>", "", content, flags=re.IGNORECASE)
    # If unclosed reasoning block remains at the beginning, strip it cleanly
    cleaned = re.sub(r"^\s*<(think|thought|reasoning)>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_citation_brackets(content: str) -> str:
    """Normalizes Unicode lenticular (【 】) and fullwidth (［ ］) bracket variants around citations to ASCII [ ]."""
    if not content:
        return ""
    return (
        content.replace("【", "[")
        .replace("】", "]")
        .replace("［", "[")
        .replace("］", "]")
    )


class OmniRouteLLMProvider(BaseLLMProvider):
    """OmniRoute OpenAI-Compatible LLM Provider adapter for ResearchLens AI.

    Design & Architecture Principles:
    1. Single Production Gateway:
       ResearchLens AI interacts exclusively with OmniRoute's OpenAI-compatible
       `/v1/chat/completions` endpoint. All upstream provider credentials (Google Gemini,
       Groq, OpenAI, Anthropic, etc.), load balancing, rate limiting, and failover combos
       are configured and managed entirely inside OmniRoute.
    2. Strict Grounding & Citation Attribution:
       Injects strict instructions requiring the model to ground claims exclusively in
       provided evidence chunks and cite them with standard `[CHUNK:<id>]` marker syntax.
    3. Synchronous Execution Model:
       FastAPI dispatches synchronous `def` endpoints (`post_message`, `query_rag`) to
       an AnyIO threadpool worker (`ThreadPoolExecutor`). Using a synchronous `httpx.Client`
       runs inside worker threads, avoiding event loop blockage while maintaining 100%
       compatibility with synchronous SQLAlchemy ORM sessions.
    4. Granular Error Translation & Resilient Retries:
       - Transient connection drops (`httpx.ConnectError`, `httpx.RemoteProtocolError`) are
         retried up to `max_retries` (default: 1).
       - HTTP 429 is NOT retried: OmniRoute already executes its internal combo fallback chain.
         A 429 from OmniRoute means upstream capacity is exhausted; retrying immediately would
         amplify latency. Fails fast with `LLMQuotaExceededError`.
       - Timeouts are NOT retried: avoids compounding long generation delays. Fails fast with
         `LLMProviderError`.
       - HTTP 400 Bad Request inspects payload (`error.code` / `error.message`) to distinguish
         context window exhaustion from unknown model configuration errors.
    5. Testable Client Injection:
       Accepts an optional injected `client` (`httpx.Client`) for fast, deterministic unit
       testing without making live network calls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        client: Optional[httpx.Client] = None,
    ):
        """Initializes the OmniRoute provider with gateway endpoint, credentials, and options.

        Args:
            base_url: Base URL for OmniRoute OpenAI-compatible API (e.g. 'http://localhost:20128/v1').
            api_key: Optional OmniRoute gateway bearer API key.
            model_name: Model or Combo identifier (e.g. 'gemini-fallback-combo').
            timeout_seconds: Request timeout in seconds. Defaults to settings.OMNIROUTE_TIMEOUT_SECONDS.
            max_retries: Maximum number of retries for transient transport errors. Defaults to 1.
            client: Optional injected httpx.Client instance for testing.
        """
        if base_url is not None:
            self._base_url = str(base_url).strip().rstrip("/")
        else:
            self._base_url = str(getattr(settings, "OMNIROUTE_BASE_URL", DEFAULT_OMNIROUTE_BASE_URL)).strip().rstrip("/")

        self._api_key = (
            api_key
            if api_key is not None
            else getattr(settings, "OMNIROUTE_API_KEY", None)
        )

        if model_name is not None:
            self._model_name = str(model_name).strip()
        else:
            self._model_name = str(getattr(settings, "OMNIROUTE_MODEL", DEFAULT_OMNIROUTE_MODEL)).strip()
        self._timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "OMNIROUTE_TIMEOUT_SECONDS", 60.0)
        )
        self._max_retries = int(
            max_retries
            if max_retries is not None
            else getattr(settings, "OMNIROUTE_MAX_RETRIES", 1)
        )
        self._client = client

    @property
    def provider_name(self) -> str:
        """Returns canonical provider identifier."""
        return "omniroute"

    @property
    def model_name(self) -> str:
        """Returns the active model or combo identifier."""
        return self._model_name

    @property
    def base_url(self) -> str:
        """Returns the configured OmniRoute base URL."""
        return self._base_url

    @property
    def timeout_seconds(self) -> float:
        """Returns configured request timeout in seconds."""
        return self._timeout_seconds

    @property
    def max_retries(self) -> int:
        """Returns configured maximum retries for transient transport drops."""
        return self._max_retries

    def validate_configuration(self) -> bool:
        """Validates that base URL and model name are present and valid.

        Returns:
            True if configuration is valid.

        Raises:
            LLMConfigurationError: If base URL or model name is empty or invalid.
        """
        if self._client is not None:
            # Injected test client
            if not self._model_name:
                raise LLMConfigurationError("OmniRoute model name cannot be empty.")
            return True

        if not self._base_url or not self._base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError(
                f"OmniRoute base URL must be a valid HTTP or HTTPS URL, got '{self._base_url}'."
            )

        if not self._model_name:
            raise LLMConfigurationError("OmniRoute model name cannot be empty.")

        return True

    @classmethod
    def _build_evidence_context(cls, context_chunks: List[LLMChunkContext]) -> str:
        """Formats retrieved passage chunks into structured evidence blocks for the prompt."""
        passage_blocks: List[str] = []
        for chunk in context_chunks:
            title = chunk.document_title or "Untitled Document"
            passage_blocks.append(
                f"[CHUNK:{chunk.chunk_id}] From \"{title}\" (Page {chunk.page_number}):\n{chunk.content}"
            )
        return "\n\n---\n\n".join(passage_blocks)

    @classmethod
    def _build_system_instruction(cls, custom_instruction: Optional[str] = None) -> str:
        """Constructs grounding directives, numerical precision rules, and citation format rules."""
        base_instruction = (
            "You are ResearchLens AI, an evidence-grounded research assistant.\n"
            "Your task is to answer the user's question using ONLY the provided evidence passages.\n\n"
            "STRICT GROUNDING & CITATION RULES:\n"
            "1. Ground every statement, factual claim, and numerical value exclusively in the provided evidence passages.\n"
            "   Do not assume, extrapolate, or use outside knowledge.\n"
            "2. Immediately attribute each factual statement, metric, or table finding to its supporting passage using the exact marker tag:\n"
            "   [CHUNK:<id>], where <id> is the integer chunk ID.\n"
            "   Example: \"Self-attention computes representations without recurrence [CHUNK:101].\"\n"
            "3. If multiple passages support a claim, include each relevant marker (e.g. \"[CHUNK:101] [CHUNK:102]\").\n"
            "4. Only cite chunk IDs that are explicitly present in the provided evidence. Never invent or hallucinate chunk IDs.\n"
            "5. NUMERICAL & TABLE FIDELITY:\n"
            "   - When reporting numerical values, experimental metrics (such as Precision, Recall, FAR, F1, Accuracy, AUC, p-values, or error rates), or table entries, PRESERVE THE EXACT PRECISION AND DECIMAL PLACES shown in the source text or table.\n"
            "   - Never round, truncate, or substitute approximations (for example, do not replace 0.9742 with 0.97, or 0.6364 with 0.64) unless the user explicitly requests rounding.\n"
            "   - Quote or report the exact numbers as recorded in the evidence.\n"
            "6. EXPERIMENTAL CONDITIONS & TABLE DISAMBIGUATION:\n"
            "   - Carefully match the exact dataset, model variant, and experimental condition requested (e.g., distinguishing results with vs. without data balancing/SMOTE, raw vs. preprocessed data, baseline vs. proposed methods, or primary performance tables vs. ablation/summary tables).\n"
            "   - Prefer the table or passage directly corresponding to the requested experimental condition rather than general summary statistics or conflicting comparisons.\n"
            "7. AMBIGUITY & INCONSISTENCY:\n"
            "   - If the source passages contain ambiguous, conflicting, or multiple differing values for the same metric across different tables or experiments, explicitly describe that discrepancy and identify the respective tables/conditions instead of silently picking, combining, or averaging numbers.\n"
            "8. INSUFFICIENT EVIDENCE:\n"
            "   - If the provided evidence is insufficient to answer the question, clearly state:\n"
            "     \"Based on the provided research context, there is insufficient evidence to answer this question.\"\n"
            "     Do not fabricate an answer.\n"
            "9. ADAPTIVE MARKDOWN FORMATTING & PROFESSIONAL STRUCTURE:\n"
            "   - Format responses using clean, modern, adaptive Markdown tailored to the complexity of the inquiry.\n"
            "   - Begin directly with the answer or core finding without repetitive conversational filler, throat-clearing, or boilerplate intros.\n"
            "   - For simple factual or lookup questions, keep the answer direct, concise, and focused without unnecessary headings.\n"
            "   - For multi-part, analytical, or comparative responses, use clear structural elements:\n"
            "     * Meaningful Markdown section headings (## or ###) that describe the content.\n"
            "     * Short, readable paragraphs.\n"
            "     * Bullet points or numbered steps for lists, causal factors, or procedures.\n"
            "     * Markdown tables for numerical comparisons, metrics across datasets or models, and tabular benchmarks.\n"
            "     * Bold key metrics, terms, or findings sparingly to aid scanning without excessive bolding.\n"
            "     * Inline code (`...`) or fenced code blocks (```...```) for technical syntax, hyperparameters, equations, or code.\n"
            "     * A brief takeaway or implication only when helpful; avoid redundant conclusion summaries.\n"
            "   - Avoid rigid template headings, decorative emojis, and repetitive summaries.\n"
            "   - CITATION MARKER PLACEMENT IN MARKDOWN:\n"
            "     * Keep [CHUNK:<id>] tags as regular text outside of code blocks or inline code spans (never write `[CHUNK:1]`). Use standard ASCII square brackets (never unicode lenticular brackets 【】).\n"
            "     * Attach citation tags directly to the claims, sentences, or table cells they support.\n"
            "10. MULTI-TURN DIALOGUE & GROUNDING ISOLATION:\n"
            "   - When prior conversation turns are provided, use them for dialogue context and resolving pronouns or referents.\n"
            "   - CRITICAL GROUNDING REQUIREMENT: Previous conversation turns (including previous assistant replies) are NOT primary scientific literature evidence.\n"
            "   - NEVER cite previous conversation turns as [CHUNK:<id>]. All citations must point exclusively to the explicit EVIDENCE PASSAGES provided for the current turn."
        )
        if custom_instruction and custom_instruction.strip():
            return f"{base_instruction}\n\nAdditional Guidance:\n{custom_instruction.strip()}"
        return base_instruction

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generates a grounded completion for the provided request via OmniRoute.

        Args:
            request: Validated LLMRequest payload with prompt and retrieved context chunks.

        Returns:
            LLMResponse containing raw completion with [CHUNK:<id>] markers and token metadata.

        Raises:
            LLMConfigurationError: If gateway configuration or authentication fails.
            LLMQuotaExceededError: If OmniRoute upstream rate limits or quotas are exhausted.
            LLMContextLimitExceededError: If context window length is exceeded.
            LLMProviderError: If the upstream gateway fails, times out, or returns an error.
        """
        self.validate_configuration()

        # Handle empty context passages without wasting upstream API calls
        if not request.context_chunks:
            fallback_text = (
                f"No relevant context was provided to answer the prompt: '{request.prompt}'. "
                "No source passages were supplied."
            )
            prompt_tokens = max(1, len(request.prompt) // 4)
            completion_tokens = max(1, len(fallback_text) // 4)
            return LLMResponse(
                content=fallback_text,
                model_name=self._model_name,
                provider_name=self.provider_name,
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        evidence_text = self._build_evidence_context(request.context_chunks)
        user_prompt = (
            f"EVIDENCE PASSAGES:\n---\n{evidence_text}\n---\n\n"
            f"USER QUESTION:\n{request.prompt}\n\n"
            "GROUNDING & PRECISION DIRECTIVES:\n"
            "- Answer using ONLY the evidence passages above.\n"
            "- Start with a direct answer. Use adaptive Markdown (headings, bullets, and tables for numerical comparisons) where appropriate for scannability.\n"
            "- Cite supporting passage chunks using exact [CHUNK:<id>] tags in regular text (never inside backticks/code spans).\n"
            "- Quote exact numerical values and metrics with their full decimal precision as reported in the source table (do not round or approximate).\n"
            "- Ensure reported metrics match the exact dataset, model, and experimental condition (e.g. with vs. without SMOTE) specified in the question.\n\n"
            f"ANSWER:"
        )
        system_instruction = self._build_system_instruction(request.system_instruction)

        # Standard OpenAI-compatible chat completions payload
        messages = [{"role": "system", "content": system_instruction}]

        # Inject bounded prior dialogue history turns
        if request.conversation_history:
            for turn in request.conversation_history:
                turn_role = turn.role.lower()
                if turn_role in ("user", "assistant"):
                    messages.append({"role": turn_role, "content": turn.content})

        messages.append({"role": "user", "content": user_prompt})
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None and request.max_tokens > 0:
            payload["max_tokens"] = request.max_tokens

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "ResearchLensAI/1.0",
        }
        if self._api_key and str(self._api_key).strip():
            headers["Authorization"] = f"Bearer {str(self._api_key).strip()}"

        endpoint_url = f"{self._base_url}/chat/completions"

        # Execute HTTP request with bounded transient-only retries
        t_start = time.monotonic()
        response: Optional[httpx.Response] = None
        last_transient_exc: Optional[Exception] = None

        attempts = 1 + max(0, self._max_retries)
        for attempt in range(1, attempts + 1):
            try:
                if self._client is not None:
                    response = self._client.post(
                        endpoint_url,
                        json=payload,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self._timeout_seconds) as client:
                        response = client.post(
                            endpoint_url,
                            json=payload,
                            headers=headers,
                        )
                # Successful response received (could still be error status code)
                break
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_transient_exc = exc
                if attempt < attempts:
                    logger.warning(
                        "Transient connection failure contacting OmniRoute (attempt %d/%d): %s. Retrying...",
                        attempt,
                        attempts,
                        _sanitize_error_message(str(exc), self._api_key),
                    )
                    time.sleep(0.5 * attempt)
                    continue
                # Exhausted retries
                api_duration_ms = (time.monotonic() - t_start) * 1000
                self._handle_transport_exception(exc, api_duration_ms)
            except Exception as exc:
                # Timeouts, request errors, etc. do NOT retry
                api_duration_ms = (time.monotonic() - t_start) * 1000
                self._handle_transport_exception(exc, api_duration_ms)

        if response is None and last_transient_exc:
            api_duration_ms = (time.monotonic() - t_start) * 1000
            self._handle_transport_exception(last_transient_exc, api_duration_ms)

        api_duration_ms = (time.monotonic() - t_start) * 1000

        # Handle non-200 HTTP responses
        if response.status_code != 200:
            self._handle_http_error(response, api_duration_ms)

        # Parse OpenAI-compatible success response
        try:
            data = response.json()
        except Exception as exc:
            logger.error("OmniRoute returned non-JSON response (status 200): %s", response.text[:200])
            raise LLMProviderError("OmniRoute returned an invalid response format.") from exc

        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            logger.error("OmniRoute response missing 'choices': %s", data)
            raise LLMProviderError("OmniRoute returned empty choices.")

        choice = choices[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        finish_reason = str(choice.get("finish_reason") or "stop").lower()

        # Clean reasoning tags if emitted by thinking models and normalize bracket variants
        content = _normalize_citation_brackets(_strip_reasoning_tags(raw_content))

        if not content:
            if "safety" in finish_reason or "content_filter" in finish_reason:
                content = "Response was withheld due to provider content safety filters."
            else:
                content = (
                    "Based on the provided research context, there is insufficient evidence to answer this question."
                )

        # Extract token usage metadata if provided
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None

        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        if prompt_tokens is None:
            prompt_tokens = max(1, len(user_prompt) // 4)
        if completion_tokens is None:
            completion_tokens = max(1, len(content) // 4)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        effective_model = data.get("model") or self._model_name

        logger.info(
            "OmniRoute API request completed in %.2fms [model: %s, finish_reason: %s, prompt_tokens: %s, completion_tokens: %s, total_tokens: %s]",
            api_duration_ms,
            effective_model,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        return LLMResponse(
            content=content,
            model_name=effective_model,
            provider_name=self.provider_name,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def reformulate_query(self, query: str, history: List[ChatMessageItem]) -> str:
        """Reformulates a conversational follow-up query into a standalone retrieval query via OmniRoute.

        Args:
            query: The user's latest follow-up question.
            history: Chronological list of prior conversation turns.

        Returns:
            Standalone search query containing resolved entities and nouns suitable for vector retrieval.
        """
        if not history:
            return query

        self.validate_configuration()

        history_lines = []
        for turn in history:
            role_label = "User" if turn.role.lower() == "user" else "Assistant"
            content_snippet = turn.content[:350].replace("\n", " ").strip()
            if content_snippet:
                history_lines.append(f"{role_label}: {content_snippet}")

        history_text = "\n".join(history_lines)

        reformulation_system = (
            "You are an academic query reformulator. Your task is to rewrite a follow-up question "
            "into a single standalone academic search query suitable for dense vector retrieval.\n"
            "Rules:\n"
            "1. Resolve pronouns (it, its, they, their, them, this, that, etc.) and anaphoric references using the conversation history.\n"
            "2. Preserve all specific domain entities, dataset names, model names, and metrics.\n"
            "3. Return ONLY the rewritten query text. Do NOT include explanations, quotes, preambles, or markdown.\n"
            "4. If the question is already fully standalone, return it verbatim."
        )

        user_content = (
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"FOLLOW-UP QUESTION:\n{query}\n\n"
            f"STANDALONE SEARCH QUERY:"
        )

        messages = [
            {"role": "system", "content": reformulation_system},
            {"role": "user", "content": user_content},
        ]
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 128,
        }
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "ResearchLensAI/1.0",
        }
        if self._api_key and str(self._api_key).strip():
            headers["Authorization"] = f"Bearer {str(self._api_key).strip()}"

        endpoint_url = f"{self._base_url}/chat/completions"
        timeout = min(self._timeout_seconds, 15.0)

        t_start = time.monotonic()
        try:
            if self._client is not None:
                resp = self._client.post(
                    endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(
                        endpoint_url,
                        json=payload,
                        headers=headers,
                    )

            if resp.status_code != 200:
                logger.warning(
                    "OmniRoute query reformulation returned status %d. Falling back to original query.",
                    resp.status_code,
                )
                return query

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return query
            rewritten = choices[0].get("message", {}).get("content", "").strip()
            rewritten = rewritten.strip('"\'`')
            if rewritten:
                reform_duration_ms = (time.monotonic() - t_start) * 1000
                logger.info(
                    "OmniRoute query reformulation succeeded in %.2fms: '%s' -> '%s'",
                    reform_duration_ms,
                    query,
                    rewritten,
                )
                return rewritten
            return query
        except Exception as exc:
            logger.warning(
                "OmniRoute query reformulation failed (%s). Falling back to original query.",
                _sanitize_error_message(str(exc), self._api_key),
            )
            return query

    def _handle_transport_exception(self, exc: Exception, duration_ms: float = 0.0) -> None:
        """Handles low-level socket, connection, and timeout exceptions."""
        raw_msg = str(exc)
        sanitized_msg = _sanitize_error_message(raw_msg, self._api_key)

        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            logger.error(
                "OmniRoute request timed out after %.2fms (configured limit: %.1fs): %s",
                duration_ms,
                self._timeout_seconds,
                sanitized_msg,
            )
            raise LLMProviderError(
                f"OmniRoute request timed out after {self._timeout_seconds:.1f}s. Please try again."
            ) from exc

        if isinstance(exc, httpx.ConnectError):
            logger.error(
                "OmniRoute gateway is unreachable at %s after %.2fms: %s",
                self._base_url,
                duration_ms,
                sanitized_msg,
            )
            raise LLMProviderError(
                f"OmniRoute gateway is unreachable at {self._base_url}. Ensure the OmniRoute service is running."
            ) from exc

        logger.error("OmniRoute transport error after %.2fms: %s", duration_ms, sanitized_msg)
        raise LLMProviderError(f"OmniRoute communication error: {sanitized_msg}") from exc

    def _handle_http_error(self, response: httpx.Response, duration_ms: float = 0.0) -> None:
        """Translates HTTP error status codes and JSON error details into typed LLMErrors."""
        status_code = response.status_code
        raw_text = response.text
        sanitized_body = _sanitize_error_message(raw_text, self._api_key)

        # Attempt to parse OpenAI-compatible error payload
        error_code = ""
        error_message = ""
        try:
            err_data = response.json()
            if isinstance(err_data, dict):
                error_obj = err_data.get("error", err_data)
                if isinstance(error_obj, dict):
                    error_code = str(error_obj.get("code") or "").lower()
                    error_message = str(error_obj.get("message") or "")
        except Exception:
            error_message = sanitized_body

        detail_msg = error_message or sanitized_body
        combined_text = f"{error_code} {detail_msg}".lower()

        logger.error(
            "OmniRoute HTTP %d after %.2fms [code: %s]: %s",
            status_code,
            duration_ms,
            error_code or "none",
            _sanitize_error_message(detail_msg, self._api_key),
        )

        # 401 / 403 Authentication failure
        if status_code in (401, 403):
            raise LLMConfigurationError(
                "OmniRoute authentication failed. Verify OMNIROUTE_API_KEY in your configuration."
            )

        # 429 Rate Limit / Quota Exceeded (OmniRoute already exhausted combo fallbacks)
        if status_code == 429 or "rate_limit" in combined_text or "quota" in combined_text:
            raise LLMQuotaExceededError(
                "OmniRoute upstream providers exhausted rate limits or quota. Please try again later."
            )

        # 400 Bad Request: Dissect cause
        if status_code == 400:
            if (
                error_code == "context_length_exceeded"
                or "maximum context" in combined_text
                or "context window" in combined_text
                or "token count exceeds" in combined_text
            ):
                raise LLMContextLimitExceededError(
                    "The query and retrieved context exceed the model context length."
                )
            if error_code == "model_not_found" or "does not exist" in combined_text or "not found" in combined_text:
                raise LLMConfigurationError(
                    f"OmniRoute model or combo '{self._model_name}' was not found: {detail_msg}"
                )
            raise LLMProviderError(f"OmniRoute rejected the request: {detail_msg}")

        # 504 Gateway Timeout
        if status_code == 504 or "deadline" in combined_text or "timeout" in combined_text:
            raise LLMProviderError(
                f"OmniRoute gateway reported an upstream timeout after {self._timeout_seconds:.1f}s."
            )

        # 500 / 502 / 503 Upstream server or gateway error
        if 500 <= status_code < 600:
            raise LLMProviderError(f"OmniRoute gateway error (HTTP {status_code}): {detail_msg}")

        # Any other unexpected status
        raise LLMProviderError(f"OmniRoute request failed with HTTP {status_code}: {detail_msg}")
