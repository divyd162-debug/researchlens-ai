"""Unit tests for QueryRequest Pydantic validation schema."""
from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.models.chat import ChatScopeType
from app.schemas.rag import QueryRequest


class TestQueryRequestValidation:
    """Tests schema validation, bounds checks, scope pairing, and extra field rejection for QueryRequest."""

    def test_valid_defaults(self):
        """QueryRequest populates expected defaults when only query is provided."""
        req = QueryRequest(query="What is transformer architecture?")
        assert req.query == "What is transformer architecture?"
        assert req.scope_type == ChatScopeType.GLOBAL
        assert req.scope_id is None
        assert req.top_k == 5
        assert req.similarity_threshold == 0.0

    def test_query_whitespace_trimmed(self):
        """Leading and trailing whitespace is trimmed from the query string."""
        req = QueryRequest(query="   \t  How does backpropagation work?  \n ")
        assert req.query == "How does backpropagation work?"

    def test_empty_query_rejected(self):
        """Empty query string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")
        assert "Query must contain non-whitespace characters." in str(exc_info.value)

    def test_whitespace_only_query_rejected(self):
        """Whitespace-only query string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="    \t\n\r  ")
        assert "Query must contain non-whitespace characters." in str(exc_info.value)

    def test_query_length_boundary_accepted(self):
        """Query of exactly 1,000 characters is accepted."""
        query_1000 = "a" * 1000
        req = QueryRequest(query=query_1000)
        assert len(req.query) == 1000

    def test_query_length_exceeding_1000_rejected(self):
        """Query exceeding 1,000 characters raises ValidationError."""
        query_1001 = "a" * 1001
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query=query_1001)
        assert "Query cannot exceed 1000 characters" in str(exc_info.value) or "String should have at most 1000 characters" in str(exc_info.value)

    def test_query_trimmed_before_length_validation(self):
        """Query with surrounding whitespace that trims to <= 1000 characters is accepted."""
        query_padded = "   " + ("b" * 1000) + "   "
        req = QueryRequest(query=query_padded)
        assert len(req.query) == 1000

    def test_top_k_bounds(self):
        """top_k must be between 1 and 50 inclusive."""
        # Lower bound boundary
        assert QueryRequest(query="test", top_k=1).top_k == 1
        # Upper bound boundary
        assert QueryRequest(query="test", top_k=50).top_k == 50

        # top_k < 1 rejected
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=-5)

        # top_k > 50 rejected
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=51)

    def test_similarity_threshold_bounds(self):
        """similarity_threshold must be between -1.0 and 1.0 inclusive."""
        # Valid boundary values
        assert QueryRequest(query="test", similarity_threshold=-1.0).similarity_threshold == -1.0
        assert QueryRequest(query="test", similarity_threshold=0.0).similarity_threshold == 0.0
        assert QueryRequest(query="test", similarity_threshold=1.0).similarity_threshold == 1.0

        # Out-of-bounds low rejected
        with pytest.raises(ValidationError):
            QueryRequest(query="test", similarity_threshold=-1.01)

        # Out-of-bounds high rejected
        with pytest.raises(ValidationError):
            QueryRequest(query="test", similarity_threshold=1.01)

    def test_global_scope_requires_scope_id_none(self):
        """GLOBAL scope requires scope_id to be None."""
        # Valid GLOBAL
        req = QueryRequest(query="test", scope_type=ChatScopeType.GLOBAL, scope_id=None)
        assert req.scope_type == ChatScopeType.GLOBAL
        assert req.scope_id is None

        # GLOBAL with scope_id rejected
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", scope_type=ChatScopeType.GLOBAL, scope_id=uuid4())
        assert "scope_id must be None when scope_type is GLOBAL" in str(exc_info.value)

    def test_document_scope_requires_scope_id(self):
        """DOCUMENT scope requires non-None scope_id."""
        doc_id = uuid4()
        req = QueryRequest(query="test", scope_type=ChatScopeType.DOCUMENT, scope_id=doc_id)
        assert req.scope_type == ChatScopeType.DOCUMENT
        assert req.scope_id == doc_id

        # Missing scope_id rejected
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", scope_type=ChatScopeType.DOCUMENT, scope_id=None)
        assert "scope_id is required when scope_type is DOCUMENT" in str(exc_info.value)

    def test_collection_scope_requires_scope_id(self):
        """COLLECTION scope requires non-None scope_id."""
        col_id = uuid4()
        req = QueryRequest(query="test", scope_type=ChatScopeType.COLLECTION, scope_id=col_id)
        assert req.scope_type == ChatScopeType.COLLECTION
        assert req.scope_id == col_id

        # Missing scope_id rejected
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", scope_type=ChatScopeType.COLLECTION, scope_id=None)
        assert "scope_id is required when scope_type is COLLECTION" in str(exc_info.value)

    def test_unknown_fields_rejected(self):
        """Extra or unknown fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", unexpected_field="malicious_payload")  # type: ignore[call-arg]
        assert "extra_forbidden" in str(exc_info.value) or "Extra inputs are not permitted" in str(exc_info.value)

    def test_client_supplied_user_id_rejected(self):
        """Supplying user_id in the request payload is strictly rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", user_id=uuid4())  # type: ignore[call-arg]
        assert "extra_forbidden" in str(exc_info.value) or "Extra inputs are not permitted" in str(exc_info.value)

    def test_invalid_scope_type_rejected(self):
        """Invalid scope_type string values are rejected."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", scope_type="INVALID_SCOPE")  # type: ignore[arg-type]
