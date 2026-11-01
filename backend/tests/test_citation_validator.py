import uuid
import pytest
from pydantic import ValidationError
from app.schemas.rag import RetrievedContextChunk
from app.services.llm.citation_parser import CitationParser
from app.services.llm.citation_validator import CitationValidator, ValidationResult


class TestCitationValidator:
    """Tests for validating parsed citations against authorized retrieved context."""

    @pytest.fixture
    def doc_id_1(self):
        return uuid.UUID("11111111-1111-1111-1111-111111111111")

    @pytest.fixture
    def doc_id_2(self):
        return uuid.UUID("22222222-2222-2222-2222-222222222222")

    @pytest.fixture
    def retrieved_map(self, doc_id_1, doc_id_2):
        """Provides a sample map of authorized retrieved context chunks."""
        return {
            101: RetrievedContextChunk(
                chunk_id=101,
                document_id=doc_id_1,
                content="Attention mechanism maps query and key-value pairs.",
                page_number=3,
                document_title="Attention Is All You Need",
                similarity_score=0.89,
                char_start=0,
                char_end=51,
            ),
            102: RetrievedContextChunk(
                chunk_id=102,
                document_id=doc_id_2,
                content="Multi-head attention allows attending to different representation subspaces.",
                page_number=4,
                document_title="Attention Is All You Need",
                similarity_score=0.82,
                char_start=0,
                char_end=76,
            ),
        }

    def test_all_citations_valid_and_grounded(self, retrieved_map, doc_id_1, doc_id_2):
        """Verifies valid citations match retrieved chunks and build correct snapshots."""
        text = "Transformers use attention [CHUNK:101] and multi-head attention [CHUNK:102]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)

        assert len(result.citations) == 2
        assert result.unknown_chunk_ids == []
        assert result.malformed_markers == []
        assert result.total_valid_citations == 2
        assert result.total_unique_citations == 2

        # Verify snapshot fields
        c1 = result.citations[0]
        assert c1.chunk_id == 101
        assert c1.document_id == doc_id_1
        assert c1.quoted_text == "Attention mechanism maps query and key-value pairs."
        assert c1.page_number == 3
        assert c1.document_title == "Attention Is All You Need"
        assert c1.similarity_score == pytest.approx(0.89)
        assert c1.citation_frequency == 1

        c2 = result.citations[1]
        assert c2.chunk_id == 102
        assert c2.document_id == doc_id_2
        assert c2.quoted_text == "Multi-head attention allows attending to different representation subspaces."
        assert c2.page_number == 4
        assert c2.citation_frequency == 1

    def test_snapshot_is_frozen_and_cannot_be_mutated(self, retrieved_map):
        """Verifies CitationSnapshot instances are immutable and raise ValidationError on mutation."""
        text = "Grounding [CHUNK:101]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)
        snapshot = result.citations[0]

        with pytest.raises(ValidationError):
            snapshot.citation_frequency = 5

        with pytest.raises(ValidationError):
            snapshot.quoted_text = "Tampered text"

    def test_duplicate_citations_increment_frequency(self, retrieved_map):
        """Verifies repeated citations to the same chunk increment frequency and deduplicate snapshots."""
        text = "Fact A [CHUNK:101]. Detail on Fact A [CHUNK:101]. Conclusion on Fact A [CHUNK:101]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)

        assert len(result.citations) == 1
        assert result.total_valid_citations == 3
        assert result.total_unique_citations == 1

        snapshot = result.citations[0]
        assert snapshot.chunk_id == 101
        assert snapshot.citation_frequency == 3

    def test_unknown_hallucinated_ids_isolated(self, retrieved_map):
        """Verifies citations not present in retrieved context are isolated in unknown_chunk_ids."""
        text = "Valid [CHUNK:101] but also hallucinated [CHUNK:999] and [CHUNK:888]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == 101
        assert set(result.unknown_chunk_ids) == {999, 888}
        assert result.total_valid_citations == 1
        assert result.total_unique_citations == 1

    def test_empty_retrieved_context_all_reported_unknown(self):
        """Verifies when no context was retrieved, any emitted citation is marked unknown."""
        text = "Model claims something [CHUNK:10] and [CHUNK:20]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, {})

        assert len(result.citations) == 0
        assert result.unknown_chunk_ids == [10, 20]
        assert result.total_valid_citations == 0

    def test_empty_parsed_citations(self, retrieved_map):
        """Verifies validation with zero citation tags returns empty result cleanly."""
        text = "Plain answer with no citations."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)

        assert len(result.citations) == 0
        assert len(result.unknown_chunk_ids) == 0
        assert len(result.malformed_markers) == 0

    def test_malformed_markers_forwarded(self, retrieved_map):
        """Verifies malformed markers from parser are preserved in ValidationResult."""
        text = "Valid [CHUNK:101] but bad [CHUNK:abc] and zero [CHUNK:0]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, retrieved_map)

        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == 101
        assert result.malformed_markers == ["[CHUNK:abc]", "[CHUNK:0]"]

    def test_numerical_grounding_exact_numbers_supported(self):
        """Verifies that exact decimal values matching source text are verified with empty unsupported list."""
        doc_id = uuid.uuid4()
        chunk_map = {
            10: RetrievedContextChunk(
                chunk_id=10,
                document_id=doc_id,
                content="Table 6: CICIDS2017 without SMOTE: Naive Bayes reports Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
                page_number=7,
                document_title="IDS Paper",
                similarity_score=0.92,
                char_start=0,
                char_end=103,
            )
        }
        text = "Naive Bayes without SMOTE achieved Precision 0.6364, Recall 0.9742, and FAR 0.1820 [CHUNK:10]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, chunk_map, generated_text=text)

        assert len(result.citations) == 1
        assert result.unsupported_numbers == []

    def test_numerical_grounding_identifies_unsupported_approximations(self):
        """Verifies that rounded approximations not present in the source text are identified in unsupported_numbers."""
        doc_id = uuid.uuid4()
        chunk_map = {
            10: RetrievedContextChunk(
                chunk_id=10,
                document_id=doc_id,
                content="Table 6: CICIDS2017 without SMOTE: Naive Bayes reports Precision: 0.6364, Recall: 0.9742, FAR: 0.1820.",
                page_number=7,
                document_title="IDS Paper",
                similarity_score=0.92,
                char_start=0,
                char_end=103,
            )
        }
        # Model generated rounded approximations 0.64, 0.97, 0.18 instead of exact values
        text = "Naive Bayes without SMOTE had Precision 0.64, Recall 0.97, and FAR 0.18 [CHUNK:10]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, chunk_map, generated_text=text)

        assert len(result.citations) == 1
        # The approximations are isolated in unsupported_numbers
        assert result.unsupported_numbers == ["0.64", "0.97", "0.18"]

    def test_numerical_grounding_ignores_integers_years_and_chunk_ids(self):
        """Verifies integer tokens (e.g. Table 6, Page 7, Year 2017, Chunk 10) are not flagged as decimals."""
        doc_id = uuid.uuid4()
        chunk_map = {
            10: RetrievedContextChunk(
                chunk_id=10,
                document_id=doc_id,
                content="In 2017 dataset Table 6 on Page 7, metric was 0.9545.",
                page_number=7,
                document_title="Paper",
                similarity_score=0.9,
                char_start=0,
                char_end=53,
            )
        }
        text = "According to Table 6 on Page 7 of the 2017 study, score was 0.9545 [CHUNK:10]."
        parsed = CitationParser.parse(text)
        result = CitationValidator.validate(parsed, chunk_map, generated_text=text)

        assert len(result.citations) == 1
        assert result.unsupported_numbers == []
