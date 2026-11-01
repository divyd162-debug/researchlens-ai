"""Unit tests for CitationParser extracting and validating [CHUNK:<id>] markers."""
import pytest
from app.services.llm.citation_parser import CitationParser, ParsedCitations, MAX_BIGINT


class TestCitationParserValidMarkers:
    """Tests verifying extraction of valid positive integer citation markers."""

    def test_single_valid_marker(self):
        """Verifies extracting a single valid marker."""
        text = "This claim is supported by evidence [CHUNK:10]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [10]
        assert result.unique_chunk_ids == [10]
        assert result.malformed_markers == []
        assert result.has_citations is True

    def test_multiple_distinct_markers_in_order(self):
        """Verifies extracting multiple distinct markers in exact appearance order."""
        text = "First claim [CHUNK:101], second claim [CHUNK:42], third claim [CHUNK:999]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [101, 42, 999]
        assert result.unique_chunk_ids == [101, 42, 999]
        assert result.malformed_markers == []
        assert result.has_citations is True

    def test_repeated_markers_and_first_occurrence_order(self):
        """Verifies repeated markers preserve order in raw_ids and deduplicate in unique_ids."""
        text = "Fact A [CHUNK:5] is further discussed in [CHUNK:12]. Later, Fact A is reaffirmed [CHUNK:5]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [5, 12, 5]
        assert result.unique_chunk_ids == [5, 12]
        assert result.malformed_markers == []

    def test_whitespace_variant_markers(self):
        """Verifies whitespace inside the bracket is tolerated."""
        text = "Whitespace tolerance: [CHUNK:  42  ] and [CHUNK:   100]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [42, 100]
        assert result.unique_chunk_ids == [42, 100]
        assert result.malformed_markers == []

    def test_consecutive_repeated_markers(self):
        """Verifies back-to-back markers with no separating text."""
        text = "Combined evidence [CHUNK:1][CHUNK:2][CHUNK:1]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [1, 2, 1]
        assert result.unique_chunk_ids == [1, 2]
        assert result.malformed_markers == []

    def test_case_insensitivity(self):
        """Verifies [chunk:1] and [CHUNK:1] are both parsed."""
        text = "Lower: [chunk:10], Mixed: [Chunk:20], Upper: [CHUNK:30]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [10, 20, 30]


class TestCitationParserMalformedMarkers:
    """Tests verifying robust detection and isolation of malformed marker strings."""

    def test_zero_chunk_id_is_malformed(self):
        """Verifies [CHUNK:0] is treated as malformed because IDs start at 1."""
        text = "Zero ID is not a valid chunk: [CHUNK:0]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert result.unique_chunk_ids == []
        assert result.malformed_markers == ["[CHUNK:0]"]
        assert result.has_citations is False

    def test_negative_chunk_id_is_malformed(self):
        """Verifies negative IDs are treated as malformed."""
        text = "Negative ID: [CHUNK:-1] and [CHUNK:-42]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert result.malformed_markers == ["[CHUNK:-1]", "[CHUNK:-42]"]

    def test_non_integer_is_malformed(self):
        """Verifies alphanumeric or string IDs are treated as malformed."""
        text = "Alpha: [CHUNK:abc], float: [CHUNK:3.14], special: [CHUNK:12#]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert len(result.malformed_markers) == 3

    def test_empty_marker_is_malformed(self):
        """Verifies empty or whitespace-only marker bodies are treated as malformed."""
        text = "Empty: [CHUNK:] and whitespace: [CHUNK:   ]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert len(result.malformed_markers) == 2

    def test_extremely_large_integer_is_malformed(self):
        """Verifies integers exceeding 64-bit signed max (MAX_BIGINT) are treated as malformed."""
        huge_id = str(MAX_BIGINT + 100)
        text = f"Huge ID: [CHUNK:{huge_id}]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert len(result.malformed_markers) == 1
        assert huge_id in result.malformed_markers[0]

    def test_unclosed_marker_is_malformed(self):
        """Verifies unclosed marker tags are captured safely as malformed."""
        text = "Here is [CHUNK:123 without a closing bracket\nNext line."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert len(result.malformed_markers) == 1
        assert "[CHUNK:123" in result.malformed_markers[0]


class TestCitationParserEdgeCases:
    """Tests verifying edge cases with empty text and mixed valid/malformed tokens."""

    def test_empty_or_whitespace_text(self):
        """Verifies empty or None-like strings return empty results without error."""
        assert CitationParser.parse("").has_citations is False
        assert CitationParser.parse("   \n\t  ").has_citations is False
        assert CitationParser.parse(None).has_citations is False

    def test_text_without_any_markers(self):
        """Verifies normal text returns empty citation lists."""
        text = "This is a normal paragraph with [brackets] and (parentheses) but no chunks."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == []
        assert result.unique_chunk_ids == []
        assert result.malformed_markers == []
        assert result.has_citations is False

    def test_mixed_valid_and_malformed_markers(self):
        """Verifies valid markers are extracted while malformed markers are separately isolated."""
        text = "Valid [CHUNK:10], bad [CHUNK:abc], another valid [CHUNK:20], zero [CHUNK:0]."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [10, 20]
        assert result.unique_chunk_ids == [10, 20]
        assert result.malformed_markers == ["[CHUNK:abc]", "[CHUNK:0]"]
        assert result.has_citations is True

    def test_unicode_lenticular_and_fullwidth_brackets(self):
        """Verifies Unicode lenticular 【 】 and fullwidth ［ ］ brackets are safely normalized and parsed."""
        text = "Lenticular: **FAR: 0.1820**【CHUNK:253】【CHUNK:300】 and fullwidth ［CHUNK:101］."
        result = CitationParser.parse(text)
        assert result.raw_chunk_ids == [253, 300, 101]
        assert result.unique_chunk_ids == [253, 300, 101]
        assert result.malformed_markers == []
        assert result.has_citations is True
