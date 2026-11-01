"""Unit tests for ChunkingService verifying page-bounded chunking, offsets, overlap, and edge cases."""
import pytest
from app.services.pdf_service import ExtractedPage, PDFExtractionResult
from app.services.chunking_service import Chunk, ChunkingService


@pytest.fixture
def chunking_service() -> ChunkingService:
    """Fixture providing a standard ChunkingService instance (~400 tokens, ~50 overlap)."""
    return ChunkingService(target_tokens=400, overlap_tokens=50, chars_per_token=4.0)


@pytest.fixture
def small_chunking_service() -> ChunkingService:
    """Fixture providing a small-sized ChunkingService for granular test assertions."""
    # target_chars = 100 * 2 = 200 chars; overlap_chars = 20 * 2 = 40 chars
    return ChunkingService(target_tokens=100, overlap_tokens=20, chars_per_token=2.0)


class TestChunkingBasics:
    """Tests for basic chunking functionality and initialization validation."""

    def test_invalid_parameters_raise_value_error(self):
        """Verifies that invalid sizing parameters raise ValueError."""
        with pytest.raises(ValueError, match="target_tokens must be greater than 0"):
            ChunkingService(target_tokens=0)

        with pytest.raises(ValueError, match="overlap_tokens cannot be negative"):
            ChunkingService(target_tokens=100, overlap_tokens=-1)

        with pytest.raises(ValueError, match="overlap_tokens must be strictly less than target_tokens"):
            ChunkingService(target_tokens=100, overlap_tokens=100)

        with pytest.raises(ValueError, match="chars_per_token must be greater than 0"):
            ChunkingService(target_tokens=100, overlap_tokens=20, chars_per_token=0)

    def test_short_page_produces_single_chunk(self, chunking_service: ChunkingService):
        """Verifies that a page with text smaller than target_tokens produces exactly one chunk."""
        short_text = "This is a brief summary of the research methodology that comfortably fits in one chunk."
        page = ExtractedPage(page_number=1, text=short_text)

        chunks = chunking_service.chunk_page(page, start_chunk_index=0, document_id="doc-1")

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "doc-1"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.text == short_text
        assert chunk.start_char == 0
        assert chunk.end_char == len(short_text)
        assert page.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.token_count > 0


class TestPageBoundedAndOffsets:
    """Tests verifying page boundaries, exact character offsets, and empty pages."""

    def test_character_offsets_strictly_match_page_text_slices(
        self, small_chunking_service: ChunkingService
    ):
        """Verifies that for every chunk, page.text[chunk.start_char : chunk.end_char] == chunk.text."""
        paragraph = (
            "Machine learning models require robust evaluation metrics. "
            "Cross-validation provides reliable estimation of generalized performance. "
            "Overfitting occurs when models memorize noise rather than underlying patterns. "
            "Regularization techniques such as weight decay penalize excessive complexity. "
            "Early stopping halts training when validation loss stops improving."
        )
        page = ExtractedPage(page_number=3, text=paragraph)

        chunks = small_chunking_service.chunk_page(page, start_chunk_index=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.page_number == 3
            # Critical guarantee: slice must match chunk text exactly
            assert page.text[chunk.start_char : chunk.end_char] == chunk.text

    def test_page_boundaries_are_never_crossed(self, chunking_service: ChunkingService):
        """Verifies that chunking a multi-page document never combines text across pages."""
        extraction = PDFExtractionResult(
            total_pages=2,
            pages=[
                ExtractedPage(page_number=1, text="Page one text content about neural networks."),
                ExtractedPage(page_number=2, text="Page two text content about reinforcement learning."),
            ],
        )

        chunks = chunking_service.chunk_document(extraction, document_id="doc-abc")

        assert len(chunks) == 2
        # Page 1 chunk
        assert chunks[0].page_number == 1
        assert "Page one" in chunks[0].text
        assert "Page two" not in chunks[0].text

        # Page 2 chunk
        assert chunks[1].page_number == 2
        assert "Page two" in chunks[1].text
        assert "Page one" not in chunks[1].text

    def test_empty_and_whitespace_pages_produce_no_chunks(
        self, chunking_service: ChunkingService
    ):
        """Verifies that blank pages produce zero chunks and do not break sequential chunk indexing."""
        extraction = PDFExtractionResult(
            total_pages=3,
            pages=[
                ExtractedPage(page_number=1, text="Introduction on page one."),
                ExtractedPage(page_number=2, text="    \n\n\t   "),  # whitespace-only
                ExtractedPage(page_number=3, text="Conclusion on page three."),
            ],
        )

        chunks = chunking_service.chunk_document(extraction, document_id="doc-xyz")

        assert len(chunks) == 2
        # First chunk from page 1
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0
        assert "Introduction" in chunks[0].text

        # Second chunk from page 3, with sequential index 1 (no gaps or empty chunks)
        assert chunks[1].page_number == 3
        assert chunks[1].chunk_index == 1
        assert "Conclusion" in chunks[1].text


class TestOverlapAndProgression:
    """Tests verifying overlap between chunks, long pages, and sequential indices."""

    def test_long_page_generates_multiple_chunks_with_overlap(
        self, small_chunking_service: ChunkingService
    ):
        """Verifies that a long page produces multiple chunks with genuine overlapping text."""
        long_page_text = (
            "Section 1: Data Preprocessing.\n"
            "We normalize the input tensors to zero mean and unit variance. "
            "Missing values are imputed using median replacement strategy.\n\n"
            "Section 2: Architecture Details.\n"
            "The model consists of twelve self-attention layers with eight attention heads. "
            "Each feed-forward network contains an intermediate expansion factor of four.\n\n"
            "Section 3: Optimization Protocol.\n"
            "We optimize parameters using the AdamW algorithm with cosine annealing learning rate schedule."
        )
        page = ExtractedPage(page_number=1, text=long_page_text)

        chunks = small_chunking_service.chunk_page(page, start_chunk_index=0)

        assert len(chunks) >= 3

        # Verify sequential chunk indices
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.page_number == 1

        # Verify overlap between consecutive chunks:
        # The start of chunk[i+1] must fall before the end of chunk[i]
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            assert next_chunk.start_char < current_chunk.end_char
            assert next_chunk.start_char > current_chunk.start_char

            # Common substring exists between current and next chunk
            overlap_text = long_page_text[next_chunk.start_char : current_chunk.end_char]
            assert len(overlap_text) > 0
            assert overlap_text in current_chunk.text
            assert overlap_text in next_chunk.text

    def test_sequential_chunk_indices_across_entire_document(
        self, small_chunking_service: ChunkingService
    ):
        """Verifies that chunk_index counts up strictly: 0, 1, 2, ... across all pages."""
        text_a = "Sentence Alpha. " * 30  # Generates multiple chunks
        text_b = "Sentence Beta. " * 30   # Generates multiple chunks

        extraction = PDFExtractionResult(
            total_pages=2,
            pages=[
                ExtractedPage(page_number=1, text=text_a),
                ExtractedPage(page_number=2, text=text_b),
            ],
        )

        chunks = small_chunking_service.chunk_document(extraction, document_id="doc-seq")

        assert len(chunks) >= 4
        expected_indices = list(range(len(chunks)))
        actual_indices = [c.chunk_index for c in chunks]
        assert actual_indices == expected_indices


class TestPathologicalInputsAndEdgeCases:
    """Tests ensuring termination on extreme, pathological, or unstructured inputs."""

    def test_pathological_unbroken_text_without_spaces(
        self, small_chunking_service: ChunkingService
    ):
        """Verifies that text with zero spaces or punctuation terminates without infinite looping."""
        pathological_text = "A" * 1500  # 1500 consecutive characters without spaces or delimiters
        page = ExtractedPage(page_number=1, text=pathological_text)

        chunks = small_chunking_service.chunk_page(page, start_chunk_index=0)

        # Must terminate and produce chunks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0
            # Slices must still match exactly
            assert page.text[chunk.start_char : chunk.end_char] == chunk.text

        # Verify forward progress
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_char > chunks[i].start_char

    def test_single_giant_sentence_exceeding_target_size(
        self, small_chunking_service: ChunkingService
    ):
        """Verifies that a sentence far larger than target size is gracefully split at word boundaries."""
        words = ["hypothesis", "experiment", "measurement", "observation", "validation"]
        # Generate an 800-character sentence with no period until the end
        giant_sentence = " ".join(words * 15) + "."
        page = ExtractedPage(page_number=2, text=giant_sentence)

        chunks = small_chunking_service.chunk_page(page, start_chunk_index=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert page.text[chunk.start_char : chunk.end_char] == chunk.text
            assert chunk.page_number == 2
