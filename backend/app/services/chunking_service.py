"""Page-bounded text chunking service for ResearchLens AI."""
from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.pdf_service import ExtractedPage, PDFExtractionResult


class Chunk(BaseModel):
    """Represents a discrete text passage extracted from a single document page."""
    document_id: Optional[str] = Field(None, description="Optional document identifier supplied by caller")
    page_number: int = Field(..., description="1-based physical page number")
    chunk_index: int = Field(..., description="0-based sequential index across the document")
    text: str = Field(..., description="Plain text content of the chunk")
    start_char: int = Field(..., description="Start character offset within page.text (0-based)")
    end_char: int = Field(..., description="End character offset within page.text (0-based)")
    token_count: int = Field(..., description="Estimated token count using standard heuristic (~4 chars/token)")

    @property
    def char_count(self) -> int:
        """Returns character length of chunk text."""
        return len(self.text)


class ChunkingService:
    """Service responsible for splitting extracted PDF pages into page-bounded, overlapping chunks.

    Design Principles:
    1. Page-Bounded: Chunks NEVER cross page boundaries. Citations in academic literature
       strictly tie to specific physical pages.
    2. Exact Character Offsets: start_char and end_char strictly satisfy
       `page.text[chunk.start_char : chunk.end_char] == chunk.text`.
    3. Natural Delimiter Hierarchy: Prefers splitting at paragraphs (\\n\\n), sentences (. , ? , ! ),
       newlines (\\n), and words (spaces).
    4. Robust Forward Progress: Guarantees overlap advances and prevents infinite loops,
       even on pathological inputs (e.g. single giant words or sentences).
    5. Empty Page Safety: Blank and whitespace-only pages produce zero chunks without breaking
       sequential chunk index ordering.
    """

    # Sizing Rationale for Embedding Model Alignment:
    # SentenceTransformer model (all-MiniLM-L6-v2) has a hard max_seq_length of 256 WordPiece tokens.
    # Text exceeding 256 tokens is silently truncated by the tokenizer at inference time.
    # Targeting ~200 tokens (800 chars) with 30-token overlap guarantees that extracted passages
    # fit safely within the 256 WordPiece limit even with subword expansion and special tokens ([CLS], [SEP]).
    DEFAULT_TARGET_TOKENS = 200
    DEFAULT_OVERLAP_TOKENS = 30
    DEFAULT_CHARS_PER_TOKEN = 4.0
    MAX_MODEL_SEQUENCE_TOKENS = 256

    def __init__(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ):
        """Initializes the chunking service with configurable sizing.

        Args:
            target_tokens: Desired token size per chunk (~200 tokens default to fit safely within
                           the 256 WordPiece limit of all-MiniLM-L6-v2 without silent truncation).
            overlap_tokens: Desired token overlap between consecutive chunks on the same page (~30 default).
            chars_per_token: Approximation factor relating character count to token count.
                             Standard English rule-of-thumb is ~4 characters per token.
        """
        if target_tokens <= 0:
            raise ValueError("target_tokens must be greater than 0")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be strictly less than target_tokens")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than 0")

        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = chars_per_token

        # Compute character limits from token targets
        self.target_chars = int(target_tokens * chars_per_token)
        self.overlap_chars = int(overlap_tokens * chars_per_token)

    @property
    def max_model_tokens(self) -> int:
        """Returns the maximum sequence length (256) accepted by the standard embedding model."""
        return self.MAX_MODEL_SEQUENCE_TOKENS

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count using a standard character-length heuristic.

        Token Counting Approach & Tradeoffs:
        - We approximate tokens using `ceil(len(text) / chars_per_token)` (1 token ≈ 4 chars).
        - This is a fast, lightweight, dependency-free heuristic suitable for chunk sizing.
        - Tradeoff: It does not claim exact model-specific subword token counts (like BPE in
          sentence-transformers or tiktoken in OpenAI), but is computationally trivial and
          avoids loading heavy tokenizer models during initial chunking.
        """
        if not text:
            return 0
        return max(1, round(len(text) / self.chars_per_token))

    def _find_natural_delimiter(self, text: str, search_start: int, search_end: int) -> Optional[int]:
        """Finds the best natural boundary to split text within the search window [search_start, search_end].

        Priority order:
        1. Paragraph boundary ('\\n\\n')
        2. Sentence boundaries ('. ', '.\\n', '? ', '! ')
        3. Line boundary ('\\n')
        4. Word boundary (' ')
        """
        window = text[search_start:search_end]
        if not window:
            return None

        # 1. Paragraph boundary
        p_idx = window.rfind("\n\n")
        if p_idx != -1:
            return search_start + p_idx + 2

        # 2. Sentence boundary
        for punct in [". ", ".\n", "? ", "!\n", "! ", "? "]:
            s_idx = window.rfind(punct)
            if s_idx != -1:
                return search_start + s_idx + len(punct)

        # 3. Line boundary
        l_idx = window.rfind("\n")
        if l_idx != -1:
            return search_start + l_idx + 1

        # 4. Word boundary
        w_idx = window.rfind(" ")
        if w_idx != -1:
            return search_start + w_idx + 1

        return None

    def chunk_page(
        self,
        page: ExtractedPage,
        start_chunk_index: int = 0,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Splits an individual ExtractedPage into chunks.

        Args:
            page: ExtractedPage from PDFService containing page_number and raw text.
            start_chunk_index: Starting sequential chunk index for this page.
            document_id: Optional document identifier to attach to chunks.

        Returns:
            List of Chunk objects derived from this page.
        """
        raw_text = page.text
        if not raw_text or not raw_text.strip():
            # Empty or whitespace-only pages yield no chunks
            return []

        chunks: List[Chunk] = []
        current_chunk_idx = start_chunk_index
        text_len = len(raw_text)

        # Skip any leading whitespace on the page while preserving exact character offsets
        start = 0
        while start < text_len and raw_text[start].isspace():
            start += 1

        if start >= text_len:
            return []

        while start < text_len:
            remaining_len = text_len - start

            if remaining_len <= self.target_chars:
                # Remaining text fits in one chunk
                end = text_len
                # Trim trailing whitespace from the end boundary
                while end > start and raw_text[end - 1].isspace():
                    end -= 1

                chunk_text = raw_text[start:end]
                if chunk_text:
                    chunks.append(
                        Chunk(
                            document_id=document_id,
                            page_number=page.page_number,
                            chunk_index=current_chunk_idx,
                            text=chunk_text,
                            start_char=start,
                            end_char=end,
                            token_count=self.estimate_tokens(chunk_text),
                        )
                    )
                break

            # Find a cut point for the chunk
            max_end = start + self.target_chars
            min_end = start + max(1, self.overlap_chars)

            split_pos = self._find_natural_delimiter(raw_text, min_end, max_end)
            if split_pos is not None:
                end = split_pos
            else:
                # Fallback for pathological text with no delimiters (e.g. single 3000-char word)
                end = max_end

            # Trim trailing whitespace at end of current chunk
            trimmed_end = end
            while trimmed_end > start and raw_text[trimmed_end - 1].isspace():
                trimmed_end -= 1

            chunk_text = raw_text[start:trimmed_end]
            if chunk_text:
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_index=current_chunk_idx,
                        text=chunk_text,
                        start_char=start,
                        end_char=trimmed_end,
                        token_count=self.estimate_tokens(chunk_text),
                    )
                )
                current_chunk_idx += 1

            # Determine next start position with overlap
            candidate_next_start = end - self.overlap_chars

            # Look for a clean word boundary near the overlap point
            boundary_pos = self._find_natural_delimiter(
                raw_text,
                max(start + 1, candidate_next_start - 20),
                min(end, candidate_next_start + 20),
            )
            if boundary_pos is not None and boundary_pos > start:
                next_start = boundary_pos
            else:
                next_start = candidate_next_start

            # Strip leading whitespace for next chunk start
            while next_start < text_len and raw_text[next_start].isspace():
                next_start += 1

            # GUARANTEE FORWARD PROGRESS: next_start must strictly advance beyond start
            if next_start <= start:
                next_start = start + 1
                while next_start < text_len and raw_text[next_start].isspace():
                    next_start += 1

            start = next_start

        return chunks

    def chunk_document(
        self,
        extraction_result: PDFExtractionResult,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Splits an entire document extraction result into page-bounded chunks.

        Args:
            extraction_result: Structured result from PDFService.extract_text().
            document_id: Optional document identifier to attach to all chunks.

        Returns:
            List of all Chunk objects across the document with sequential chunk_index values.
        """
        all_chunks: List[Chunk] = []
        current_chunk_idx = 0

        for page in extraction_result.pages:
            page_chunks = self.chunk_page(
                page=page,
                start_chunk_index=current_chunk_idx,
                document_id=document_id,
            )
            all_chunks.extend(page_chunks)
            current_chunk_idx += len(page_chunks)

        return all_chunks
