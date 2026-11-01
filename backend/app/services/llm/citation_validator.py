"""Citation validator enforcing grounding against retrieved context passages."""
import re
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from app.schemas.rag import CitationSnapshot, RetrievedContextChunk
from app.services.llm.citation_parser import ParsedCitations


class ValidationResult(BaseModel):
    """Result of validating parsed citation markers against retrieved context passages."""
    citations: List[CitationSnapshot] = Field(
        default_factory=list,
        description="Verified citation snapshots in first-occurrence order with frequency counts",
    )
    unknown_chunk_ids: List[int] = Field(
        default_factory=list,
        description="Chunk IDs cited by the model that were not present in the authorized retrieved context",
    )
    malformed_markers: List[str] = Field(
        default_factory=list,
        description="Malformed marker-like tokens forwarded from the parser",
    )
    total_valid_citations: int = Field(
        default=0,
        description="Total count of valid citations including repeated references",
    )
    total_unique_citations: int = Field(
        default=0,
        description="Count of unique verified chunks cited",
    )
    unsupported_numbers: List[str] = Field(
        default_factory=list,
        description="Decimal numerical tokens in cited text that do not appear in the referenced evidence passages",
    )


class CitationValidator:
    """Validates parsed citation IDs against the exact set of retrieved context passages.

    Grounding & Factual Integrity Note:
    - This validator verifies attribution grounding: confirming that an emitted [CHUNK:<id>]
      marker matches an authorized passage retrieved for that query.
    - It does NOT claim or prove semantic entailment, factual correctness, or that the LLM's
      natural language claims logically follow from the cited text.
    - Hallucinated / unknown IDs (chunk IDs not present in retrieved_context) are safely
      isolated into `unknown_chunk_ids` rather than raising uncaught exceptions or contaminating
      the authoritative citations list.
    - Performs conservative numerical audit when generated_text is provided, isolating
      floating-point decimals not found in cited passages into `unsupported_numbers`.
    """

    @classmethod
    def find_unsupported_numbers(
        cls,
        text: str,
        retrieved_context: Dict[int, RetrievedContextChunk],
        cited_chunk_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """Identifies decimal numerical tokens in the answer that do not appear in cited evidence.

        Conservative Design Rules:
        1. Only inspects floating-point decimals with at least two decimal places (e.g. 0.64, 0.9742).
           Does NOT flag integers (e.g. chunk IDs 10, page numbers 7, years 2017, table numbers 6).
        2. Normalizes numerical strings.
        3. Checks if the exact decimal string exists anywhere in the cited chunks' content.
        4. If not found in any cited chunk, returns the ungrounded decimal string in order of appearance.
        5. Does NOT mutate the text or reject the turn, ensuring stability.
        """
        if not text or not retrieved_context:
            return []

        # Target chunks: those cited, or all retrieved if no specific IDs passed
        if cited_chunk_ids:
            target_chunks = [retrieved_context[cid] for cid in cited_chunk_ids if cid in retrieved_context]
        else:
            target_chunks = list(retrieved_context.values())

        if not target_chunks:
            return []

        combined_evidence = " ".join(c.content for c in target_chunks)

        # Extract all exact decimal number tokens present in the evidence passages
        evidence_decimals = set(re.findall(r"\b\d+\.\d+\b", combined_evidence))

        # Match decimal numbers with at least two decimal places in generated text (e.g. 0.64, 0.6364)
        decimal_matches = re.findall(r"\b\d+\.\d{2,}\b", text)

        unsupported: List[str] = []
        seen: Set[str] = set()

        for num_str in decimal_matches:
            if num_str not in seen:
                seen.add(num_str)
                # Check if this exact decimal appears as a discrete number in the evidence
                if num_str not in evidence_decimals:
                    unsupported.append(num_str)

        return unsupported

    @classmethod
    def validate(
        cls,
        parsed_citations: ParsedCitations,
        retrieved_context: Dict[int, RetrievedContextChunk],
        generated_text: Optional[str] = None,
    ) -> ValidationResult:
        """Validates parsed citations against the authorized retrieved context map.

        Args:
            parsed_citations: ParsedCitations output from CitationParser.
            retrieved_context: Mapping of chunk_id -> RetrievedContextChunk for the query.
            generated_text: Optional raw text from the LLM to inspect for numerical grounding.

        Returns:
            ValidationResult containing verified snapshots, unknown IDs, frequency counts, and unsupported numbers.
        """
        valid_counts: Dict[int, int] = {}
        first_occurrence_order: List[int] = []
        unknown_ids: List[int] = []
        seen_unknown: Set[int] = set()
        total_valid = 0

        for chunk_id in parsed_citations.raw_chunk_ids:
            if chunk_id in retrieved_context:
                total_valid += 1
                if chunk_id not in valid_counts:
                    valid_counts[chunk_id] = 1
                    first_occurrence_order.append(chunk_id)
                else:
                    valid_counts[chunk_id] += 1
            else:
                if chunk_id not in seen_unknown:
                    seen_unknown.add(chunk_id)
                    unknown_ids.append(chunk_id)

        citations_list: List[CitationSnapshot] = [
            CitationSnapshot(
                chunk_id=ctx.chunk_id,
                document_id=ctx.document_id,
                quoted_text=ctx.content,
                page_number=ctx.page_number,
                document_title=ctx.document_title,
                similarity_score=ctx.similarity_score,
                citation_frequency=valid_counts[cid],
            )
            for cid in first_occurrence_order
            for ctx in [retrieved_context[cid]]
        ]

        unsupported_nums: List[str] = []
        if generated_text:
            unsupported_nums = cls.find_unsupported_numbers(
                text=generated_text,
                retrieved_context=retrieved_context,
                cited_chunk_ids=first_occurrence_order,
            )

        return ValidationResult(
            citations=citations_list,
            unknown_chunk_ids=unknown_ids,
            malformed_markers=list(parsed_citations.malformed_markers),
            total_valid_citations=total_valid,
            total_unique_citations=len(citations_list),
            unsupported_numbers=unsupported_nums,
        )
