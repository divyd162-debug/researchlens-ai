"""Retrieval service coordinating dense vector search and authoritative PostgreSQL access filtering."""
from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.chat import ChatScopeType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.schemas.rag import RetrievedContextChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, VectorStoreNotReadyError, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class _QueryConstraints:
    """Explicit structural and experimental constraints parsed from a query."""
    structural_entity: Optional[Tuple[str, str]] = None  # (type, identifier) e.g. ("table", "6")
    negative_conditions: List[str] = field(default_factory=list)  # e.g. ["smote"]
    positive_conditions: List[str] = field(default_factory=list)  # e.g. ["smote"]
    dataset_terms: List[str] = field(default_factory=list)  # e.g. ["cicids2017"]
    model_terms: List[str] = field(default_factory=list)  # e.g. [r"na[iï]ve\s*bayes"]
    metric_terms: List[str] = field(default_factory=list)  # e.g. ["far"]


def _extract_query_constraints(query: str) -> _QueryConstraints:
    """Extracts structural entities, experimental conditions, dataset identifiers, models, and metrics."""
    if not query:
        return _QueryConstraints()

    constraints = _QueryConstraints()

    # 1. Structural entity: Table 6, Figure 2, Appendix B, etc.
    struct_match = re.search(
        r"\b(table|tbl\.?|figure|fig\.?|appendix|app\.?)\s*([0-9]+|[A-Za-z0-9]+)\b",
        query,
        re.IGNORECASE,
    )
    if struct_match:
        raw_type = struct_match.group(1).lower().rstrip(".")
        if raw_type.startswith("tbl"):
            norm_type = "table"
        elif raw_type.startswith("fig"):
            norm_type = "figure"
        elif raw_type.startswith("app"):
            norm_type = "appendix"
        else:
            norm_type = raw_type
        ident = struct_match.group(2).lower()
        constraints.structural_entity = (norm_type, ident)

    # 2. Experimental conditions
    for m in re.finditer(r"\b(without|no|w/o|non-)\s+([A-Za-z0-9_\-]+)\b", query, re.IGNORECASE):
        constraints.negative_conditions.append(m.group(2).lower())

    for m in re.finditer(r"\b(with|w/|using)\s+([A-Za-z0-9_\-]+)\b", query, re.IGNORECASE):
        constraints.positive_conditions.append(m.group(2).lower())

    # 3. Dataset identifiers: uppercase/alphanumeric tokens like CICIDS2017, NSL-KDD
    excluded_acronyms = {
        "FAR", "AUC", "ROC", "TABLE", "TBL", "FIG", "FIGURE", "ACC", "F1", "SMOTE",
        "LSTM", "BERT", "CNN", "RNN", "ML", "AI", "PAGE", "MODEL", "MODELS", "DATASET"
    }
    raw_tokens = re.findall(r"\b([A-Za-z0-9]*[A-Z]+[0-9]+[A-Za-z0-9]*|[A-Z]{2,}(?:-[A-Za-z0-9]+)+)\b", query)
    for tok in raw_tokens:
        if tok.upper() not in excluded_acronyms:
            constraints.dataset_terms.append(tok.lower())

    # 4. Model terms
    model_patterns = [
        r"na[iï]ve\s*bayes", r"logistic\s*regression", r"random\s*forest",
        r"support\s*vector\s*machine", r"\bsvm\b", r"autoencoder",
        r"decision\s*tree", r"xgboost", r"lightgbm", r"\bknn\b",
        r"neural\s*network", r"\blda\b", r"transformer", r"resnet"
    ]
    for pat in model_patterns:
        if re.search(pat, query, re.IGNORECASE):
            constraints.model_terms.append(pat)

    # 5. Metric terms
    metric_terms_list = [
        "far", "false alarm rate", "fpr", "false positive rate", "precision",
        "recall", "f1", "f1-score", "accuracy", "auc", "roc", "sensitivity", "specificity"
    ]
    for met in metric_terms_list:
        if re.search(rf"\b{re.escape(met)}\b", query, re.IGNORECASE):
            constraints.metric_terms.append(met.lower())

    return constraints


def _compute_fused_score(
    content: str,
    base_score: float,
    constraints: _QueryConstraints,
) -> float:
    """Computes constraint-adjusted fused score for candidate passages."""
    if not content:
        return base_score

    score = base_score

    # 1. Structural entity matching & suppression
    if constraints.structural_entity:
        ent_type, ent_id = constraints.structural_entity
        target_pattern = rf"\b{ent_type}s?\s*{re.escape(ent_id)}\b"
        if re.search(target_pattern, content, re.IGNORECASE):
            score += 0.45
        else:
            other_pattern = rf"\b{ent_type}s?\s*(?!{re.escape(ent_id)}\b)[0-9A-Za-z]+"
            if re.search(other_pattern, content, re.IGNORECASE):
                score -= 0.25

    # 2. Dataset terms
    if constraints.dataset_terms:
        for dt in constraints.dataset_terms:
            if re.search(rf"\b{re.escape(dt)}\b", content, re.IGNORECASE):
                score += 0.25
            else:
                score -= 0.15

    # 3. Experimental conditions
    for cond in constraints.negative_conditions:
        has_neg = bool(re.search(rf"\b(without|no|w/o|non-)\s+{re.escape(cond)}\b", content, re.IGNORECASE))
        has_pos = bool(re.search(rf"\b(with|using|w/)\s+{re.escape(cond)}\b", content, re.IGNORECASE))
        if has_neg:
            score += 0.20
        elif has_pos and not has_neg:
            score -= 0.20

    for cond in constraints.positive_conditions:
        has_neg = bool(re.search(rf"\b(without|no|w/o|non-)\s+{re.escape(cond)}\b", content, re.IGNORECASE))
        has_pos = bool(re.search(rf"\b(with|using|w/)\s+{re.escape(cond)}\b", content, re.IGNORECASE))
        if has_pos:
            score += 0.20
        elif has_neg and not has_pos:
            score -= 0.20

    # 4. Model terms
    for mt in constraints.model_terms:
        if re.search(mt, content, re.IGNORECASE):
            score += 0.10

    # 5. Metric terms
    for met in constraints.metric_terms:
        if re.search(rf"\b{re.escape(met)}\b", content, re.IGNORECASE):
            score += 0.10

    return score


class RetrievalError(Exception):
    """Base exception for retrieval service failures."""
    pass


class DocumentNotReadyError(RetrievalError):
    """Raised when a scoped target document exists but is not in READY processing status."""

    def __init__(self, filename: str, processing_status: str, message: Optional[str] = None):
        self.filename = filename
        self.processing_status = processing_status
        super().__init__(
            message
            or f"Target document '{filename}' is currently in status '{processing_status}' and not ready for queries."
        )


class RetrievalService:
    """Service providing multi-tenant vector retrieval with authoritative PostgreSQL verification.

    Core Security & Architectural Invariants:
    1. PostgreSQL is the Canonical Authority:
       FAISS is an unauthenticated, unpartitioned in-memory index. All candidate chunk IDs returned
       by FAISS MUST be joined against PostgreSQL to verify tenant ownership (`document.owner_id == user_id`),
       document readiness (`READY`), and corpus scoping (`GLOBAL`, `DOCUMENT`, `COLLECTION`).
       Chunks belonging to other users, unready documents, or outside the scope are discarded.
    2. Exact Character Offsets:
       `char_start` and `char_end` are populated directly from the persisted `Chunk` record and retain
       their exact semantic definition: zero-based character slice indices into the extracted text of the
       corresponding physical PDF page.
    3. Deterministic Ranking & Tie-Breaking:
       Candidates are filtered by `similarity_threshold` and sorted by descending `similarity_score` with
       FAISS candidate discovery rank as tie-breaker, then sliced to `top_k`.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initializes RetrievalService with optional injected dependencies for testing."""
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is not None:
            return self._embedding_service
        return get_embedding_service()

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is not None:
            return self._vector_store
        return get_vector_store()

    def retrieve_relevant_chunks(
        self,
        db: Session,
        query: str,
        user_id: UUID,
        scope_type: ChatScopeType = ChatScopeType.GLOBAL,
        scope_id: Optional[UUID] = None,
        document_ids: Optional[List[UUID]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[RetrievedContextChunk]:
        """Retrieves and authorizes relevant text chunks for a query under multi-tenant scoping.

        Args:
            db: Active SQLAlchemy database session.
            query: Non-empty search query string.
            user_id: Authenticated user UUID for multi-tenant isolation.
            scope_type: Retrieval scope (GLOBAL, DOCUMENT, COLLECTION, DOCUMENTS).
            scope_id: Target Document ID or Collection ID when scoped.
            document_ids: Target Document IDs when scope_type is DOCUMENTS.
            top_k: Maximum number of passage contexts to retrieve (must be > 0).
            similarity_threshold: Minimum cosine similarity score required for candidate chunks.

        Returns:
            Ranked list of verified `RetrievedContextChunk` objects.

        Raises:
            ValueError: If inputs or scope parameters are invalid, or target scope entity is unowned/not found.
            DocumentNotReadyError: If target scoped document is found but not in READY status.
        """
        # 1. Validate inputs before embedding or database queries
        if not query or not str(query).strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        if not isinstance(similarity_threshold, (int, float)):
            raise ValueError(f"similarity_threshold must be a float, got {type(similarity_threshold).__name__}.")

        cleaned_query = str(query).strip()

        # 2. Validate and normalize user_id
        if user_id is None:
            raise ValueError("user_id cannot be None.")
        if isinstance(user_id, str):
            try:
                user_id = UUID(user_id)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"user_id must be a valid UUID, got {user_id!r}.")
        elif not isinstance(user_id, UUID):
            raise ValueError(f"user_id must be a valid UUID, got {user_id!r}.")

        # 3. Fail-closed scope_type and scope_id validation
        if scope_type is None:
            raise ValueError("scope_type cannot be None. Must be one of ChatScopeType.")
        if isinstance(scope_type, str):
            try:
                scope_type = ChatScopeType(scope_type)
            except ValueError:
                raise ValueError(
                    f"Invalid scope_type: {scope_type!r}. Must be one of {[e.value for e in ChatScopeType]}."
                )
        elif not isinstance(scope_type, ChatScopeType):
            raise ValueError(
                f"Unsupported scope_type type: {type(scope_type).__name__}. Must be ChatScopeType."
            )

        if scope_type == ChatScopeType.GLOBAL:
            if scope_id is not None:
                raise ValueError("scope_id must be None when scope_type is GLOBAL.")
            if document_ids is not None:
                raise ValueError("document_ids must be None when scope_type is GLOBAL.")
        elif scope_type in (ChatScopeType.DOCUMENT, ChatScopeType.COLLECTION):
            if scope_id is None:
                raise ValueError(f"scope_id is required when scope_type is {scope_type.value}.")
            if document_ids is not None:
                raise ValueError(f"document_ids cannot be specified when scope_type is {scope_type.value}.")
            if isinstance(scope_id, str):
                try:
                    scope_id = UUID(scope_id)
                except (ValueError, TypeError, AttributeError):
                    raise ValueError(f"scope_id must be a valid UUID, got {scope_id!r}.")
            elif not isinstance(scope_id, UUID):
                raise ValueError(f"scope_id must be a valid UUID, got {scope_id!r}.")
        elif scope_type == ChatScopeType.DOCUMENTS:
            if scope_id is not None:
                raise ValueError("scope_id must be None when scope_type is DOCUMENTS.")
            if not document_ids:
                raise ValueError("document_ids is required and cannot be empty when scope_type is DOCUMENTS.")
            normalized_doc_ids: List[UUID] = []
            for did in document_ids:
                if isinstance(did, str):
                    try:
                        normalized_doc_ids.append(UUID(did))
                    except (ValueError, TypeError, AttributeError):
                        raise ValueError(f"document_ids must contain valid UUIDs, got {did!r}.")
                elif isinstance(did, UUID):
                    normalized_doc_ids.append(did)
                else:
                    raise ValueError(f"document_ids must contain valid UUIDs, got {did!r}.")
            if len(normalized_doc_ids) > 50:
                raise ValueError("document_ids cannot exceed 50 documents.")
            if len(set(normalized_doc_ids)) != len(normalized_doc_ids):
                raise ValueError("document_ids cannot contain duplicate IDs.")
            document_ids = normalized_doc_ids

        # 4. Scope Verification
        if scope_type == ChatScopeType.DOCUMENT:
            doc = db.scalar(select(Document).where(Document.id == scope_id, Document.owner_id == user_id))
            if not doc:
                raise ValueError(f"Document {scope_id} not found or not owned by user {user_id}.")
            if doc.processing_status != ProcessingStatus.READY.value:
                raise DocumentNotReadyError(
                    filename=doc.filename,
                    processing_status=doc.processing_status,
                )

        elif scope_type == ChatScopeType.COLLECTION:
            col = db.scalar(select(Collection).where(Collection.id == scope_id, Collection.owner_id == user_id))
            if not col:
                raise ValueError(f"Collection {scope_id} not found or not owned by user {user_id}.")

        elif scope_type == ChatScopeType.DOCUMENTS:
            found_docs = db.scalars(
                select(Document).where(Document.id.in_(document_ids), Document.owner_id == user_id)
            ).all()
            if len(found_docs) != len(document_ids):
                raise ValueError(f"One or more documents not found or not owned by user {user_id}.")
            for doc in found_docs:
                if doc.processing_status != ProcessingStatus.READY.value:
                    raise DocumentNotReadyError(
                        filename=doc.filename,
                        processing_status=doc.processing_status,
                    )

        # 5. Extract Query Constraints and Verify Vector Store Readiness
        constraints = _extract_query_constraints(cleaned_query)
        if not self.vector_store.is_ready:
            raise VectorStoreNotReadyError(
                "VectorStore is not ready to serve queries (index reload or recovery in progress)."
            )

        # 6. Authoritative Scope Pre-filtering of Candidate Chunk IDs to Resolve FAISS Candidate Dilution
        allowed_stmt = (
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.owner_id == user_id,
                Document.processing_status == ProcessingStatus.READY.value,
            )
        )
        if scope_type == ChatScopeType.DOCUMENT:
            allowed_stmt = allowed_stmt.where(Document.id == scope_id)
        elif scope_type == ChatScopeType.COLLECTION:
            allowed_stmt = allowed_stmt.join(
                CollectionDocument,
                CollectionDocument.document_id == Document.id,
            ).where(CollectionDocument.collection_id == scope_id)
        elif scope_type == ChatScopeType.DOCUMENTS:
            allowed_stmt = allowed_stmt.where(Document.id.in_(document_ids))
        # Note: ChatScopeType.GLOBAL is inherently bounded to all READY documents owned by user_id

        allowed_chunk_ids: List[int] = [r[0] for r in db.execute(allowed_stmt).all()]
        if not allowed_chunk_ids:
            return []

        # Dense Query Embedding & Scoped FAISS Nearest-Neighbor Search
        query_vector = self.embedding_service.embed_text(cleaned_query)
        oversample_k = max(top_k * 8, 40)

        candidates = self.vector_store.search(
            query_vector,
            top_k=oversample_k,
            allowed_ids=allowed_chunk_ids,
        )

        candidate_scores: Dict[int, float] = {c.chunk_id: c.score for c in candidates} if candidates else {}

        # 7. Lexical Structural Entity Expansion (Guarantees target table/figure candidates are pooled)
        if constraints.structural_entity:
            ent_type, ent_id = constraints.structural_entity
            lex_stmt = (
                select(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Document.owner_id == user_id,
                    Document.processing_status == ProcessingStatus.READY.value,
                    or_(
                        Chunk.content.ilike(f"%{ent_type} {ent_id}%"),
                        Chunk.content.ilike(f"%{ent_type}{ent_id}%"),
                        Chunk.content.ilike(f"%{ent_type}. {ent_id}%"),
                        Chunk.content.ilike(f"%{ent_type}: {ent_id}%"),
                    ),
                )
            )
            if scope_type == ChatScopeType.DOCUMENT:
                lex_stmt = lex_stmt.where(Document.id == scope_id)
            elif scope_type == ChatScopeType.COLLECTION:
                lex_stmt = lex_stmt.join(
                    CollectionDocument,
                    CollectionDocument.document_id == Document.id,
                ).where(CollectionDocument.collection_id == scope_id)
            elif scope_type == ChatScopeType.DOCUMENTS:
                lex_stmt = lex_stmt.where(Document.id.in_(document_ids))

            lex_rows = db.execute(lex_stmt).all()
            default_score = min(candidate_scores.values()) if candidate_scores else 0.40
            for row in lex_rows:
                lcid = row[0]
                if lcid not in candidate_scores:
                    candidate_scores[lcid] = default_score

        if not candidate_scores:
            return []

        candidate_ids = list(candidate_scores.keys())

        # 8. Authoritative PostgreSQL Join & Access Verification
        stmt = (
            select(Chunk, Document.title, Document.filename, Document.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.id.in_(candidate_ids),
                Document.owner_id == user_id,
                Document.processing_status == ProcessingStatus.READY.value,
            )
        )

        if scope_type == ChatScopeType.DOCUMENT:
            stmt = stmt.where(Document.id == scope_id)
        elif scope_type == ChatScopeType.COLLECTION:
            stmt = stmt.join(
                CollectionDocument,
                CollectionDocument.document_id == Document.id,
            ).where(CollectionDocument.collection_id == scope_id)
        elif scope_type == ChatScopeType.DOCUMENTS:
            stmt = stmt.where(Document.id.in_(document_ids))

        rows = db.execute(stmt).all()

        # 9. Compute Fused Scores, Filter, and Rank Top-K
        candidate_rank = {c.chunk_id: idx for idx, c in enumerate(candidates)} if candidates else {}
        candidate_fused_scores: Dict[int, float] = {}
        authorized_contexts: List[RetrievedContextChunk] = []

        for chunk, doc_title, doc_filename, doc_id in rows:
            base_score = candidate_scores.get(chunk.id, 0.0)
            if base_score >= similarity_threshold:
                fused_score = _compute_fused_score(chunk.content, base_score, constraints)
                candidate_fused_scores[chunk.id] = fused_score

                # Prefer explicit document title; fallback to document filename; fallback to placeholder if both missing
                resolved_title = (
                    (doc_title.strip() if doc_title and doc_title.strip() else None)
                    or (doc_filename.strip() if doc_filename and doc_filename.strip() else None)
                    or "Untitled Document"
                )
                authorized_contexts.append(
                    RetrievedContextChunk(
                        chunk_id=chunk.id,
                        document_id=doc_id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        document_title=resolved_title,
                        similarity_score=base_score,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                    )
                )

        # Sort by fused score descending, breaking ties with base similarity score and search rank
        authorized_contexts.sort(
            key=lambda x: (
                candidate_fused_scores.get(x.chunk_id, x.similarity_score),
                x.similarity_score,
                -candidate_rank.get(x.chunk_id, 9999),
            ),
            reverse=True,
        )
        return authorized_contexts[:top_k]
