"""Unit tests for SQLAlchemy 2.0 ORM models in ResearchLens AI.

Tests verify:
1. Model construction, default values, and attribute normalization (e.g. email normalization).
2. BIGINT autoincrement ID generation for Chunk model.
3. Database constraints (check constraints, unique constraints, offset rules).
4. Chat scope constraints and safe session archiving upon target entity deletion.
5. Citation Snapshot Pattern: preserving quoted text, page number, and document title
   when chunks are deleted (chunk_id becomes NULL via ON DELETE SET NULL).
6. Cascade deletion behaviors for users, documents, messages, and collections.
7. Multi-tenant ownership boundary validation rules.

Note on Dialect Testing:
These tests run against an in-memory SQLite engine configured with `PRAGMA foreign_keys=ON`.
While SQLite faithfully executes table definitions, foreign key actions, and check constraints,
PostgreSQL-specific production behaviors (e.g. sequence generation under extreme concurrency)
should be verified against a live PostgreSQL staging database.
"""
import uuid
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models import (
    ChatScopeType,
    ChatSession,
    Chunk,
    Collection,
    CollectionDocument,
    Document,
    Message,
    MessageCitation,
    ProcessingStatus,
    SenderType,
    User,
)


@pytest.fixture
def db_engine():
    """Creates a clean in-memory SQLite engine with foreign keys enabled."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Provides an isolated database session rolled back after each test."""
    session_factory = sessionmaker(bind=db_engine, autocommit=False, autoflush=False, expire_on_commit=False)
    session: Session = session_factory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_user(db_session: Session) -> User:
    """Creates and persists an ephemeral test user."""
    user = User(
        email="researcher@example.com",
        hashed_password="secure_argon2_hash",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    db_session.commit()
    return user


class TestUserModel:
    """Tests for User model construction, validation, and defaults."""

    def test_user_creation_and_email_normalization(self, db_session: Session):
        """Verifies email is trimmed, lowercased, and default fields are set."""
        user = User(
            email="  LeadingTrailing@Example.COM  ",
            hashed_password="hashed_pw_value",
        )
        db_session.add(user)
        db_session.commit()

        assert isinstance(user.id, uuid.UUID)
        assert user.email == "leadingtrailing@example.com"
        assert user.is_active is True
        assert user.is_verified is False
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_invalid_email_raises_value_error(self):
        """Verifies invalid or non-email strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid email address"):
            User(email="invalid-no-at-sign", hashed_password="pw")

        with pytest.raises(ValueError, match="non-empty string"):
            User(email="", hashed_password="pw")

    def test_unique_email_constraint(self, db_session: Session, sample_user: User):
        """Verifies duplicate emails violate uniqueness constraint."""
        dup_user = User(
            email="RESEARCHER@example.com",  # Same as sample_user after normalization
            hashed_password="another_hash",
        )
        db_session.add(dup_user)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestCollectionAndDocumentModels:
    """Tests for Collection, Document, and association models."""

    def test_create_document_with_defaults(self, db_session: Session, sample_user: User):
        """Verifies document creation and default processing status."""
        doc = Document(
            owner_id=sample_user.id,
            filename="nature_paper.pdf",
            title="A Breakthrough in Transformer Attention",
            file_path="uploads/nature_paper.pdf",
            file_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            file_size_bytes=1048576,
            page_count=12,
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.processing_status == ProcessingStatus.PENDING.value
        assert doc.error_message is None
        assert doc.created_at is not None

    def test_invalid_processing_status_violates_constraint(self, db_session: Session, sample_user: User):
        """Verifies check constraint rejects unsupported processing status."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            file_path="uploads/paper.pdf",
            file_hash_sha256="abc123hash",
            file_size_bytes=1024,
            processing_status="INVALID_STATUS",
        )
        db_session.add(doc)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_collection_name_uniqueness_per_owner(self, db_session: Session, sample_user: User):
        """Verifies collection names are unique per user, but allowed across different users."""
        c1 = Collection(owner_id=sample_user.id, name="Deep Learning")
        db_session.add(c1)
        db_session.commit()

        # Duplicate for same user fails
        c2 = Collection(owner_id=sample_user.id, name="Deep Learning")
        db_session.add(c2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

        # Same name for a different user succeeds
        other_user = User(email="other@example.com", hashed_password="pw")
        db_session.add(other_user)
        db_session.commit()

        c3 = Collection(owner_id=other_user.id, name="Deep Learning")
        db_session.add(c3)
        db_session.commit()
        assert c3.id is not None

    def test_collection_document_association(self, db_session: Session, sample_user: User):
        """Verifies many-to-many relationship via CollectionDocument."""
        col = Collection(owner_id=sample_user.id, name="NLP Papers")
        doc = Document(
            owner_id=sample_user.id,
            filename="nlp.pdf",
            file_path="uploads/nlp.pdf",
            file_hash_sha256="hash123",
            file_size_bytes=2048,
        )
        db_session.add_all([col, doc])
        db_session.commit()

        assoc = CollectionDocument(collection_id=col.id, document_id=doc.id)
        db_session.add(assoc)
        db_session.commit()

        # Verify relationship navigation
        db_session.refresh(col)
        assert len(col.documents) == 1
        assert col.documents[0].filename == "nlp.pdf"


class TestChunkModelAndAutoincrement:
    """Tests for Chunk model, autoincrement ID generation, and offset constraints."""

    def test_chunk_autoincrement_id_generation(self, db_session: Session, sample_user: User):
        """Verifies Chunk IDs generate sequential positive 64-bit integer IDs for FAISS."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            file_path="uploads/paper.pdf",
            file_hash_sha256="hash456",
            file_size_bytes=1000,
        )
        db_session.add(doc)
        db_session.commit()

        c1 = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Introduction to deep neural networks.",
            char_start=0,
            char_end=38,
            token_count=10,
        )
        c2 = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=1,
            content="Recent advances in attention mechanisms.",
            char_start=39,
            char_end=79,
            token_count=10,
        )
        db_session.add_all([c1, c2])
        db_session.commit()

        assert isinstance(c1.id, int)
        assert isinstance(c2.id, int)
        assert c1.id > 0
        assert c2.id == c1.id + 1
        assert c1.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert c1.char_count == len(c1.content)

    def test_chunk_unique_document_and_index(self, db_session: Session, sample_user: User):
        """Verifies duplicate chunk_index within the same document violates uniqueness."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            file_path="uploads/paper.pdf",
            file_hash_sha256="hash789",
            file_size_bytes=1000,
        )
        db_session.add(doc)
        db_session.commit()

        c1 = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Part 1",
            char_start=0,
            char_end=6,
            token_count=2,
        )
        c2 = Chunk(
            document_id=doc.id,
            chunk_index=0,  # Duplicate index
            page_number=1,
            content="Part 2",
            char_start=7,
            char_end=13,
            token_count=2,
        )
        db_session.add_all([c1, c2])
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_chunk_offset_constraints(self, db_session: Session, sample_user: User):
        """Verifies char_end >= char_start and char_start >= 0 constraints."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            file_path="uploads/paper.pdf",
            file_hash_sha256="hash012",
            file_size_bytes=1000,
        )
        db_session.add(doc)
        db_session.commit()

        # char_end < char_start should fail
        invalid_chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Invalid offsets",
            char_start=50,
            char_end=10,  # Invalid
            token_count=5,
        )
        db_session.add(invalid_chunk)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestChatSessionAndScopeConstraints:
    """Tests for ChatSession scoping, check constraints, and safe archiving."""

    def test_global_chat_valid_and_invalid_targets(self, db_session: Session, sample_user: User):
        """Verifies GLOBAL chat requires target_document_id and target_collection_id to be NULL."""
        valid_global = ChatSession(
            owner_id=sample_user.id,
            title="Global Chat",
            scope_type=ChatScopeType.GLOBAL.value,
        )
        db_session.add(valid_global)
        db_session.commit()
        assert valid_global.id is not None

        # GLOBAL with target_document_id fails
        doc = Document(
            owner_id=sample_user.id,
            filename="doc.pdf",
            file_path="doc.pdf",
            file_hash_sha256="hash111",
            file_size_bytes=500,
        )
        db_session.add(doc)
        db_session.commit()

        invalid_global = ChatSession(
            owner_id=sample_user.id,
            title="Invalid Global",
            scope_type=ChatScopeType.GLOBAL.value,
            target_document_id=doc.id,
        )
        db_session.add(invalid_global)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_scoped_chat_and_archiving(self, db_session: Session, sample_user: User):
        """Verifies DOCUMENT chat requires target_document_id when active, and allows NULL when archived."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            file_path="paper.pdf",
            file_hash_sha256="hash222",
            file_size_bytes=500,
        )
        db_session.add(doc)
        db_session.commit()

        # Active DOCUMENT chat with target is valid
        active_chat = ChatSession(
            owner_id=sample_user.id,
            title="Chat on Paper",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add(active_chat)
        db_session.commit()

        # Active DOCUMENT chat WITHOUT target violates constraint
        invalid_chat = ChatSession(
            owner_id=sample_user.id,
            title="Invalid Document Chat",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=None,
            is_archived=False,
        )
        db_session.add(invalid_chat)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

        # Safe Archiving: When target document is deleted, archiving the session
        # clears target_document_id while keeping scope_type='DOCUMENT'
        active_chat.archive()
        db_session.commit()

        assert active_chat.is_archived is True
        assert active_chat.target_document_id is None
        assert active_chat.scope_type == ChatScopeType.DOCUMENT.value  # NEVER silently converted to GLOBAL!

        # Deleting the document proceeds safely
        db_session.delete(doc)
        db_session.commit()

    def test_collection_scoped_chat_and_archiving(self, db_session: Session, sample_user: User):
        """Verifies COLLECTION chat requires target_collection_id when active, and allows NULL when archived."""
        col = Collection(owner_id=sample_user.id, name="Vision Models")
        db_session.add(col)
        db_session.commit()

        # Active COLLECTION chat with target is valid
        active_chat = ChatSession(
            owner_id=sample_user.id,
            title="Chat on Vision",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        db_session.add(active_chat)
        db_session.commit()

        # COLLECTION chat with document target violates constraint
        invalid_col_doc = ChatSession(
            owner_id=sample_user.id,
            title="Invalid Col Doc Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            target_document_id=uuid.uuid4(),
        )
        db_session.add(invalid_col_doc)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

        # Active COLLECTION chat WITHOUT target violates constraint
        invalid_chat = ChatSession(
            owner_id=sample_user.id,
            title="Invalid Empty Collection Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=None,
            is_archived=False,
        )
        db_session.add(invalid_chat)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

        # Archiving collection chat safely clears target_collection_id
        active_chat.archive()
        db_session.commit()

        assert active_chat.is_archived is True
        assert active_chat.target_collection_id is None
        assert active_chat.scope_type == ChatScopeType.COLLECTION.value  # Never silently becomes GLOBAL

    def test_deleting_active_document_without_archiving_fails_constraint(
        self, db_session: Session, sample_user: User
    ):
        """Verifies deleting a document while an active scoped chat references it is blocked by DB check constraint.
        
        Design Rule:
        Deleting an active scoped target without the required archive workflow is unsupported.
        The application service must explicitly archive the session before deleting the document.
        """
        doc = Document(
            owner_id=sample_user.id,
            filename="active_target.pdf",
            file_path="active_target.pdf",
            file_hash_sha256="hash_active",
            file_size_bytes=1000,
        )
        db_session.add(doc)
        db_session.commit()

        active_chat = ChatSession(
            owner_id=sample_user.id,
            title="Active Chat",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add(active_chat)
        db_session.commit()

        # Attempting to delete document directly without archiving chat violates check constraint
        db_session.delete(doc)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_deleting_active_collection_without_archiving_fails_constraint(
        self, db_session: Session, sample_user: User
    ):
        """Verifies deleting a collection while an active scoped chat references it is blocked by DB check constraint."""
        col = Collection(owner_id=sample_user.id, name="Active Collection")
        db_session.add(col)
        db_session.commit()

        active_chat = ChatSession(
            owner_id=sample_user.id,
            title="Active Collection Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        db_session.add(active_chat)
        db_session.commit()

        # Attempting to delete collection directly without archiving chat violates check constraint
        db_session.delete(col)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestCitationSnapshotPattern:
    """Tests verifying citation snapshots are preserved when referenced chunks are deleted."""

    def test_citation_preserved_when_chunk_deleted(self, db_session: Session, sample_user: User):
        """Verifies ON DELETE SET NULL on chunk_id while preserving quoted_text, page, and doc_title."""
        # 1. Setup doc, chunk, chat session, message
        doc = Document(
            owner_id=sample_user.id,
            filename="attention.pdf",
            title="Attention Is All You Need",
            file_path="uploads/attention.pdf",
            file_hash_sha256="sha_attention",
            file_size_bytes=2048,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=3,
            content="An attention function can be described as mapping a query and a set of key-value pairs to an output.",
            char_start=100,
            char_end=202,
            token_count=24,
        )
        chat = ChatSession(
            owner_id=sample_user.id,
            title="Transformer Q&A",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
        )
        db_session.add_all([chunk, chat])
        db_session.commit()

        message = Message(
            chat_session_id=chat.id,
            sender_type=SenderType.ASSISTANT.value,
            content="The authors define attention as mapping queries and key-values to outputs.",
        )
        db_session.add(message)
        db_session.commit()

        # 2. Add citation snapshot referencing chunk
        citation = MessageCitation(
            message_id=message.id,
            chunk_id=chunk.id,
            quoted_text="An attention function can be described as mapping a query and a set of key-value pairs to an output.",
            page_number=3,
            document_title="Attention Is All You Need",
            similarity_score=0.92,
        )
        db_session.add(citation)
        db_session.commit()
        citation_id = citation.id

        # 3. Delete the chunk directly
        db_session.delete(chunk)
        db_session.commit()

        # 4. Verify citation row still exists with preserved snapshot evidence
        retrieved_citation = db_session.get(MessageCitation, citation_id)
        assert retrieved_citation is not None
        assert retrieved_citation.chunk_id is None  # Foreign key set to NULL
        assert retrieved_citation.quoted_text == "An attention function can be described as mapping a query and a set of key-value pairs to an output."
        assert retrieved_citation.page_number == 3
        assert retrieved_citation.document_title == "Attention Is All You Need"
        assert retrieved_citation.similarity_score == pytest.approx(0.92)

    def test_citation_preserved_when_parent_document_deleted(self, db_session: Session, sample_user: User):
        """Verifies deleting the document cascade-deletes the chunk, but SET NULL preserves citation snapshot."""
        doc = Document(
            owner_id=sample_user.id,
            filename="paper.pdf",
            title="Residual Learning",
            file_path="uploads/paper.pdf",
            file_hash_sha256="sha_resnet",
            file_size_bytes=4096,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Deep residual networks are easier to optimize.",
            char_start=0,
            char_end=46,
            token_count=8,
        )
        chat = ChatSession(owner_id=sample_user.id, title="ResNet Chat")
        db_session.add_all([chunk, chat])
        db_session.commit()

        msg = Message(
            chat_session_id=chat.id,
            sender_type=SenderType.ASSISTANT.value,
            content="ResNets use skip connections.",
        )
        db_session.add(msg)
        db_session.commit()

        citation = MessageCitation(
            message_id=msg.id,
            chunk_id=chunk.id,
            quoted_text="Deep residual networks are easier to optimize.",
            page_number=1,
            document_title="Residual Learning",
            similarity_score=0.88,
        )
        db_session.add(citation)
        db_session.commit()
        cit_id = citation.id

        # Delete the entire Document (which cascades to Chunks)
        db_session.delete(doc)
        db_session.commit()
        db_session.expire_all()

        # Verify citation snapshot survives
        retrieved = db_session.get(MessageCitation, cit_id)
        assert retrieved is not None
        assert retrieved.chunk_id is None
        assert retrieved.quoted_text == "Deep residual networks are easier to optimize."
        assert retrieved.document_title == "Residual Learning"

    def test_citation_preserved_when_orm_relationships_are_preloaded(
        self, db_session: Session, sample_user: User
    ):
        """Verifies snapshot preservation when relationships (doc.chunks, chunk.citations) are loaded in memory."""
        doc = Document(
            owner_id=sample_user.id,
            filename="preloaded.pdf",
            title="Preloaded Relationships Test",
            file_path="uploads/preloaded.pdf",
            file_hash_sha256="sha_preloaded",
            file_size_bytes=2048,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Preloaded memory test passage.",
            char_start=0,
            char_end=30,
            token_count=5,
        )
        chat = ChatSession(owner_id=sample_user.id, title="Preload Chat")
        db_session.add_all([chunk, chat])
        db_session.commit()

        msg = Message(
            chat_session_id=chat.id,
            sender_type=SenderType.ASSISTANT.value,
            content="Memory answer",
        )
        db_session.add(msg)
        db_session.commit()

        citation = MessageCitation(
            message_id=msg.id,
            chunk_id=chunk.id,
            quoted_text="Preloaded memory test passage.",
            page_number=1,
            document_title="Preloaded Relationships Test",
            similarity_score=0.95,
        )
        db_session.add(citation)
        db_session.commit()
        cit_id = citation.id

        # Explicitly touch and populate ORM collections in memory
        assert len(doc.chunks) == 1
        assert len(chunk.citations) == 1
        assert len(msg.citations) == 1

        # Delete the document with relationships actively loaded in the identity map
        db_session.delete(doc)
        db_session.commit()
        db_session.expire_all()

        saved_cit = db_session.get(MessageCitation, cit_id)
        assert saved_cit is not None
        assert saved_cit.chunk_id is None
        assert saved_cit.quoted_text == "Preloaded memory test passage."
        assert saved_cit.document_title == "Preloaded Relationships Test"


class TestCascadeDeletions:
    """Tests verifying cascade deletions on users and messages."""

    def test_user_deletion_cascades_to_documents_and_chunks(self, db_session: Session, sample_user: User):
        """Verifies deleting a User cascades to all their Documents and Chunks."""
        doc = Document(
            owner_id=sample_user.id,
            filename="user_doc.pdf",
            file_path="uploads/user_doc.pdf",
            file_hash_sha256="user_doc_hash",
            file_size_bytes=1024,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="User passage.",
            char_start=0,
            char_end=13,
            token_count=3,
        )
        db_session.add(chunk)
        db_session.commit()
        doc_id = doc.id
        chunk_id = chunk.id

        db_session.delete(sample_user)
        db_session.commit()
        db_session.expire_all()

        assert db_session.get(Document, doc_id) is None
        assert db_session.get(Chunk, chunk_id) is None

    def test_message_deletion_cascades_to_citations(self, db_session: Session, sample_user: User):
        """Verifies deleting a Message cascades to its MessageCitations."""
        chat = ChatSession(owner_id=sample_user.id, title="Test Chat")
        db_session.add(chat)
        db_session.commit()

        msg = Message(
            chat_session_id=chat.id,
            sender_type=SenderType.ASSISTANT.value,
            content="Answer text",
        )
        db_session.add(msg)
        db_session.commit()

        cit = MessageCitation(
            message_id=msg.id,
            chunk_id=None,
            quoted_text="Source excerpt",
            page_number=1,
            document_title="Sample Title",
        )
        db_session.add(cit)
        db_session.commit()
        cit_id = cit.id

        db_session.delete(msg)
        db_session.commit()
        db_session.expire_all()

        assert db_session.get(MessageCitation, cit_id) is None


class TestMessageAndCitationConstraints:
    """Tests for Message and MessageCitation check constraints."""

    def test_invalid_sender_type_violates_constraint(self, db_session: Session, sample_user: User):
        """Verifies sender_type must be in ('USER', 'ASSISTANT', 'SYSTEM')."""
        chat = ChatSession(owner_id=sample_user.id, title="Chat")
        db_session.add(chat)
        db_session.commit()

        invalid_msg = Message(
            chat_session_id=chat.id,
            sender_type="ROBOT",  # Invalid
            content="Hello",
        )
        db_session.add(invalid_msg)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_citation_page_number_positive(self, db_session: Session, sample_user: User):
        """Verifies page_number >= 1 check constraint."""
        chat = ChatSession(owner_id=sample_user.id, title="Chat")
        db_session.add(chat)
        db_session.commit()

        msg = Message(chat_session_id=chat.id, sender_type=SenderType.USER.value, content="Hi")
        db_session.add(msg)
        db_session.commit()

        invalid_cit = MessageCitation(
            message_id=msg.id,
            quoted_text="Text",
            page_number=0,  # Invalid: must be >= 1
            document_title="Title",
        )
        db_session.add(invalid_cit)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_citation_similarity_score_range(self, db_session: Session, sample_user: User):
        """Verifies similarity_score must be within [-1.0, 1.0]."""
        chat = ChatSession(owner_id=sample_user.id, title="Chat")
        db_session.add(chat)
        db_session.commit()

        msg = Message(chat_session_id=chat.id, sender_type=SenderType.USER.value, content="Hi")
        db_session.add(msg)
        db_session.commit()

        invalid_cit = MessageCitation(
            message_id=msg.id,
            quoted_text="Text",
            page_number=1,
            document_title="Title",
            similarity_score=1.5,  # Invalid: > 1.0
        )
        db_session.add(invalid_cit)
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestMultiTenantOwnershipBoundaries:
    """Tests clarifying that application service layer must enforce cross-owner boundaries."""

    def test_application_service_must_verify_owner_match_for_collections(
        self, db_session: Session, sample_user: User
    ):
        """Demonstrates that application services must guard against cross-owner associations.
        
        The database schema links collection_id and document_id. It is the application service's
        responsibility to ensure both belong to the current authenticated user.
        """
        other_user = User(email="intruder@example.com", hashed_password="pw")
        db_session.add(other_user)
        db_session.commit()

        user_col = Collection(owner_id=sample_user.id, name="My Research")
        other_doc = Document(
            owner_id=other_user.id,
            filename="private.pdf",
            file_path="private.pdf",
            file_hash_sha256="hash_private",
            file_size_bytes=500,
        )
        db_session.add_all([user_col, other_doc])
        db_session.commit()

        # Simulated application service validation function
        def add_document_to_collection(current_user_id: uuid.UUID, col: Collection, doc: Document):
            if col.owner_id != current_user_id or doc.owner_id != current_user_id:
                raise PermissionError("Cannot associate entities across different owner accounts")
            return CollectionDocument(collection_id=col.id, document_id=doc.id)

        # Attempting cross-owner association through service logic is safely blocked
        with pytest.raises(PermissionError, match="Cannot associate entities across different owner accounts"):
            add_document_to_collection(sample_user.id, user_col, other_doc)
