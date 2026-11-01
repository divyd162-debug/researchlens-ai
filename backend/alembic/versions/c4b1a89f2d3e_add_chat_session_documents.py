"""add_chat_session_documents

Revision ID: c4b1a89f2d3e
Revises: ea009db203de
Create Date: 2026-10-25 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4b1a89f2d3e'
down_revision: Union[str, Sequence[str], None] = 'ea009db203de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add chat_session_documents table and update scope constraints."""
    op.create_table(
        'chat_session_documents',
        sa.Column('chat_session_id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['chat_session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('chat_session_id', 'document_id')
    )
    op.create_index(op.f('ix_chat_session_documents_chat_session_id'), 'chat_session_documents', ['chat_session_id'], unique=False)
    op.create_index(op.f('ix_chat_session_documents_document_id'), 'chat_session_documents', ['document_id'], unique=False)

    copy_from = None
    try:
        if context.is_offline_mode():
            from app.models.chat import ChatSession
            copy_from = ChatSession.__table__
    except Exception:
        pass

    with op.batch_alter_table('chat_sessions', copy_from=copy_from) as batch_op:
        batch_op.drop_constraint('ck_chat_session_scope_type', type_='check')
        batch_op.create_check_constraint(
            'ck_chat_session_scope_type',
            "scope_type IN ('GLOBAL', 'COLLECTION', 'DOCUMENT', 'DOCUMENTS')"
        )
        batch_op.drop_constraint('ck_chat_session_scope_targets', type_='check')
        batch_op.create_check_constraint(
            'ck_chat_session_scope_targets',
            "(scope_type = 'GLOBAL' AND target_document_id IS NULL AND target_collection_id IS NULL) OR "
            "(scope_type = 'DOCUMENT' AND target_collection_id IS NULL AND (target_document_id IS NOT NULL OR is_archived)) OR "
            "(scope_type = 'COLLECTION' AND target_document_id IS NULL AND (target_collection_id IS NOT NULL OR is_archived)) OR "
            "(scope_type = 'DOCUMENTS' AND target_document_id IS NULL AND target_collection_id IS NULL)"
        )


def downgrade() -> None:
    """Downgrade schema to remove chat_session_documents table and restore prior scope constraints."""
    copy_from = None
    try:
        if context.is_offline_mode():
            from app.models.chat import ChatSession
            copy_from = ChatSession.__table__
    except Exception:
        pass

    with op.batch_alter_table('chat_sessions', copy_from=copy_from) as batch_op:
        batch_op.drop_constraint('ck_chat_session_scope_targets', type_='check')
        batch_op.create_check_constraint(
            'ck_chat_session_scope_targets',
            "(scope_type = 'GLOBAL' AND target_document_id IS NULL AND target_collection_id IS NULL) OR "
            "(scope_type = 'DOCUMENT' AND target_collection_id IS NULL AND (target_document_id IS NOT NULL OR is_archived)) OR "
            "(scope_type = 'COLLECTION' AND target_document_id IS NULL AND (target_collection_id IS NOT NULL OR is_archived))"
        )
        batch_op.drop_constraint('ck_chat_session_scope_type', type_='check')
        batch_op.create_check_constraint(
            'ck_chat_session_scope_type',
            "scope_type IN ('GLOBAL', 'COLLECTION', 'DOCUMENT')"
        )

    op.drop_index(op.f('ix_chat_session_documents_document_id'), table_name='chat_session_documents')
    op.drop_index(op.f('ix_chat_session_documents_chat_session_id'), table_name='chat_session_documents')
    op.drop_table('chat_session_documents')
