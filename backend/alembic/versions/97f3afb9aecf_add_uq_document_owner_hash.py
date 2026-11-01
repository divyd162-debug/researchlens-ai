"""add_uq_document_owner_hash

Revision ID: 97f3afb9aecf
Revises: 2a3a1d2c48f5
Create Date: 2026-10-17 18:10:10.403748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '97f3afb9aecf'
down_revision: Union[str, Sequence[str], None] = '2a3a1d2c48f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to enforce unique constraint on documents (owner_id, file_hash_sha256)."""
    dialect_name = op.get_context().dialect.name
    if dialect_name == "sqlite":
        op.create_index(
            "uq_document_owner_hash",
            "documents",
            ["owner_id", "file_hash_sha256"],
            unique=True,
        )
    else:
        op.create_unique_constraint(
            "uq_document_owner_hash",
            "documents",
            ["owner_id", "file_hash_sha256"],
        )


def downgrade() -> None:
    """Downgrade schema to drop unique constraint on documents (owner_id, file_hash_sha256)."""
    dialect_name = op.get_context().dialect.name
    if dialect_name == "sqlite":
        op.drop_index("uq_document_owner_hash", table_name="documents")
    else:
        op.drop_constraint("uq_document_owner_hash", "documents", type_="unique")
