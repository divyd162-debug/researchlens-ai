"""add_document_lease_token

Revision ID: ea009db203de
Revises: 97f3afb9aecf
Create Date: 2026-10-18 01:03:28.666924

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ea009db203de'
down_revision: Union[str, Sequence[str], None] = '97f3afb9aecf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add lease_token to documents."""
    op.add_column("documents", sa.Column("lease_token", sa.String(length=64), nullable=True))
    op.create_index("ix_documents_lease_token", "documents", ["lease_token"])


def downgrade() -> None:
    """Downgrade schema to remove lease_token from documents."""
    op.drop_index("ix_documents_lease_token", table_name="documents")
    op.drop_column("documents", "lease_token")
