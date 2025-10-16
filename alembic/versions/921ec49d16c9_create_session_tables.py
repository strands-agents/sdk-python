"""create_session_tables

Revision ID: 921ec49d16c9
Revises: 
Create Date: 2025-10-14 21:20:55.405888

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '921ec49d16c9'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create strands session tables."""
    # Create strands_sessions table
    op.create_table(
        'strands_sessions',
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('session_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('session_id')
    )

    # Create strands_agents table
    op.create_table(
        'strands_agents',
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('state', sa.JSON(), nullable=False),
        sa.Column('conversation_manager_state', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('session_id', 'agent_id'),
        sa.ForeignKeyConstraint(
            ['session_id'],
            ['strands_sessions.session_id'],
            ondelete='CASCADE'
        )
    )

    # Create strands_messages table
    op.create_table(
        'strands_messages',
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=False),
        sa.Column('message', sa.JSON(), nullable=False),
        sa.Column('redact_message', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('session_id', 'agent_id', 'message_id'),
        sa.ForeignKeyConstraint(
            ['session_id', 'agent_id'],
            ['strands_agents.session_id', 'strands_agents.agent_id'],
            ondelete='CASCADE'
        )
    )


def downgrade() -> None:
    """Drop strands session tables."""
    op.drop_table('strands_messages')
    op.drop_table('strands_agents')
    op.drop_table('strands_sessions')
