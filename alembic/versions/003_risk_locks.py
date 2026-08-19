"""Добавить risk_locks и risk_close_events — Protections (freqtrade-style автопаузы:
Cooldown/StoplossGuard/LosingStreak после плохой серии сделок).

revision: 003
down_revision: 002
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '003'
down_revision: Optional[str] = '002'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    op.create_table(
        'risk_locks',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('scope_key', sa.String(100), nullable=False),
        sa.Column('reason', sa.String(255), nullable=False),
        sa.Column('until', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    op.create_index('ix_risk_locks_scope_until', 'risk_locks', ['scope_key', 'until'])

    op.create_table(
        'risk_close_events',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('scope_key', sa.String(100), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('reason', sa.String(50), nullable=False),
        sa.Column('pnl', sa.DECIMAL, nullable=True),
        sa.Column('closed_at', sa.DateTime, nullable=False),
    )
    op.create_index('ix_risk_close_events_scope_closed', 'risk_close_events', ['scope_key', 'closed_at'])
    op.create_index('ix_risk_close_events_reason_closed', 'risk_close_events', ['reason', 'closed_at'])


def downgrade():
    op.drop_index('ix_risk_close_events_reason_closed', table_name='risk_close_events')
    op.drop_index('ix_risk_close_events_scope_closed', table_name='risk_close_events')
    op.drop_table('risk_close_events')
    op.drop_index('ix_risk_locks_scope_until', table_name='risk_locks')
    op.drop_table('risk_locks')
