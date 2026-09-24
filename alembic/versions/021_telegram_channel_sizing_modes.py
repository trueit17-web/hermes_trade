"""Добавить telegram_channels.leverage_mode и risk_per_trade_pct —
подгонка плеча под SL канала вместо урезания SL и размер позиции от риска.

revision: 021
down_revision: 020
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '021'
down_revision: Optional[str] = '020'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.add_column(
            sa.Column('leverage_mode', sa.String(length=20), nullable=False, server_default='cap_sl')
        )
        batch_op.add_column(sa.Column('risk_per_trade_pct', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.drop_column('risk_per_trade_pct')
        batch_op.drop_column('leverage_mode')
