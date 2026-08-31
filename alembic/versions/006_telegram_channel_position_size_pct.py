"""Добавить telegram_channels.position_size_pct — базовый размер позиции
(% от баланса) для сигналов канала. Раньше был захардкожен 5.0 для ВСЕХ
каналов одинаково (main.py: _execute_telegram_signal), хотя доверие к
разным каналам обычно разное.

revision: 006
down_revision: 005
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '006'
down_revision: Optional[str] = '005'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.add_column(
            sa.Column('position_size_pct', sa.Float(), nullable=False, server_default='5.0')
        )


def downgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.drop_column('position_size_pct')
