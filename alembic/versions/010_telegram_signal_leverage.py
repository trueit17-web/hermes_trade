"""Добавить telegram_signals.parsed_leverage — кредитное плечо, явно
указанное каналом в тексте сигнала (например "Кредитное плечо: х35").
Некоторые фьючерсные каналы задают его per-сигнал, отдельно от глобальной
настройки settings.futures_leverage — см. _execute_telegram_signal/
executor._execute_real_order.

revision: 010
down_revision: 009
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '010'
down_revision: Optional[str] = '009'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.add_column(
            sa.Column('parsed_leverage', sa.DECIMAL(precision=36, scale=18), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.drop_column('parsed_leverage')
