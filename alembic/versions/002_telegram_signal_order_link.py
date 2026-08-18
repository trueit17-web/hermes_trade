"""Добавить telegram_signals.executed_order_id — связь сигнала с открытым по нему ордером
(нужна для статистики по каналам: executed_trade_id проставляется только при закрытии позиции,
а этот столбец позволяет найти открывающий ордер сразу в момент исполнения сигнала).

revision: 002
down_revision: initial
"""

from typing import Optional

from alembic import op
from sqlalchemy import Column, ForeignKey, Integer

revision: str = '002'
down_revision: Optional[str] = 'initial'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.add_column(
            Column(
                'executed_order_id', Integer,
                ForeignKey('orders.id', name='fk_telegram_signals_executed_order_id'),
                nullable=True,
            )
        )


def downgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.drop_column('executed_order_id')
