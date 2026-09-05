"""Добавить trades.leverage — плечо, реально подтверждённое биржей на
момент закрытия позиции. trades.pnl_pct считается от полной номинальной
стоимости позиции (entry_price*amount), а не от маржи — плечо нужно
дашборду, чтобы дополнительно показать PnL% от маржи (pnl_pct*leverage),
как обычно считает сам канал/трейдер (реальный инцидент: пользователь не
мог понять, почему канал заявляет 21%+ прибыли, а бот показывает 0.66%
— разница именно в базе расчёта процента, номинал vs маржа).

revision: 013
down_revision: 012
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '013'
down_revision: Optional[str] = '012'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('trades') as batch_op:
        batch_op.add_column(
            sa.Column('leverage', sa.Float(), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('trades') as batch_op:
        batch_op.drop_column('leverage')
