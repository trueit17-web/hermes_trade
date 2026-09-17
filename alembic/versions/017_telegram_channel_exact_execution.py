"""Добавить telegram_channels.exact_execution — исключить канал из общих
для всех каналов автоправок сигнала (капы/дефолты SL, лимит плеча,
масштабирование размера по expectancy_sizing, см. _execute_telegram_signal
в main.py). При включении цена входа/SL/TP/плечо исполняются ровно такими,
какими их прислал канал. Портфельные ограничения (лимит числа/суммарной
экспозиции позиций, kill switch, пауза, дневной лимит убытка) не
затрагиваются.

revision: 017
down_revision: 016
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '017'
down_revision: Optional[str] = '016'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.add_column(
            sa.Column('exact_execution', sa.Boolean(), nullable=False, server_default='false')
        )


def downgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.drop_column('exact_execution')
