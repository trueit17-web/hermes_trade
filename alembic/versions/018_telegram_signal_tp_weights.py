"""Добавить telegram_signals.parsed_tp_weights и .parsed_post_tp1_sl_rule —
явные правила управления позицией после TP1, которые канал иногда задаёт
свободным текстом сигнала ("Фиксируем 50% на первой цели, 25% на второй
цели и 25% на оставшейся и после первой цели ставим стоп в без убыток"),
см. extract_tp_weights/extract_post_tp1_sl_rule в channel_monitor.py. NULL
в обоих полях — канал ничего не указал, используется дефолт бота (равное
распределение долей и halfway_to_entry_stop_price).

revision: 018
down_revision: 017
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '018'
down_revision: Optional[str] = '017'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.add_column(sa.Column('parsed_tp_weights', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('parsed_post_tp1_sl_rule', sa.String(length=20), nullable=True))


def downgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.drop_column('parsed_post_tp1_sl_rule')
        batch_op.drop_column('parsed_tp_weights')
