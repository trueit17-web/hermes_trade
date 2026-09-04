"""Добавить telegram_signals.reject_reason — человекочитаемая причина
отклонения сигнала (пороги качества, режим торговли, уже открытая позиция,
шорт на споте, отрицательное матожидание канала и т.п.). Раньше "Итог
сделки" для отклонённых сигналов в дашборде всегда показывал "—" — сама
причина была только в логах, найти её для конкретного отклонённого сигнала
задним числом было невозможно.

revision: 011
down_revision: 010
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '011'
down_revision: Optional[str] = '010'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.add_column(
            sa.Column('reject_reason', sa.String(length=255), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.drop_column('reject_reason')
