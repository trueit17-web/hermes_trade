"""Расширить orders.order_id_exchange с VARCHAR(100) до TEXT.

Реальный инцидент (прод, 2026-09-20): позиция закрылась на бирже вне
цикла бота тремя отдельными частичными сделками — _record_external_close
записывает их ID через запятую в это поле (3 UUID по 36 символов + 2
запятые = 110 символов), что превышало прежний лимит VARCHAR(100) и
роняло всю запись Order/Trade с StringDataRightTruncationError, при этом
позиция уже была удалена из памяти бота — закрытие терялось из истории.

revision: 019
down_revision: 018
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '019'
down_revision: Optional[str] = '018'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.alter_column(
            'order_id_exchange',
            existing_type=sa.String(length=100),
            type_=sa.Text(),
            existing_nullable=True,
        )


def downgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.alter_column(
            'order_id_exchange',
            existing_type=sa.Text(),
            type_=sa.String(length=100),
            existing_nullable=True,
        )
