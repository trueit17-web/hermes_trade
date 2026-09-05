"""Добавить orders.notification_message_id — id отправленного в Telegram
уведомления об открытии позиции. Нужно, чтобы последующие уведомления по
этой же сделке (частичное/полное закрытие по TP/SL, ручное закрытие)
отправлялись ОТВЕТОМ на исходное сообщение о сигнале (reply_to_message_id),
а не отдельными несвязанными сообщениями — иначе в чате нет способа
понять, к какой именно из множества открытых позиций относится очередное
уведомление о закрытии.

revision: 012
down_revision: 011
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '012'
down_revision: Optional[str] = '011'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.add_column(
            sa.Column('notification_message_id', sa.Integer(), nullable=True)
        )


def downgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.drop_column('notification_message_id')
