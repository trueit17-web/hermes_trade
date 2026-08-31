"""Добавить telegram_signals.parsed_take_profits — реальные цели канала
(ближайшая первая), если их несколько; parsed_tp хранит только финальную.
Без этого поля ручное подтверждение pending-сигнала теряло бы
многоуровневый TP — исходное сообщение к моменту подтверждения уже
недоступно.

revision: 007
down_revision: 006
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '007'
down_revision: Optional[str] = '006'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.add_column(sa.Column('parsed_take_profits', sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table('telegram_signals') as batch_op:
        batch_op.drop_column('parsed_take_profits')
