"""Добавить orders.market_type — рынок ("spot"/"futures"), на котором
реально был размещён ордер. settings.market_type — глобальный тумблер, а не
свойство позиции; без этого поля восстановление позиции при рестарте и её
дальнейшее ведение (SL, закрытие) шли через клиент ТЕКУЩЕГО положения
тумблера, а не через тот рынок, на котором позиция была реально открыта.

revision: 008
down_revision: 007
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '008'
down_revision: Optional[str] = '007'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.add_column(
            sa.Column('market_type', sa.String(10), nullable=False, server_default='spot')
        )


def downgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.drop_column('market_type')
