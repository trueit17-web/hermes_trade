"""Добавить orders.fee_currency — без валюты комиссия в дашборде была
неоднозначной цифрой (спот-комиссия обычно списывается не в quote-валюте
пары, а в полученном активе при покупке или в quote при продаже).

revision: 005
down_revision: 004
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '005'
down_revision: Optional[str] = '004'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.add_column(sa.Column('fee_currency', sa.String(20), nullable=True))


def downgrade():
    with op.batch_alter_table('orders') as batch_op:
        batch_op.drop_column('fee_currency')
