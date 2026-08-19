"""Добавить risk_close_events.pnl_pct — нужен для expectancy-based sizing
(средний % доходности на сделку по источнику сигнала), Protections сам по
себе использовал только pnl (абсолютный).

revision: 004
down_revision: 003
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '004'
down_revision: Optional[str] = '003'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('risk_close_events') as batch_op:
        batch_op.add_column(sa.Column('pnl_pct', sa.Float, nullable=True, server_default='0'))


def downgrade():
    with op.batch_alter_table('risk_close_events') as batch_op:
        batch_op.drop_column('pnl_pct')
