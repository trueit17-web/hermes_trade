"""Добавить telegram_channels.market — рынок ("spot"/"futures"), на котором
исполняются сигналы ИМЕННО этого канала. Раньше все каналы делили один
глобальный settings.market_type (тумблер в шапке дашборда), хотя разные
каналы обычно рассчитаны на разный тип торговли.

revision: 009
down_revision: 008
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '009'
down_revision: Optional[str] = '008'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.add_column(
            sa.Column('market', sa.String(10), nullable=False, server_default='spot')
        )


def downgrade():
    with op.batch_alter_table('telegram_channels') as batch_op:
        batch_op.drop_column('market')
