"""Колонки simulated_* в historical_signals — второй этап плана (после
миграции 015): реальная метка исхода сигнала (win/loss/break-even/
unresolved), посчитанная src/telegram/signal_outcome_simulation.py
прогоном по историческим свечам биржи той же логики частичных TP и
ступенчатого SL, что и у живого _check_position_exit.

revision: 016
down_revision: 015
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '016'
down_revision: Optional[str] = '015'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    with op.batch_alter_table('historical_signals') as batch_op:
        batch_op.add_column(sa.Column('simulated_outcome', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('simulated_pnl_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('simulated_exit_reason', sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column('simulated_tp_hit_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('simulated_at', sa.DateTime(), nullable=True))


def downgrade():
    with op.batch_alter_table('historical_signals') as batch_op:
        batch_op.drop_column('simulated_at')
        batch_op.drop_column('simulated_tp_hit_count')
        batch_op.drop_column('simulated_exit_reason')
        batch_op.drop_column('simulated_pnl_pct')
        batch_op.drop_column('simulated_outcome')
