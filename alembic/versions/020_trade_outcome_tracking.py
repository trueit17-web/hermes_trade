"""Добавить trade_outcome_tracking — движение цены ПОСЛЕ закрытия сделки,
по запросу пользователя: понять, как нужно было бы выставить вход/SL/TP,
чтобы поймать максимум доступной прибыли. Заполняется периодической
задачей (см. trade_outcome_tracker.py/TradingBot._track_closed_trade_
outcomes), не самим фактом закрытия сделки — таблица может быть пустой
до первого запуска задачи.

revision: 020
down_revision: 019
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '020'
down_revision: Optional[str] = '019'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    op.create_table(
        'trade_outcome_tracking',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('trade_id', sa.Integer(), sa.ForeignKey('trades.id'), nullable=False, unique=True),
        sa.Column('symbol_id', sa.Integer(), sa.ForeignKey('symbols.id'), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.DECIMAL(), nullable=False),
        sa.Column('close_price', sa.DECIMAL(), nullable=False),
        sa.Column('close_time', sa.DateTime(), nullable=False),
        sa.Column('baseline_pnl_pct', sa.Float(), nullable=False),
        sa.Column('best_price', sa.DECIMAL(), nullable=True),
        sa.Column('best_price_at', sa.DateTime(), nullable=True),
        sa.Column('worst_price', sa.DECIMAL(), nullable=True),
        sa.Column('worst_price_at', sa.DateTime(), nullable=True),
        sa.Column('last_price', sa.DECIMAL(), nullable=True),
        sa.Column('last_checked_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='tracking'),
        sa.Column('verdict', sa.String(length=30), nullable=True),
        sa.Column('optimal_pnl_pct', sa.Float(), nullable=True),
        sa.Column('missed_pnl_pct', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index(
        'ix_trade_outcome_tracking_status', 'trade_outcome_tracking', ['status'],
    )


def downgrade():
    op.drop_index('ix_trade_outcome_tracking_status', table_name='trade_outcome_tracking')
    op.drop_table('trade_outcome_tracking')
