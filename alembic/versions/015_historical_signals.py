"""Таблица historical_signals — бэкафилл прошлых сообщений Telegram-канала
(см. src/telegram/history_backfill.py), отдельно от live-таблицы
telegram_signals, чтобы не искажать статистику канала (win rate,
expectancy sizing) историческими строками без реального исполнения.
Единственная цель — накопить размеченные примеры для будущей ML-модели
качества сигнала (обсуждение с пользователем: 20 живых закрытых сделок
недостаточно для обучения).

revision: 015
down_revision: 014
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '015'
down_revision: Optional[str] = '014'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    op.create_table(
        'historical_signals',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('channel_id', sa.Integer(), sa.ForeignKey('telegram_channels.id'), nullable=False),
        sa.Column('telegram_message_id', sa.Integer(), nullable=False),
        sa.Column('raw_message', sa.Text(), nullable=False),
        sa.Column('message_date', sa.DateTime(), nullable=False),
        sa.Column('parse_status', sa.String(length=20), nullable=False),
        sa.Column('parsed_pair', sa.String(length=50), nullable=True),
        sa.Column('parsed_side', sa.String(length=10), nullable=True),
        sa.Column('parsed_entry', sa.DECIMAL(), nullable=True),
        sa.Column('parsed_sl', sa.DECIMAL(), nullable=True),
        sa.Column('parsed_tp', sa.DECIMAL(), nullable=True),
        sa.Column('parsed_take_profits', sa.JSON(), nullable=True),
        sa.Column('parsed_leverage', sa.DECIMAL(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_unique_constraint(
        'uq_historical_signal_message', 'historical_signals', ['channel_id', 'telegram_message_id'],
    )
    op.create_index(
        'ix_historical_signals_channel_date', 'historical_signals', ['channel_id', 'message_date'],
    )


def downgrade():
    op.drop_index('ix_historical_signals_channel_date', table_name='historical_signals')
    op.drop_constraint('uq_historical_signal_message', 'historical_signals', type_='unique')
    op.drop_table('historical_signals')
