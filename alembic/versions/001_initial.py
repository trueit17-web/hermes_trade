"""Alembic миграции для crypto-bot.

Этот файл содержит начальную миграцию, которая создаёт все таблицы
из src/db/models.py.

Для генерации новых миграций используйте:
    alembic revision --autogenerate -m "description"
    alembic upgrade head

Для отката:
    alembic downgrade -1
"""

from typing import Optional

from alembic import op
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String, Text,
    ForeignKey, Index, UniqueConstraint, Text as SqlText,
)
from sqlalchemy.dialects.postgresql import DECIMAL

# Импортируем все модели, чтобы Alembic мог сгенерировать миграции
# В продакшене это не требуется, так как миграции уже сгенерированы
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Для SQLite используем стандартные типы
# Для PostgreSQL можно использовать специфичные типы

revision: str = 'initial'
down_revision: Optional[str] = None
branch_labels: Optional[str] = None
depends_on: Optional[str] = None

def upgrade():
    """
    Создать все таблицы базы данных.
    """
    
    # === exchanges ===
    op.create_table(
        'exchanges',
        Column('id', Integer, primary_key=True),
        Column('name', String(50), unique=True, nullable=False),
        Column('api_key_enc', String(512), nullable=True),
        Column('api_secret_enc', String(512), nullable=True),
        Column('is_paper', Boolean, default=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('updated_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === symbols ===
    op.create_table(
        'symbols',
        Column('id', Integer, primary_key=True),
        Column('exchange_id', Integer, ForeignKey('exchanges.id')),
        Column('symbol', String(50), nullable=False),
        Column('active', Boolean, default=True),
        Column('base_asset', String(20)),
        Column('quote_asset', String(20)),
        Column('tick_size', Float, nullable=True),
        Column('step_size', Float, nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        UniqueConstraint('exchange_id', 'symbol', name='uq_exchange_symbol'),
        Index('ix_symbols_symbol', 'symbol'),
    )
    
    # === strategies ===
    op.create_table(
        'strategies',
        Column('id', Integer, primary_key=True),
        Column('name', String(100), nullable=False),
        Column('strategy_type', String(30), nullable=False),
        Column('params', JSON, default={}),
        Column('active', Boolean, default=True),
        Column('weight', Float, default=1.0),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('updated_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === signals ===
    op.create_table(
        'signals',
        Column('id', Integer, primary_key=True),
        Column('strategy_id', Integer, ForeignKey('strategies.id')),
        Column('symbol_id', Integer, ForeignKey('symbols.id')),
        Column('side', String(10), nullable=False),
        Column('confidence', Float),
        Column('entry_price', DECIMAL(precision=36, scale=18), nullable=True),
        Column('stop_loss', DECIMAL(precision=36, scale=18), nullable=True),
        Column('take_profit', DECIMAL(precision=36, scale=18), nullable=True),
        Column('position_size_pct', Float),
        Column('timeframe', String(10)),
        Column('rationale', Text),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === orders ===
    op.create_table(
        'orders',
        Column('id', Integer, primary_key=True),
        Column('exchange_id', Integer, ForeignKey('exchanges.id')),
        Column('symbol_id', Integer, ForeignKey('symbols.id')),
        Column('strategy_id', Integer, ForeignKey('strategies.id'), nullable=True),
        Column('signal_id', Integer, ForeignKey('signals.id'), nullable=True),
        Column('side', String(10), nullable=False),
        Column('order_type', String(20)),
        Column('amount', DECIMAL(precision=36, scale=18), nullable=False),
        Column('price', DECIMAL(precision=36, scale=18), nullable=True),
        Column('status', String(20), default='open'),
        Column('filled_amount', DECIMAL(precision=36, scale=18), default=0),
        Column('filled_price', DECIMAL(precision=36, scale=18), nullable=True),
        Column('fee', DECIMAL(precision=36, scale=18), default=0),
        Column('stop_loss', DECIMAL(precision=36, scale=18), nullable=True),
        Column('take_profit', DECIMAL(precision=36, scale=18), nullable=True),
        Column('order_id_exchange', String(100), nullable=True),
        Column('client_order_id', String(100), nullable=True),
        Column('notes', Text, nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('updated_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('closed_at', DateTime, nullable=True),
    )
    
    # === trades ===
    op.create_table(
        'trades',
        Column('id', Integer, primary_key=True),
        Column('symbol_id', Integer, ForeignKey('symbols.id')),
        Column('strategy_id', Integer, ForeignKey('strategies.id'), nullable=True),
        Column('order_open_id', Integer, ForeignKey('orders.id'), nullable=True),
        Column('order_close_id', Integer, ForeignKey('orders.id'), nullable=True),
        Column('direction', String(10), nullable=False),
        Column('entry_price', DECIMAL(precision=36, scale=18), nullable=False),
        Column('exit_price', DECIMAL(precision=36, scale=18), nullable=True),
        Column('amount', DECIMAL(precision=36, scale=18), nullable=False),
        Column('pnl', DECIMAL(precision=36, scale=18), default=0),
        Column('pnl_pct', Float),
        Column('holding_seconds', Integer, default=0),
        Column('features_at_open', JSON, nullable=True),
        Column('outcome', String(20), nullable=True),
        Column('is_open', Boolean, default=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('closed_at', DateTime, nullable=True),
    )
    
    # === candles ===
    op.create_table(
        'candles',
        Column('id', Integer, primary_key=True),
        Column('exchange_id', Integer, ForeignKey('exchanges.id')),
        Column('symbol_id', Integer, ForeignKey('symbols.id')),
        Column('timeframe', String(10), nullable=False),
        Column('open_time', DateTime, nullable=False),
        Column('open', DECIMAL(precision=36, scale=18)),
        Column('high', DECIMAL(precision=36, scale=18)),
        Column('low', DECIMAL(precision=36, scale=18)),
        Column('close', DECIMAL(precision=36, scale=18)),
        Column('volume', DECIMAL(precision=36, scale=18)),
        Column('close_time', DateTime, nullable=True),
        UniqueConstraint('exchange_id', 'symbol_id', 'timeframe', 'open_time', name='uq_candle_unique'),
        Index('ix_candles_symbol_tf', 'symbol_id', 'timeframe'),
    )
    
    # === telegram_channels ===
    op.create_table(
        'telegram_channels',
        Column('id', Integer, primary_key=True),
        Column('channel_id', String(100), unique=True, nullable=False),
        Column('channel_title', String(200), nullable=True),
        Column('parser_type', String(20), default='regex'),
        Column('parser_config', JSON, default={}),
        Column('quality_threshold', Float, default=0.5),
        Column('auto_execute', Boolean, default=False),
        Column('active', Boolean, default=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('updated_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === telegram_signals ===
    op.create_table(
        'telegram_signals',
        Column('id', Integer, primary_key=True),
        Column('channel_id', Integer, ForeignKey('telegram_channels.id')),
        Column('raw_message', Text, nullable=False),
        Column('message_date', DateTime),
        Column('parsed_pair', String(50), nullable=True),
        Column('parsed_side', String(10), nullable=True),
        Column('parsed_entry', DECIMAL(precision=36, scale=18), nullable=True),
        Column('parsed_sl', DECIMAL(precision=36, scale=18), nullable=True),
        Column('parsed_tp', DECIMAL(precision=36, scale=18), nullable=True),
        Column('quality_score', Float, nullable=True),
        Column('decision', String(30), default='pending'),
        Column('executed_trade_id', Integer, ForeignKey('trades.id'), nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === ml_models ===
    op.create_table(
        'ml_models',
        Column('id', Integer, primary_key=True),
        Column('model_type', String(50), nullable=False),
        Column('strategy_id', Integer, ForeignKey('strategies.id'), nullable=True),
        Column('version', Integer, nullable=False),
        Column('model_path', String(500), nullable=True),
        Column('params', JSON),
        Column('metrics', JSON),
        Column('is_active', Boolean, default=False),
        Column('is_shadow', Boolean, default=False),
        Column('released_at', DateTime, nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Index('ix_ml_model_active', 'model_type', 'is_active'),
    )
    
    # === ml_features ===
    op.create_table(
        'ml_features',
        Column('id', Integer, primary_key=True),
        Column('symbol', String(50), nullable=False),
        Column('timeframe', String(10), nullable=False),
        Column('timestamp', DateTime, nullable=False),
        Column('features', JSON, nullable=False),
        Column('label_direction', Integer, nullable=True),
        Column('label_volatility', Float, nullable=True),
        Column('source', String(20)),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_feature_unique'),
        Index('ix_features_symbol_ts', 'symbol', 'timestamp'),
    )
    
    # === bot_config ===
    op.create_table(
        'bot_config',
        Column('id', Integer, primary_key=True),
        Column('config_key', String(100), unique=True, nullable=False),
        Column('config_value', JSON, nullable=False),
        Column('source', String(20), default='default'),
        Column('updated_by', String(50), nullable=True),
        Column('updated_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
    )
    
    # === bot_events ===
    op.create_table(
        'bot_events',
        Column('id', Integer, primary_key=True),
        Column('level', String(20), nullable=False),
        Column('source', String(50), nullable=False),
        Column('message', Text, nullable=False),
        Column('details', JSON, nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Index('ix_events_created', 'created_at'),
        Index('ix_events_level_source', 'level', 'source'),
    )
    
    # === performance_snapshots ===
    op.create_table(
        'performance_snapshots',
        Column('id', Integer, primary_key=True),
        Column('snapshot_time', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Column('total_balance', DECIMAL(precision=36, scale=18)),
        Column('open_pnl', DECIMAL(precision=36, scale=18), default=0),
        Column('realized_pnl', DECIMAL(precision=36, scale=18), default=0),
        Column('daily_pnl', DECIMAL(precision=36, scale=18), default=0),
        Column('weekly_pnl', DECIMAL(precision=36, scale=18), default=0),
        Column('num_open_positions', Integer, default=0),
        Column('num_trades_today', Integer, default=0),
        Column('max_drawdown', Float, nullable=True),
        Column('sharpe_ratio', Float, nullable=True),
        Column('win_rate', Float, nullable=True),
    )
    
    # === trade_decision_logs ===
    op.create_table(
        'trade_decision_logs',
        Column('id', Integer, primary_key=True),
        Column('trade_id', Integer, ForeignKey('trades.id')),
        Column('step_order', Integer, nullable=False),
        Column('step_type', String(30), nullable=False),
        Column('description', Text, nullable=False),
        Column('details', JSON, nullable=True),
        Column('created_at', DateTime, default=op.inline_literal('CURRENT_TIMESTAMP')),
        Index('ix_decision_log_trade', 'trade_id'),
    )


def downgrade():
    """
    Удалить все таблицы базы данных (in обратном порядке).
    """
    # Удаляем в обратном порядке создания (unnrelated constraints first)
    op.drop_table('trade_decision_logs')
    op.drop_table('performance_snapshots')
    op.drop_table('bot_events')
    op.drop_table('bot_config')
    op.drop_table('ml_features')
    op.drop_table('ml_models')
    op.drop_table('telegram_signals')
    op.drop_table('telegram_channels')
    op.drop_table('candles')
    op.drop_table('trades')
    op.drop_table('orders')
    op.drop_table('signals')
    op.drop_table('strategies')
    op.drop_table('symbols')
    op.drop_table('exchanges')
