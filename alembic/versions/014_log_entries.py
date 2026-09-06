"""Персистентная таблица log_entries — /logs веб-панели раньше отдавал
только in-memory ring-буфер (RingBufferHandler, capacity=2000), который
терялся целиком при каждом рестарте процесса и перезаписывался за
десятки минут активной торговли, из-за чего диагностика инцидента
(например, почему сверка позиции не смогла найти закрывающую сделку)
была невозможна уже вскоре после события. Пишется фоновым flush-циклом
(main.py: _flush_logs_to_db_loop) без ограничения на возраст записи —
история хранится целиком.

revision: 014
down_revision: 013
"""

from typing import Optional

from alembic import op
import sqlalchemy as sa

revision: str = '014'
down_revision: Optional[str] = '013'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade():
    op.create_table(
        'log_entries',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('level', sa.String(length=10), nullable=False),
        sa.Column('logger', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
    )
    op.create_index('ix_log_entries_timestamp', 'log_entries', ['timestamp'])
    op.create_index('ix_log_entries_level', 'log_entries', ['level'])
    op.create_index('ix_log_entries_logger', 'log_entries', ['logger'])


def downgrade():
    op.drop_index('ix_log_entries_logger', table_name='log_entries')
    op.drop_index('ix_log_entries_level', table_name='log_entries')
    op.drop_index('ix_log_entries_timestamp', table_name='log_entries')
    op.drop_table('log_entries')
