"""Risk Manager — управление рисками: лимиты, sizing, SL/TP, circuit breakers."""
import logging
from datetime import datetime, timedelta
from typing import Any

from src.config import settings
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class RiskProfile:
    """Профиль риска для бота — все параметры риск-менеджмента."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.daily_loss_limit_usd = self.params.get("daily_loss_limit_usd", settings.risk_daily_loss_limit_usd)
        self.max_open_positions = self.params.get("max_open_positions", settings.risk_max_open_positions)
        self.max_position_size_pct = self.params.get("max_position_size_pct", settings.risk_max_position_size_pct)
        self.max_drawdown_pct = self.params.get("max_drawdown_pct", settings.risk_max_drawdown_pct)
        self.cooldown_seconds = self.params.get("cooldown_seconds", settings.risk_cooldown_seconds)
        self.max_correlation_pairs = self.params.get("max_correlation_pairs", 3)

    def update(self, new_params: dict):
        """Обновить параметры профиля."""
        self.params.update(new_params)
        self.daily_loss_limit_usd = new_params.get("daily_loss_limit_usd", self.daily_loss_limit_usd)
        self.max_open_positions = new_params.get("max_open_positions", self.max_open_positions)
        self.max_position_size_pct = new_params.get("max_position_size_pct", self.max_position_size_pct)
        self.max_drawdown_pct = new_params.get("max_drawdown_pct", self.max_drawdown_pct)
        self.cooldown_seconds = new_params.get("cooldown_seconds", self.cooldown_seconds)
        self.max_correlation_pairs = new_params.get("max_correlation_pairs", self.max_correlation_pairs)
        logger.info(f"RiskProfile обновлён: {self.params}")


class RiskState:
    """Состояние риска в реальном времени."""

    def __init__(self):
        # Стартовый капитал аккаунта, а не "баланс на момент первого вызова
        # update_balance()" — иначе после каждого рестарта базой для расчёта
        # просадки становился бы текущий баланс, и любая уже случившаяся
        # просадка стала бы невидимой для max_drawdown_pct.
        self.start_balance = settings.startup_capital_usdt
        self.current_balance = settings.startup_capital_usdt
        self.daily_pnl = 0.0
        self.daily_loss_limit_reached = False
        self.daily_loss_reset_time: datetime | None = None
        self.open_positions_count = 0
        self.open_positions: dict[str, float] = {}  # symbol -> size_pct
        self.total_drawdown_pct = 0.0
        self.max_drawdown_reached = 0.0
        self.last_trade_time: datetime | None = None
        self.cooldown_active = False
        self.paused = False
        self.kill_switch_active = False

        # Пороговые значения (по умолчанию из настроек, синхронизируются
        # с RiskProfile через RiskManager при изменении конфигурации)
        self.max_open_positions = settings.risk_max_open_positions
        self.daily_loss_limit_usd = settings.risk_daily_loss_limit_usd
        self.max_drawdown_pct = settings.risk_max_drawdown_pct
        self.cooldown_seconds = settings.risk_cooldown_seconds

    def update_balance(self, balance: float):
        """Обновить текущий баланс."""
        self.current_balance = balance

    def update_daily_pnl(self, pnl: float):
        """Обновить daily PnL."""
        self.daily_pnl += pnl
        if self.daily_pnl <= -self.daily_loss_limit_usd:
            self.daily_loss_limit_reached = True
            logger.warning(
                f"🚨 Дневной лимит убытков достигнут: {self.daily_pnl:.2f} / {self.daily_loss_limit_usd:.2f}"
            )

    def check_daily_loss_limit_reset(self):
        """Проверить, можно ли сбросить daily loss лимит (начало нового дня)."""
        if self.daily_loss_limit_reached:
            now = utcnow()
            if self.daily_loss_reset_time is None:
                # Сброс произойдёт на следующую полночь, а не в момент срабатывания лимита
                next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                self.daily_loss_reset_time = next_midnight
            elif now >= self.daily_loss_reset_time:
                self.daily_pnl = 0.0
                self.daily_loss_limit_reached = False
                logger.info(f"🔄 Daily PnL сброшен (новый день, {now.date()})")
                self.daily_loss_reset_time = None

    def add_open_position(self, symbol: str, size_pct: float):
        """Добавить открытую позицию."""
        self.open_positions[symbol] = size_pct
        self.open_positions_count = len(self.open_positions)
        logger.debug(
            f"Позиция добавлена: {symbol} ({size_pct:.1f}%), "
            f"всего: {self.open_positions_count}/{self.max_open_positions}"
        )

    def remove_open_position(self, symbol: str):
        """Удалить открытую позицию."""
        self.open_positions.pop(symbol, None)
        self.open_positions_count = len(self.open_positions)

    def check_cooldown(self) -> bool:
        """Проверить кулдаун после последней сделки."""
        if self.cooldown_seconds <= 0:
            return False

        if self.last_trade_time is None:
            return False

        elapsed = (utcnow() - self.last_trade_time).total_seconds()
        if elapsed < self.cooldown_seconds:
            self.cooldown_active = True
            remaining = self.cooldown_seconds - elapsed
            logger.debug(f"Кулдаун активен, осталось {remaining:.0f}s")
            return True

        self.cooldown_active = False
        return False

    def record_trade_time(self):
        """Записать время последней сделки."""
        self.last_trade_time = utcnow()
        self.cooldown_active = True

    def check_max_positions(self) -> bool:
        """Проверить, не превышен ли лимит открытых позиций."""
        return self.open_positions_count >= self.max_open_positions

    def check_daily_loss(self) -> bool:
        """Проверить, достигнут ли дневной лимит убытков."""
        return self.daily_loss_limit_reached or self.paused or self.kill_switch_active

    def check_max_drawdown(self, current_equity: float) -> bool:
        """Проверить, не достигнут ли максимальный даун-драфт."""
        if self.start_balance > 0:
            drawdown = (self.start_balance - current_equity) / self.start_balance * 100
            self.total_drawdown_pct = drawdown
            self.max_drawdown_reached = max(self.max_drawdown_reached, drawdown)
            return drawdown > self.max_drawdown_pct
        return False

    def pause(self):
        """Приостановить торговлю."""
        self.paused = True
        logger.warning("Торговля приостановлена (risk pause)")

    def resume(self):
        """Возобновить торговлю."""
        self.paused = False
        logger.info("Торговля возобновлена")

    def trigger_kill_switch(self):
        """Активировать kill switch."""
        self.kill_switch_active = True
        self.paused = True
        logger.critical("🔴 KILL SWITCH АКТИВИРОВАН — все торговые операции остановлены")

    def clear_kill_switch(self):
        """Сбросить kill switch."""
        self.kill_switch_active = False
        logger.info("Kill switch сброшен")


class RiskManager:
    """
    Риск-менеджер — проверяет каждое действие на соответствие профилю риска.
    """

    def __init__(self, profile: RiskProfile | None = None):
        self.profile = profile or RiskProfile()
        self.state = RiskState()
        self._sync_state_from_profile()
        self.last_pnl_update: datetime | None = None

    def _sync_state_from_profile(self):
        """Синхронизировать пороговые значения state с текущим профилем риска."""
        self.state.max_open_positions = self.profile.max_open_positions
        self.state.daily_loss_limit_usd = self.profile.daily_loss_limit_usd
        self.state.max_drawdown_pct = self.profile.max_drawdown_pct
        self.state.cooldown_seconds = self.profile.cooldown_seconds

    def reload_from_settings(self):
        """
        Перечитать пороги риска из settings.risk_* прямо сейчас.

        risk_manager — модульный синглтон, конструируется в момент импорта
        модуля (import risk_manager в main.py), а load_settings_overrides()
        (переопределения, сохранённые через дашборд в bot_config)
        применяется намного позже, уже внутри TradingBot.initialize(). То
        есть RiskProfile() в __init__ всегда читает settings.risk_* ДО того,
        как overrides из БД успевают примениться — лимит, изменённый через
        дашборд и корректно сохранённый, после рестарта тихо откатывался
        обратно к значению из .env/дефолту. Нужно явно перечитать профиль
        после load_settings_overrides().
        """
        self.profile.daily_loss_limit_usd = settings.risk_daily_loss_limit_usd
        self.profile.max_open_positions = settings.risk_max_open_positions
        self.profile.max_position_size_pct = settings.risk_max_position_size_pct
        self.profile.max_drawdown_pct = settings.risk_max_drawdown_pct
        self.profile.cooldown_seconds = settings.risk_cooldown_seconds
        self._sync_state_from_profile()

    async def restore_daily_pnl_from_db(self):
        """
        Восстановить daily_pnl из уже закрытых сегодня сделок при старте бота.

        daily_pnl живёт только в памяти — без этого рестарт в течение дня
        обнулял бы счётчик, и дневной лимит убытков (risk_daily_loss_limit_usd)
        эффективно переставал действовать до конца суток, разрешая новые
        просадки поверх уже случившихся сегодня.
        """
        from sqlalchemy import func, select

        from src.db.models import Trade
        from src.db.session import get_session

        today_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            async with get_session() as session:
                total = (
                    await session.execute(
                        select(func.sum(Trade.pnl)).where(Trade.closed_at >= today_start)
                    )
                ).scalar() or 0
        except Exception as e:
            logger.warning(f"Не удалось восстановить дневной PnL из БД: {e}")
            return

        self.state.daily_pnl = float(total)
        if self.state.daily_pnl <= -self.state.daily_loss_limit_usd:
            self.state.daily_loss_limit_reached = True
        logger.info(f"🔗 Дневной PnL восстановлен из БД: {self.state.daily_pnl:.2f}")

    def configure(self, params: dict):
        """Обновить конфигурацию риска."""
        self.profile.update(params)
        self._sync_state_from_profile()

    def reset_for_new_paper_account(self):
        """
        Сбросить риск-состояние вместе со сбросом paper-аккаунта
        (execution_engine.reset_paper_account): базу для просадки и дневной
        PnL — на стартовый капитал, снять паузу, если она была вызвана
        именно старой просадкой (kill switch — отдельный, более серьёзный
        стоп, сбрасывается только вручную через /risk/clear-kill-switch).
        """
        self.state.start_balance = settings.startup_capital_usdt
        self.state.current_balance = settings.startup_capital_usdt
        self.state.daily_pnl = 0.0
        self.state.daily_loss_limit_reached = False
        self.state.total_drawdown_pct = 0.0
        self.state.max_drawdown_reached = 0.0
        self.state.open_positions_count = 0
        self.state.open_positions = {}
        if not self.state.kill_switch_active:
            self.state.paused = False
        logger.warning(f"🔄 Риск-состояние сброшено вместе с paper-аккаунтом: старт={self.state.start_balance:.2f}")

    def reset_for_real_account(self, balance: float):
        """
        Пересчитать базу для просадки от реального баланса биржи при входе
        в real-режим (первое подключение, смена биржи/sandbox). До этого
        start_balance был захардкожен на settings.startup_capital_usdt
        (paper-ориентированный дефолт, обычно 10000) и никогда не совпадал
        с реальным балансом — это гарантированно давало ложную просадку
        (вплоть до 100%, если баланс к тому же читался как 0) и мгновенную
        паузу торговли сразу после переключения в real. В отличие от
        reset_for_new_paper_account, открытые позиции не затрагиваются —
        реальный аккаунт не "обнуляется", а его состояние просто уже
        восстановлено отдельно из БД.
        """
        self.state.start_balance = balance
        self.state.current_balance = balance
        self.state.daily_pnl = 0.0
        self.state.daily_loss_limit_reached = False
        self.state.total_drawdown_pct = 0.0
        self.state.max_drawdown_reached = 0.0
        if not self.state.kill_switch_active:
            self.state.paused = False
        logger.warning(f"🔄 Риск-состояние пересчитано от реального баланса биржи: старт={balance:.2f}")

    def get_state(self) -> dict:
        """Получить текущее состояние риска."""
        self.state.check_cooldown()
        return {
            "daily_loss_limit": self.profile.daily_loss_limit_usd,
            "daily_pnl": self.state.daily_pnl,
            "daily_loss_limit_reached": self.state.daily_loss_limit_reached,
            "max_open_positions": self.profile.max_open_positions,
            "open_positions_count": self.state.open_positions_count,
            "open_positions": dict(self.state.open_positions),
            "max_position_size_pct": self.profile.max_position_size_pct,
            "max_drawdown_pct": self.profile.max_drawdown_pct,
            "total_drawdown_pct": self.state.total_drawdown_pct,
            "max_drawdown_reached": self.state.max_drawdown_reached,
            "cooldown_seconds": self.profile.cooldown_seconds,
            "cooldown_active": self.state.cooldown_active,
            "paused": self.state.paused,
            "kill_switch": self.state.kill_switch_active,
            "start_balance": self.state.start_balance,
            "current_balance": self.state.current_balance,
        }

    def can_trade(self) -> bool:
        """Можно ли открыть новую позицию?"""
        if self.state.kill_switch_active:
            return False
        if self.state.paused:
            return False
        if self.state.daily_loss_limit_reached:
            return False
        # check_cooldown() пересчитывает флаг по прошедшему времени, а не
        # голое чтение state.cooldown_active — record_trade_time() только
        # ВКЛЮЧАЕТ его после закрытия (в т.ч. частичного, TP1/TP2), но
        # выключить обратно по истечении cooldown_seconds могла только
        # check_cooldown(), которая раньше нигде не вызывалась — после
        # первого же закрытия за время работы процесса флаг оставался True
        # навсегда, блокируя вообще все новые входы до рестарта, независимо
        # от значения cooldown_seconds (поэтому его изменение в настройках
        # не имело вообще никакого эффекта).
        if self.state.check_cooldown():
            return False
        return not self.state.check_max_positions()

    def check_signal(self, signal: Any) -> tuple[bool, str | None]:
        """
        Проверить сигнал на соответствие риск-профилю.
        Возвращает (can_execute, reason).
        """
        if not self.can_trade():
            reason = "Торговля остановлена"
            if self.state.kill_switch_active:
                reason = "Kill switch активен"
            elif self.state.paused:
                reason = "Торговля приостановлена"
            elif self.state.daily_loss_limit_reached:
                reason = f"Дневной лимит убытков достигнут ({self.state.daily_pnl:.2f} / {self.profile.daily_loss_limit_usd:.2f})"
            elif self.state.cooldown_active:
                reason = "Кулдаун активен"
            elif self.state.check_max_positions():
                reason = f"Достигнут лимит позиций ({self.state.open_positions_count}/{self.profile.max_open_positions})"
            return False, reason

        # Проверка размера позиции
        size_pct = signal.position_size_pct if hasattr(signal, 'position_size_pct') else signal.get("position_size_pct", 0)
        if size_pct > self.profile.max_position_size_pct:
            return False, f"Размер позиции {size_pct:.1f}% превышает лимит {self.profile.max_position_size_pct:.1f}%"

        # Проверка корреляции
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get("symbol", "")
        if symbol in self.state.open_positions:
            return False, f"Уже есть позиция по {symbol}"

        return True, "OK"

    def adjust_position_size(
        self,
        signal: Any,
        current_balance: float,
        volatility_adj: float = 1.0,
    ) -> float:
        """
        Рассчитать размер позиции с учётом риска.
        volatility_adj: коэффициент волатильности (0.5 = уменьшить в 2 раза, 2.0 = увеличить)
        """
        base_size_pct = signal.position_size_pct if hasattr(signal, 'position_size_pct') else signal.get("position_size_pct", 5.0)

        # Ограничение максимального размера
        max_size = self.profile.max_position_size_pct
        adjusted_size = min(base_size_pct, max_size)

        # Корректировка по волатильности
        adjusted_size = adjusted_size * volatility_adj

        # Расчёт абсолютного размера в USDT
        position_value = current_balance * (adjusted_size / 100)
        return max(0, position_value)

    def on_trade_closed(self, trade_pnl: float):
        """Обработка закрытия сделки."""
        self.state.update_daily_pnl(trade_pnl)
        self.state.record_trade_time()

        logger.info(
            f"📊 Сделка закрыта | PnL: {trade_pnl:.2f} USDT | "
            f"Дневной PnL: {self.state.daily_pnl:.2f} | "
            f"Достигнут лимит: {self.state.daily_loss_limit_reached}"
        )

        # Проверка сброса daily loss
        self.state.check_daily_loss_limit_reset()

    def on_position_added(self, symbol: str, size_pct: float):
        """Добавить позицию в состояние."""
        self.state.add_open_position(symbol, size_pct)

    def on_position_closed(self, symbol: str):
        """Удалить позицию из состояния."""
        self.state.remove_open_position(symbol)

    def on_balance_update(self, balance: float):
        """Обновить баланс."""
        self.state.update_balance(balance)
        if self.state.check_max_drawdown(balance):
            self.state.pause()
            logger.warning(
                f"⚠️ Максимальный даун-драфт достигнут: "
                f"{self.state.total_drawdown_pct:.2f}% > {self.profile.max_drawdown_pct:.2f}%"
            )


# Глобальный экземпляр
risk_manager = RiskManager()
