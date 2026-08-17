"""Decision Tree Logger — записывает полную цепочку решений для каждой сделки."""
import logging
from datetime import datetime
from typing import Any, Optional

from src.db.session import get_session
from src.db.models import TradeDecisionLog, Trade
from src.utils.logging import logger

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Логирует каждый шаг принятия решения для сделки.
    Позволяет потом просматривать "почему эта сделка была открыта/закрыта".

    Шаги:
    1. market_data — какие данные были на момент сигнала
    2. strategy_signal — какая стратегия сгенерировала сигнал и почему
    3. ml_score — ML предсказание (если использовалось)
    4. risk_check — результат проверки риск-менеджером
    5. execution — исполнение ордера
    """

    def __init__(self):
        self._current_trade_id: Optional[int] = None
        self._steps: list[dict] = []

    def start_trade(self, trade_id: int):
        """Начать логирование для сделки."""
        self._current_trade_id = trade_id
        self._steps = []

    def log_step(
        self,
        step_type: str,
        description: str,
        details: Optional[dict] = None,
    ):
        """Записать шаг решения."""
        if self._current_trade_id is None:
            logger.warning(f"DecisionLogger: попытка log_step без текущей сделки (step_type={step_type})")
            return

        step_order = len(self._steps) + 1
        step = {
            "trade_id": self._current_trade_id,
            "step_order": step_order,
            "step_type": step_type,
            "description": description,
            "details": details or {},
            "timestamp": datetime.utcnow(),
        }
        self._steps.append(step)
        logger.debug(f"DecisionLog [{step_order}] {step_type}: {description}")

    def log_market_data(self, symbol: str, timeframe: str, price: float, features: dict):
        """Лог market data на момент сигнала."""
        self.log_step(
            "market_data",
            f"Рыночные данные: {symbol} {timeframe} @ {price:.2f}",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "price": price,
                "features": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in list(features.items())[:20]},
            },
        )

    def log_strategy_signal(
        self,
        strategy_id: str,
        strategy_name: str,
        signal_side: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        rationale: str,
    ):
        """Лог сигнала от стратегии."""
        self.log_step(
            "strategy_signal",
            f"Стратегия {strategy_name} ({strategy_id}): {signal_side.upper()} confident={confidence:.2f}",
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "side": signal_side,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "rationale": rationale,
            },
        )

    def log_ml_score(
        self,
        model_type: str,
        model_version: int,
        proba_up: float,
        proba_down: float,
        proba_neutral: float,
        feature_importance: Optional[dict] = None,
    ):
        """Лог ML предсказания."""
        self.log_step(
            "ml_score",
            f"ML {model_type} v{model_version}: P(up)={proba_up:.2f} P(down)={proba_down:.2f}",
            {
                "model_type": model_type,
                "model_version": model_version,
                "proba_up": proba_up,
                "proba_down": proba_down,
                "proba_neutral": proba_neutral,
                "feature_importance": feature_importance,
            },
        )

    def log_risk_check(
        self,
        decision: str,  # allowed, rejected
        reason: str,
        context: dict,
    ):
        """Лог проверки риск-менеджером."""
        self.log_step(
            "risk_check",
            f"Риск-менеджмент: {'✅ Допущено' if decision == 'allowed' else '❌ Отклонено'} — {reason}",
            {
                "decision": decision,
                "reason": reason,
                "context": context,
            },
        )

    def log_execution(
        self,
        order_id: str,
        order_type: str,
        amount: float,
        price: float,
        status: str,
        fee: float,
    ):
        """Лог исполнения ордера."""
        self.log_step(
            "execution",
            f"Ордер {order_id}: {order_type} {amount:.6f} @ {price:.2f} → {status}",
            {
                "order_id": order_id,
                "order_type": order_type,
                "amount": amount,
                "price": price,
                "status": status,
                "fee": fee,
            },
        )

    def log_position_update(
        self,
        position_id: int,
        current_price: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float,
    ):
        """Лог обновления позиции."""
        self.log_step(
            "position_update",
            f"Позиция #{position_id}: цена {current_price:.2f}, PnL {unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.2f}%)",
            {
                "position_id": position_id,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            },
        )

    def save_to_db(self) -> list[int]:
        """Сохранить все шаги в БД. Возвращает список сохранённых ID."""
        if self._current_trade_id is None or not self._steps:
            return []

        saved_ids = []
        try:
            async def _save():
                async with get_session() as session:
                    for step in self._steps:
                        log_entry = TradeDecisionLog(
                            trade_id=self._current_trade_id,
                            step_order=step["step_order"],
                            step_type=step["step_type"],
                            description=step["description"],
                            details=step["details"],
                        )
                        session.add(log_entry)
                        await session.flush()
                        saved_ids.append(log_entry.id)
                    await session.commit()

            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_save())
        except Exception as e:
            logger.error(f"Ошибка сохранения decision log в БД: {e}")

        return saved_ids


# Глобальный экземпляр (используется в main.py)
decision_logger = DecisionLogger()
