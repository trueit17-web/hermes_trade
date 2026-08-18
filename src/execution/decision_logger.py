"""Decision Tree Logger — записывает полную цепочку решений для каждой сделки."""
import logging
from typing import Any, Optional

from src.db.session import get_session
from src.db.models import TradeDecisionLog
from src.utils.logging import logger
from src.utils.timeutils import utcnow

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
    6. position_update — закрытие позиции (SL/TP/ручное)

    TradeDecisionLog.trade_id — это FK на Trade, а Trade создаётся только
    при ЗАКРЫТИИ позиции (открытие даёт только Order). Поэтому шаги нельзя
    писать в БД сразу: они копятся в памяти по order_id открывающего ордера
    (attach_to_order) и сбрасываются в БД только когда позиция закрывается
    и появляется настоящий trade_id (flush_for_trade). Если ордер так и не
    привёл к сделке (сигнал отклонён риском/качеством), накопленные шаги
    просто отбрасываются — писать их некуда, Trade для них никогда не будет.
    """

    def __init__(self):
        self._active_steps: list[dict] = []
        self._pending_by_order: dict[int, list[dict]] = {}

    def begin(self):
        """Начать новую цепочку решений (перед обработкой одного символа/сигнала)."""
        self._active_steps = []

    def log_step(
        self,
        step_type: str,
        description: str,
        details: Optional[dict] = None,
    ):
        """Записать шаг в текущую (ещё не привязанную к ордеру) цепочку."""
        step_order = len(self._active_steps) + 1
        step = {
            "step_order": step_order,
            "step_type": step_type,
            "description": description,
            "details": details or {},
            "timestamp": utcnow(),
        }
        self._active_steps.append(step)
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

    def attach_to_order(self, order_id: Optional[int]):
        """
        Привязать накопленную с последнего begin() цепочку шагов к id
        открывающего ордера — она будет ждать закрытия позиции, чтобы
        попасть в БД вместе с настоящим trade_id (см. flush_for_trade).
        """
        if order_id is not None and self._active_steps:
            self._pending_by_order[order_id] = self._active_steps
        self._active_steps = []

    async def flush_for_trade(
        self,
        order_id: Optional[int],
        trade_id: int,
        close_description: Optional[str] = None,
        close_details: Optional[dict] = None,
    ) -> list[int]:
        """
        Записать в БД цепочку шагов, накопленную при открытии ордера
        order_id, привязав её к закрытой сделке trade_id, вместе с
        завершающим шагом о закрытии позиции. Возвращает id сохранённых
        записей (пустой список, если писать было нечего).
        """
        steps = list(self._pending_by_order.pop(order_id, [])) if order_id is not None else []

        if close_description is not None:
            steps.append({
                "step_order": len(steps) + 1,
                "step_type": "position_update",
                "description": close_description,
                "details": close_details or {},
            })

        if not steps:
            return []

        saved_ids = []
        try:
            async with get_session() as session:
                for step in steps:
                    log_entry = TradeDecisionLog(
                        trade_id=trade_id,
                        step_order=step["step_order"],
                        step_type=step["step_type"],
                        description=step["description"],
                        details=step["details"],
                    )
                    session.add(log_entry)
                    await session.flush()
                    saved_ids.append(log_entry.id)
                await session.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения decision log в БД: {e}")

        return saved_ids


# Глобальный экземпляр (используется в main.py)
decision_logger = DecisionLogger()
