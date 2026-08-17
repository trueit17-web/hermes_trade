"""Execution Engine — отправка ордеров, исполнение, трекинг."""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

import ccxt.async_support as ccxt
from sqlalchemy import select

from src.config import settings
from src.db.session import get_session
from src.db.models import (
    Order, Trade, Strategy, Symbol, Exchange, TradeDecisionLog,
)
from src.risk.risk_manager import risk_manager
from src.event_bus import (
    event_bus,
    OrderEvent,
    TradeEvent,
)
from src.utils.logging import logger, log_trade
from src.utils.crypto import decrypt_api_key, decrypt_secret

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Движок исполнения ордеров."""

    def __init__(self):
        self.exchange: Optional[ccxt.Exchange] = None
        self.exchange_id: Optional[str] = None
        self.is_paper: bool = settings.is_paper
        self.paper_balance: float = settings.startup_capital_usdt
        self.paper_positions: dict[str, dict] = {}
        self.order_counter = 0

    async def initialize(self, exchange_id: str = "binance"):
        """Инициализация подключения к бирже."""
        self.exchange_id = exchange_id

        if self.is_paper:
            logger.info("📄 Execution Engine: Paper Trading режим")
            self.paper_balance = settings.startup_capital_usdt
            return

        # Real mode: подключаемся к бирже
        try:
            api_key = settings.binance_api_key or ""
            api_secret = settings.binance_api_secret or ""

            if not api_key or not api_secret:
                logger.warning("⚠️ API ключи не указаны, переключаемся в paper режим")
                self.is_paper = True
                self.paper_balance = settings.startup_capital_usdt
                return

            if exchange_id == "binance":
                self.exchange = ccxt.binance({
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            elif exchange_id == "bybit":
                self.exchange = ccxt.bybit({
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            else:
                exchange_class = getattr(ccxt, exchange_id, None)
                if exchange_class:
                    self.exchange = exchange_class({
                        "apiKey": api_key,
                        "secret": api_secret,
                        "enableRateLimit": True,
                    })

            await self.exchange.load_markets()
            logger.info(f"🔗 Execution Engine: подключено к {exchange_id}")

            # Проверка баланса
            try:
                balance = await self.exchange.fetch_balance()
                total_usdt = sum(
                    v for k, v in balance.items()
                    if isinstance(v, (int, float)) and k.endswith("USDT")
                )
                self.paper_balance = total_usdt
                logger.info(f"💰 Баланс USDT: {total_usdt:.2f}")
            except Exception as e:
                logger.warning(f"Не удалось получить баланс: {e}")

        except Exception as e:
            logger.error(f"Ошибка инициализации биржи {exchange_id}: {e}")
            self.is_paper = True

    async def close(self):
        """Закрыть соединение с биржей."""
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Execution Engine: соединение закрыто")

    async def _resolve_symbol_id(self, session, symbol: str) -> tuple[int, int]:
        """Получить (или создать) id биржи и торговой пары в БД."""
        exchange_name = self.exchange_id or "paper"

        exchange = (
            await session.execute(select(Exchange).where(Exchange.name == exchange_name))
        ).scalar_one_or_none()
        if exchange is None:
            exchange = Exchange(name=exchange_name, is_paper=self.is_paper)
            session.add(exchange)
            await session.flush()

        symbol_row = (
            await session.execute(
                select(Symbol).where(Symbol.exchange_id == exchange.id, Symbol.symbol == symbol)
            )
        ).scalar_one_or_none()
        if symbol_row is None:
            base_asset, _, quote_asset = symbol.partition("/")
            symbol_row = Symbol(
                exchange_id=exchange.id,
                symbol=symbol,
                base_asset=base_asset or symbol,
                quote_asset=quote_asset,
            )
            session.add(symbol_row)
            await session.flush()

        return exchange.id, symbol_row.id

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[int] = None,
        signal_data: Optional[dict] = None,
    ) -> Optional[Order]:
        """
        Создать ордер.
        Возвращает Order объект (сохранённый в БД) или None.
        """
        if risk_manager.state.kill_switch_active:
            logger.warning(f"❌ Попытка создать ордер при активном kill switch: {symbol}")
            return None

        if not self.can_execute():
            logger.warning(f"❌ Исполнение отклонено: {symbol} {side}")
            return None

        client_order_id = str(uuid.uuid4())[:12]

        # Получить цену исполнения
        execution_price = price
        if order_type == "market" and execution_price is None:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                execution_price = ticker["last"] or ticker["bid"] or ticker["ask"]
            except Exception as e:
                logger.warning(f"Не удалось получить цену для {symbol}: {e}")
                if execution_price is None:
                    return None

        # Расчёт комиссии
        fee_pct = 0.001  # 0.1% для spot
        if self.is_paper:
            fee_pct = settings.paper_fee_pct / 100

        fee = amount * execution_price * fee_pct

        # Создание ордера
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": execution_price,
            "client_order_id": client_order_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "fee": fee,
            "strategy_id": strategy_id,
            "signal_data": signal_data,
        }

        logger.info(
            f"📝 Создание ордера: {side.upper()} {amount:.4f} {symbol} @ {execution_price:.2f} | "
            f"SL={stop_loss:.2f} TP={take_profit:.2f} | ID={client_order_id}"
        )

        if self.is_paper:
            return await self._execute_paper_order(order_data)
        else:
            return await self._execute_real_order(order_data)

    async def _execute_paper_order(self, order_data: dict) -> Optional[Order]:
        """Экземпляр paper trading ордера."""
        symbol = order_data["symbol"]
        side = order_data["side"]
        amount = order_data["amount"]
        price = order_data["price"]
        fee = order_data["fee"]

        # Симуляция slippage
        slippage_pct = settings.paper_slippage_pct / 100
        slippage = price * slippage_pct
        if side == "buy":
            price += slippage
        else:
            price -= slippage

        # Обновление paper баланса и позиций
        position_value = amount * price
        if side == "buy":
            self.paper_balance -= position_value + fee
            if symbol not in self.paper_positions:
                self.paper_positions[symbol] = {"amount": 0, "entry_price": 0, "side": "long"}
            self.paper_positions[symbol]["amount"] += amount
            self.paper_positions[symbol]["entry_price"] = (
                (self.paper_positions[symbol]["entry_price"] * self.paper_positions[symbol]["amount"] + price * amount)
                / (self.paper_positions[symbol]["amount"])
            )
            self.paper_positions[symbol]["side"] = "long"
        else:
            if symbol in self.paper_positions:
                pos = self.paper_positions[symbol]
                if pos["amount"] >= amount:
                    self.paper_balance += position_value - fee
                    pos["amount"] -= amount
                    if pos["amount"] <= 0:
                        del self.paper_positions[symbol]
                    # частичное закрытие не меняет среднюю цену входа оставшейся позиции
                else:
                    logger.warning(f"Paper: недостаточно позиции {symbol} для закрытия")
                    return None

        logger.info(f"📄 Paper ордер исполнен: {side.upper()} {amount:.4f} {symbol} @ {price:.2f}")

        # Создание Order объекта в БД
        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            order = Order(
                exchange_id=exchange_id,
                symbol_id=symbol_id,
                side=side,
                order_type=order_data["type"],
                amount=amount,
                price=price,
                status="filled",
                filled_amount=amount,
                filled_price=price,
                fee=fee,
                stop_loss=order_data["stop_loss"],
                take_profit=order_data["take_profit"],
                client_order_id=order_data["client_order_id"],
                notes=f"Paper trading",
            )
            session.add(order)
            await session.flush()
            order_id = order.id

            # Логирование decision tree
            log_entry = TradeDecisionLog(
                trade_id=0,
                step_order=1,
                step_type="execution",
                description=f"Paper ордер исполнен: {side.upper()} {amount:.4f} {symbol} @ {price:.2f}",
                details={"order_data": order_data},
            )
            session.add(log_entry)
            await session.commit()

        # Создание TradeEvent
        trade_event = TradeEvent(
            type="trade_event",
            trade_id=order_id,
            symbol=symbol,
            direction="long" if side == "buy" else "short",
            entry_price=price,
            exit_price=price,
            amount=amount,
            pnl=0,
            pnl_pct=0,
            holding_seconds=0,
            outcome="pending",
            is_opening=True,
            timestamp=datetime.utcnow().timestamp(),
        )
        await event_bus.publish(trade_event)

        return order

    async def _execute_real_order(self, order_data: dict) -> Optional[Order]:
        """Реальный ордер через биржу."""
        symbol = order_data["symbol"]
        side = order_data["side"]
        amount = order_data["amount"]
        price = order_data["price"]

        try:
            if order_data["type"] == "market":
                if side == "buy":
                    order = await self.exchange.create_market_buy_order(symbol, amount)
                else:
                    order = await self.exchange.create_market_sell_order(symbol, amount)
            elif order_data["type"] == "limit":
                if side == "buy":
                    order = await self.exchange.create_limit_buy_order(symbol, price, amount)
                else:
                    order = await self.exchange.create_limit_sell_order(symbol, price, amount)
            else:
                logger.error(f"Неизвестный тип ордера: {order_data['type']}")
                return None

            logger.info(f"✅ Ордер исполнен на бирже: {order['id']} | {side.upper()} {amount:.4f} {symbol}")

            async with get_session() as session:
                exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
                order_obj = Order(
                    exchange_id=exchange_id,
                    symbol_id=symbol_id,
                    side=side,
                    order_type=order_data["type"],
                    amount=amount,
                    price=price,
                    status="filled",
                    filled_amount=order["filled"] or amount,
                    filled_price=order["price"],
                    fee=order["fee"] or 0,
                    stop_loss=order_data["stop_loss"],
                    take_profit=order_data["take_profit"],
                    order_id_exchange=order["id"],
                    client_order_id=order_data["client_order_id"],
                )
                session.add(order_obj)
                await session.flush()
                order_id = order_obj.id

            trade_event = TradeEvent(
                type="trade_event",
                trade_id=order_id,
                symbol=symbol,
                direction="long" if side == "buy" else "short",
                entry_price=price,
                exit_price=price,
                amount=amount,
                pnl=0,
                pnl_pct=0,
                holding_seconds=0,
                outcome="pending",
                is_opening=True,
                timestamp=datetime.utcnow().timestamp(),
            )
            await event_bus.publish(trade_event)

            return order_obj

        except Exception as e:
            logger.error(f"❌ Ошибка исполнения реального ордера {symbol}: {e}")
            return None

    def can_execute(self) -> bool:
        """Можно ли исполнить ордер?"""
        return not risk_manager.state.kill_switch_active and not risk_manager.state.paused

    def get_paper_balance(self) -> float:
        """Текущий paper баланс."""
        return self.paper_balance

    def get_paper_positions(self) -> dict:
        """Текущие paper позиции."""
        return dict(self.paper_positions)

    async def get_real_balance(self) -> Optional[float]:
        """Получить реальный баланс в USDT."""
        if not self.exchange:
            return None
        try:
            balance = await self.exchange.fetch_balance()
            return sum(v for k, v in balance.items() if isinstance(v, (int, float)) and k.endswith("USDT"))
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None


# Глобальный экземпляр
execution_engine = ExecutionEngine()
