"""Execution Engine — отправка ордеров, исполнение, трекинг."""
import logging
import uuid

import ccxt.async_support as ccxt
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import selectinload

from src.config import settings
from src.db.models import (
    Exchange,
    Order,
    Strategy,
    Symbol,
    TelegramSignal,
    Trade,
    TradeDecisionLog,
)
from src.db.session import get_session
from src.event_bus import (
    TradeEvent,
    event_bus,
)
from src.risk.risk_manager import risk_manager
from src.utils.timeutils import utcnow, utcnow_timestamp

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Движок исполнения ордеров."""

    def __init__(self):
        self.exchange: ccxt.Exchange | None = None
        self.exchange_id: str | None = None
        self.is_paper: bool = settings.is_paper
        self.paper_balance: float = settings.startup_capital_usdt
        self.paper_positions: dict[str, dict] = {}
        self.real_positions: dict[str, dict] = {}
        self.last_prices: dict[str, float] = {}
        self.order_counter = 0

    def get_open_positions(self) -> dict:
        """Открытые позиции для текущего режима (paper или real)."""
        return dict(self.paper_positions if self.is_paper else self.real_positions)

    async def initialize(self, exchange_id: str = "binance"):
        """Инициализация подключения к бирже."""
        self.exchange_id = exchange_id

        if self.is_paper:
            logger.info("📄 Execution Engine: Paper Trading режим")
            await self._restore_paper_state_from_db()
            return

        # Real mode: подключаемся к бирже
        try:
            # Раньше здесь БЕЗУСЛОВНО брались settings.binance_api_key/secret
            # независимо от exchange_id — подключение к Bybit реально шло по
            # Binance-ключам (или падало в paper, если их не было), а
            # собственные ключи Bybit нигде не читались вообще.
            credentials: dict[str, tuple[str | None, str | None, str | None]] = {
                "binance": (settings.binance_api_key, settings.binance_api_secret, None),
                "bybit": (settings.bybit_api_key, settings.bybit_api_secret, None),
                # OKX, в отличие от Binance/Bybit, требует третий секрет
                # (passphrase) в каждом запросе — ccxt называет его "password".
                "okx": (settings.okx_api_key, settings.okx_api_secret, settings.okx_passphrase),
            }
            api_key, api_secret, passphrase = credentials.get(exchange_id, (None, None, None))
            missing_passphrase = exchange_id == "okx" and not passphrase

            if not api_key or not api_secret or missing_passphrase:
                logger.warning(f"⚠️ API ключи {exchange_id} не указаны, переключаемся в paper режим")
                self.is_paper = True
                await self._restore_paper_state_from_db()
                return

            exchange_class = getattr(ccxt, exchange_id, None)
            if exchange_class is None:
                logger.error(f"ccxt не поддерживает биржу '{exchange_id}'")
                self.is_paper = True
                await self._restore_paper_state_from_db()
                return

            exchange_config: dict = {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
            if passphrase:
                exchange_config["password"] = passphrase
            self.exchange = exchange_class(exchange_config)

            if settings.use_exchange_sandbox:
                # Тот же API-ключ, но запросы идут на demo/testnet-счёт биржи
                # вместо реального — ccxt сам подменяет нужные адреса
                # (testnet.binance.vision для Binance, demo-режим Bybit/OKX).
                self.exchange.set_sandbox_mode(True)

            await self.exchange.load_markets()
            logger.info(
                f"🔗 Execution Engine: подключено к {exchange_id}"
                f"{' (демо-счёт)' if settings.use_exchange_sandbox else ' (LIVE, реальные средства)'}"
            )

            await self._warn_if_okx_trade_permission_missing(exchange_id)

            # Проверка баланса
            try:
                balance = await self.exchange.fetch_balance()
                total_usdt = self._extract_usdt_balance(balance)
                self.paper_balance = total_usdt
                logger.info(f"💰 Баланс USDT: {total_usdt:.2f}")
                # Без этого start_balance риск-менеджера оставался на
                # захардкоженном settings.startup_capital_usdt (paper-дефолт)
                # и не совпадал с реальным балансом — сразу после
                # переключения в real это давало ложную просадку (иногда
                # ровно 100%, если баланс к тому же читался как 0 — см.
                # _extract_usdt_balance) и мгновенную паузу торговли.
                risk_manager.reset_for_real_account(total_usdt)
            except Exception as e:
                logger.warning(f"Не удалось получить баланс: {e}")

            await self._restore_real_positions_from_db()

        except Exception as e:
            logger.error(f"Ошибка инициализации биржи {exchange_id}: {e}")
            self.is_paper = True

    async def _warn_if_okx_trade_permission_missing(self, exchange_id: str):
        """
        OKX (в отличие от Binance/Bybit) отдаёт список прав, реально выданных
        API-ключу, прямо в GET /account/config — поле "perm", например
        "read_only,trade". Раньше отсутствие права "trade" обнаруживалось
        только когда первый реальный ордер падал с 50123 "API Key does not
        have trading permission" — иногда через часы после подключения.
        Проверка read-only, поэтому срабатывает даже если ключу не хватает
        именно "trade" (сам запрос его не требует).
        """
        if exchange_id != "okx":
            return
        try:
            accounts = await self.exchange.fetch_accounts()
            perm = (accounts[0]["info"].get("perm") or "") if accounts else ""
            if perm and "trade" not in perm.split(","):
                logger.warning(
                    f"⚠️ OKX API-ключ без права 'Trade' (текущие права: {perm}). "
                    f"Реальные ордера будут отклоняться биржей — включите Trade "
                    f"permission в OKX API Management "
                    f"({'Demo Trading' if settings.use_exchange_sandbox else 'обычный'} ключ)."
                )
        except Exception as e:
            logger.debug(f"Не удалось проверить права OKX API-ключа: {e}")

    async def _load_open_positions_from_db(
        self, is_paper: bool,
    ) -> tuple[dict[str, dict] | None, float, float]:
        """
        Реконструировать открытые позиции из БД для указанного режима (paper/real).

        Открытая позиция = ордер (status=filled) на бирже нужного типа
        (Order.exchange -> Exchange.is_paper), у которого объём ещё не
        полностью выбран закрывающими сделками. BUY-ордер открывает LONG,
        SELL-ордер открывает SHORT — раньше здесь смотрели только на
        side=="buy", из-за чего любая открытая SHORT-позиция при рестарте
        бота молча пропадала из paper_positions/real_positions (без единой
        закрывающей Trade-записи — она никогда не появлялась и в истории
        закрытых сделок). Позиция может быть закрыта ЧАСТИЧНО (уровни
        TP1/TP2) — остаток реконструируется как filled_amount минус сумма
        Trade.amount всех сделок, ссылающихся на этот ордер как
        order_open_id, а число таких сделок — это tp_hit_count (сколько
        уровней TP уже сработало). order_close_id из этого же расчёта
        исключается отдельно: закрытие SHORT создаёт BUY-ордер, а закрытие
        LONG — SELL-ордер (см. close_paper_position/close_real_position),
        которые иначе неотличимы от новой открытой позиции противоположной
        стороны.
        Возвращает (позиции, реализованный PnL, себестоимость открытых позиций)
        или (None, 0, 0) при ошибке запроса.
        """
        try:
            async with get_session() as session:
                close_order_ids = set(
                    (
                        await session.execute(
                            select(Trade.order_close_id).where(Trade.order_close_id.is_not(None))
                        )
                    ).scalars().all()
                )
                partial_trades = (
                    await session.execute(
                        select(Trade.order_open_id, Trade.amount)
                        .where(Trade.order_open_id.is_not(None))
                    )
                ).all()
                closed_by_order: dict[int, float] = {}
                hits_by_order: dict[int, int] = {}
                for order_open_id, trade_amount in partial_trades:
                    closed_by_order[order_open_id] = closed_by_order.get(order_open_id, 0.0) + float(trade_amount)
                    hits_by_order[order_open_id] = hits_by_order.get(order_open_id, 0) + 1

                # PnL и открытые ордера должны считаться в рамках одного режима —
                # иначе, например, реальный PnL просачивался бы в paper-баланс.
                realized_pnl = (
                    await session.execute(
                        select(func.sum(Trade.pnl))
                        .join(Symbol, Trade.symbol_id == Symbol.id)
                        .join(Exchange, Symbol.exchange_id == Exchange.id)
                        .where(Exchange.is_paper == is_paper)
                    )
                ).scalar() or 0

                orders = (
                    await session.execute(
                        select(Order)
                        .join(Exchange, Order.exchange_id == Exchange.id)
                        .options(selectinload(Order.symbol), selectinload(Order.strategy))
                        .where(
                            Order.side.in_(("buy", "sell")),
                            Order.status == "filled",
                            Exchange.is_paper == is_paper,
                        )
                        .order_by(Order.created_at.asc())
                    )
                ).scalars().all()
        except Exception as e:
            logger.warning(f"Не удалось восстановить открытые позиции из БД: {e}")
            return None, 0.0, 0.0

        positions: dict[str, dict] = {}
        cost_basis = 0.0
        for o in orders:
            if o.id in close_order_ids or o.symbol is None:
                continue

            filled_amount = float(o.filled_amount or o.amount)
            already_closed = closed_by_order.get(o.id, 0.0)
            amount = filled_amount - already_closed
            if amount <= 1e-9:
                continue  # эта открывающая позиция уже закрыта полностью

            symbol = o.symbol.symbol
            price = float(o.filled_price or o.price)
            position_side = "long" if o.side == "buy" else "short"
            # Комиссия открытия относится к остатку объёма пропорционально —
            # иначе для частично закрытой позиции она либо задваивалась бы
            # (уже учтена в PnL прошлых частичных закрытий), либо терялась.
            fee = float(o.fee or 0) * (amount / filled_amount) if filled_amount else 0.0
            # cost_basis (спишется с paper_balance ниже) осмыслен только для
            # long — открытие long списывает amount*price+fee с баланса,
            # открытие short маржу не резервирует (см. _execute_paper_order).
            if position_side == "long":
                cost_basis += amount * price + fee

            pos = positions.setdefault(symbol, {
                "amount": 0.0, "entry_price": 0.0, "side": position_side,
                "strategy_id": None, "stop_loss": None, "take_profit": None,
                "order_id": None, "entry_fee": 0.0, "opened_at": None,
                "tp_hit_count": 0,
            })
            pos["entry_price"] = (
                (pos["entry_price"] * pos["amount"] + price * amount) / (pos["amount"] + amount)
                if (pos["amount"] + amount) else price
            )
            pos["amount"] += amount
            pos["side"] = position_side
            pos["strategy_id"] = o.strategy.name if o.strategy else pos["strategy_id"]
            pos["stop_loss"] = float(o.stop_loss) if o.stop_loss else pos["stop_loss"]
            pos["take_profit"] = float(o.take_profit) if o.take_profit else pos["take_profit"]
            pos["order_id"] = o.id
            pos["entry_fee"] = fee
            pos["tp_hit_count"] = max(pos["tp_hit_count"], hits_by_order.get(o.id, 0))
            if pos["opened_at"] is None:
                pos["opened_at"] = o.created_at

        # Если хотя бы один уровень TP уже сработал, SL остатка позиции был
        # передвинут в безубыток (см. _check_position_exit в main.py) — это
        # решение никогда не пишется в Order.stop_loss в БД, поэтому
        # применяем то же правило здесь, а не восстанавливаем исходный SL.
        for pos in positions.values():
            if pos["tp_hit_count"] >= 1:
                pos["stop_loss"] = pos["entry_price"]

        return positions, float(realized_pnl), cost_basis

    async def _restore_paper_state_from_db(self):
        """
        Восстановить paper_balance и paper_positions из БД при старте процесса.

        paper_positions/paper_balance раньше существовали только в памяти —
        каждый рестарт бота (в т.ч. через кнопку в дашборде) молча обнулял
        баланс до startup_capital_usdt и "терял" все открытые позиции, хотя
        в БД (Order/Trade) вся история оставалась цела.
        """
        self.paper_balance = settings.startup_capital_usdt
        self.paper_positions = {}

        positions, realized_pnl, cost_basis = await self._load_open_positions_from_db(is_paper=True)
        if positions is None:
            return

        self.paper_positions = positions
        self.paper_balance = settings.startup_capital_usdt + realized_pnl - cost_basis

        if self.paper_positions:
            logger.info(
                f"♻️ Восстановлено {len(self.paper_positions)} открытых paper-позиций из БД: "
                f"{list(self.paper_positions.keys())} | баланс: {self.paper_balance:.2f}"
            )
        else:
            logger.info(f"📄 Открытых позиций в БД нет | баланс: {self.paper_balance:.2f}")

    async def _restore_real_positions_from_db(self):
        """
        Восстановить real_positions из БД при старте процесса (см.
        _restore_paper_state_from_db — тот же смысл, для реального режима).
        Баланс для real режима не реконструируется — он всегда берётся
        напрямую с биржи через fetch_balance().
        """
        self.real_positions = {}

        positions, _, _ = await self._load_open_positions_from_db(is_paper=False)
        if positions is None:
            return

        self.real_positions = positions

        if self.real_positions:
            logger.info(
                f"♻️ Восстановлено {len(self.real_positions)} открытых реальных позиций из БД: "
                f"{list(self.real_positions.keys())}"
            )
        else:
            logger.info("Открытых реальных позиций в БД нет")

    async def reset_paper_account(self) -> dict:
        """
        Полностью сбросить paper-аккаунт: удалить всю историю paper-ордеров
        и сделок (только для paper-бирж — реальные данные не затрагиваются,
        они хранятся под отдельным Exchange-рядом, см. _resolve_symbol_id)
        и вернуть баланс к startup_capital_usdt. Нужно, когда накопленная
        история — результат уже исправленных багов (див. dev-changelog) и
        нет смысла держать её как базу для расчёта просадки/win rate.

        Возвращает {"orders_deleted", "trades_deleted"}.
        """
        async with get_session() as session:
            paper_exchange_ids = [
                e.id for e in (
                    await session.execute(select(Exchange).where(Exchange.is_paper == True))  # noqa: E712
                ).scalars().all()
            ]
            if not paper_exchange_ids:
                order_ids, trade_ids = [], []
            else:
                symbol_ids = [
                    s.id for s in (
                        await session.execute(
                            select(Symbol).where(Symbol.exchange_id.in_(paper_exchange_ids))
                        )
                    ).scalars().all()
                ]
                order_ids = [
                    o.id for o in (
                        await session.execute(
                            select(Order).where(Order.exchange_id.in_(paper_exchange_ids))
                        )
                    ).scalars().all()
                ]
                trade_ids = [
                    t.id for t in (
                        await session.execute(
                            select(Trade).where(Trade.symbol_id.in_(symbol_ids))
                        )
                    ).scalars().all()
                ] if symbol_ids else []

                # Telegram-сигналы отвязываем, а не удаляем — сырое
                # сообщение и решение бота остаются в истории канала,
                # просто теряют ссылку на удалённые ордер/сделку.
                if order_ids:
                    await session.execute(
                        update(TelegramSignal)
                        .where(TelegramSignal.executed_order_id.in_(order_ids))
                        .values(executed_order_id=None)
                    )
                if trade_ids:
                    await session.execute(
                        update(TelegramSignal)
                        .where(TelegramSignal.executed_trade_id.in_(trade_ids))
                        .values(executed_trade_id=None)
                    )
                    await session.execute(
                        delete(TradeDecisionLog).where(TradeDecisionLog.trade_id.in_(trade_ids))
                    )
                    await session.execute(delete(Trade).where(Trade.id.in_(trade_ids)))
                if order_ids:
                    await session.execute(delete(Order).where(Order.id.in_(order_ids)))

            await session.commit()

        self.paper_positions = {}
        self.paper_balance = settings.startup_capital_usdt

        logger.warning(
            f"🔄 Paper-аккаунт сброшен: удалено ордеров={len(order_ids)}, сделок={len(trade_ids)}, "
            f"баланс возвращён к {self.paper_balance:.2f}"
        )
        return {"orders_deleted": len(order_ids), "trades_deleted": len(trade_ids)}

    async def close(self):
        """Закрыть соединение с биржей."""
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Execution Engine: соединение закрыто")

    async def _resolve_symbol_id(self, session, symbol: str) -> tuple[int, int]:
        """Получить (или создать) id биржи и торговой пары в БД."""
        base_name = self.exchange_id or "paper"
        # Exchange.name уникально, а paper- и real-исполнение используют один
        # и тот же exchange_id (например "binance") — раньше это заставляло
        # оба режима писать ордера в один и тот же Exchange-ряд, чей is_paper
        # отражал только то, какой режим создал ряд первым (обычно paper).
        # Разносим по имени, иначе восстановление открытых позиций не может
        # отличить paper-ордера от реальных.
        exchange_name = base_name if not self.is_paper else f"{base_name}_paper"

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

    async def _resolve_strategy_id(self, session, strategy_name: str | None) -> int | None:
        """Получить (или создать) id стратегии в БД по её строковому идентификатору."""
        if not strategy_name:
            return None

        strategy_row = (
            await session.execute(select(Strategy).where(Strategy.name == strategy_name))
        ).scalar_one_or_none()
        if strategy_row is None:
            strategy_row = Strategy(name=strategy_name, strategy_type="rule")
            session.add(strategy_row)
            await session.flush()

        return strategy_row.id

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float | None = None,
        order_type: str = "market",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy_id: int | None = None,
        signal_data: dict | None = None,
    ) -> Order | None:
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

        sl_str = f"{stop_loss:.2f}" if stop_loss is not None else "—"
        tp_str = f"{take_profit:.2f}" if take_profit is not None else "—"
        logger.info(
            f"📝 Создание ордера: {side.upper()} {amount:.4f} {symbol} @ {execution_price:.2f} | "
            f"SL={sl_str} TP={tp_str} | ID={client_order_id}"
        )

        if self.is_paper:
            return await self._execute_paper_order(order_data)
        else:
            return await self._execute_real_order(order_data)

    async def _execute_paper_order(self, order_data: dict) -> Order | None:
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
        is_opening_order = False
        if side == "buy":
            self.paper_balance -= position_value + fee
            if symbol not in self.paper_positions:
                self.paper_positions[symbol] = {"amount": 0, "entry_price": 0, "side": "long"}
            pos = self.paper_positions[symbol]
            # Средневзвешенная цена входа считается по СТАРОМУ количеству —
            # раньше формула брала уже обновлённое (amount+=amount) значение
            # как вес старой цены, задваивая её вклад при любой доливке
            # позиции (напр. 1@100 + 1@200 давало entry_price=200 вместо
            # верных 150).
            new_amount = pos["amount"] + amount
            pos["entry_price"] = (pos["entry_price"] * pos["amount"] + price * amount) / new_amount
            pos["amount"] = new_amount
            pos["side"] = "long"
            # Источник сигнала, SL/TP — для отображения и ручного закрытия
            # в дашборде; при доливке позиции другим ордером отражают
            # последний ордер, не историю целиком.
            pos["strategy_id"] = order_data.get("strategy_id")
            pos["stop_loss"] = order_data.get("stop_loss")
            pos["take_profit"] = order_data.get("take_profit")
            is_opening_order = True
        else:
            if symbol in self.paper_positions and self.paper_positions[symbol].get("side") == "long":
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
            elif symbol not in self.paper_positions:
                # Открытие SHORT-позиции. Раньше этот путь не делал вообще
                # ничего (ни self.paper_positions, ни self.paper_balance не
                # менялись), но код ниже всё равно логировал "ордер
                # исполнен", создавал Order со статусом filled и публиковал
                # TradeEvent — то есть КАЖДЫЙ short-сигнал от любой стратегии
                # тихо становился фантомным ордером без реальной позиции.
                # Как и close_paper_position() для шортов, маржа при
                # открытии не резервируется — баланс корректируется только
                # на реализованный PnL в момент закрытия (это уже
                # заложенное упрощение модели, не новое поведение).
                self.paper_positions[symbol] = {
                    "amount": amount, "entry_price": price, "side": "short",
                    "strategy_id": order_data.get("strategy_id"),
                    "stop_loss": order_data.get("stop_loss"),
                    "take_profit": order_data.get("take_profit"),
                }
                is_opening_order = True
            else:
                logger.warning(
                    f"Paper: уже есть {self.paper_positions[symbol].get('side')}-позиция {symbol}, "
                    f"доливка/переворот сейчас не поддерживается"
                )
                return None

        logger.info(f"📄 Paper ордер исполнен: {side.upper()} {amount:.4f} {symbol} @ {price:.2f}")

        # Создание Order объекта в БД
        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            strategy_db_id = await self._resolve_strategy_id(session, order_data.get("strategy_id"))
            order = Order(
                exchange_id=exchange_id,
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
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
                notes="Paper trading",
            )
            session.add(order)
            await session.flush()
            order_id = order.id
            await session.commit()

        # order_id/entry_fee/opened_at нужны только для ордеров, которые
        # ОТКРЫВАЮТ позицию (buy-long или sell-short), не для тех, что её
        # уменьшают/закрывают.
        if is_opening_order and symbol in self.paper_positions:
            pos = self.paper_positions[symbol]
            pos["order_id"] = order_id
            pos["entry_fee"] = fee
            pos.setdefault("opened_at", utcnow())

        # Создание TradeEvent — только для ордеров, которые реально
        # открывают позицию; частичное/полное закрытие long через plain
        # sell (доливка телеграм-сигналом против существующей позиции)
        # раньше тоже публиковало is_opening=True с direction="short",
        # что выглядело как открытие короткой позиции в уведомлениях,
        # хотя на деле длинная позиция просто уменьшалась.
        if is_opening_order:
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
                timestamp=utcnow_timestamp(),
            )
            await event_bus.publish(trade_event)

        return order

    async def close_paper_position(
        self,
        symbol: str,
        side: str,  # long, short
        entry_price: float,
        amount: float,
        exit_price: float,
        reason: str,
        entry_fee: float = 0.0,
        holding_seconds: int = 0,
        strategy_id: str | None = None,
        order_open_id: int | None = None,
    ) -> dict | None:
        """
        Закрыть paper-позицию: посчитать PnL, обновить баланс, записать закрывающий
        Order + Trade в БД, опубликовать закрывающий TradeEvent.
        Возвращает {"pnl", "pnl_pct", "outcome"} или None при ошибке.
        """
        fee_pct = settings.paper_fee_pct / 100
        exit_fee = amount * exit_price * fee_pct

        if side == "long":
            pnl = (exit_price - entry_price) * amount - entry_fee - exit_fee
        else:
            pnl = (entry_price - exit_price) * amount - entry_fee - exit_fee

        pnl_pct = (pnl / (entry_price * amount) * 100) if entry_price and amount else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "break-even")

        # Баланс и позиция мутировались здесь, ДО записи в БД — если запись
        # падала (сетевой сбой к Postgres, что угодно), в памяти позиция уже
        # исчезала (или уменьшалась) без единой строки в Order/Trade, а
        # main.py на следующей итерации видел, что символа нет в
        # paper_positions, и тихо удалял его из своего open_positions —
        # позиция просто пропадала, не появляясь ни в открытых, ни в
        # закрытых. Мутируем только после успешного commit.
        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            strategy_db_id = await self._resolve_strategy_id(session, strategy_id)

            close_order = Order(
                exchange_id=exchange_id,
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                side="sell" if side == "long" else "buy",
                order_type="market",
                amount=amount,
                price=exit_price,
                status="filled",
                filled_amount=amount,
                filled_price=exit_price,
                fee=exit_fee,
                client_order_id=str(uuid.uuid4())[:12],
                notes=f"Paper close ({reason})",
            )
            session.add(close_order)
            await session.flush()

            trade = Trade(
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                order_open_id=order_open_id,
                order_close_id=close_order.id,
                direction=side,
                entry_price=entry_price,
                exit_price=exit_price,
                amount=amount,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_seconds=holding_seconds,
                outcome=outcome,
                is_open=False,
                closed_at=utcnow(),
            )
            session.add(trade)
            await session.commit()
            trade_id = trade.id

        # Запись в БД прошла успешно — теперь можно безопасно применить
        # эффект к живому состоянию.
        if side == "long":
            # LONG при открытии списывает принципал с баланса — при закрытии
            # возвращаем выручку от продажи (а не только pnl)
            self.paper_balance += amount * exit_price - exit_fee
        else:
            # SHORT в paper-режиме сейчас не резервирует маржу при открытии
            # (создание короткой позиции без встречной длинной не отслеживается
            # в self.paper_positions) — учитываем только реализованный PnL
            self.paper_balance += pnl

        # amount может быть частью открытой позиции (частичное закрытие по
        # уровню TP1/TP2) — уменьшаем остаток вместо того, чтобы стереть
        # позицию целиком, иначе main.py на следующей итерации решил бы,
        # что оставшаяся часть закрыта в обход основного цикла, и потерял
        # бы её отслеживание.
        pos = self.paper_positions.get(symbol)
        if pos is not None:
            remaining = pos["amount"] - amount
            if remaining <= 1e-9:
                self.paper_positions.pop(symbol, None)
            else:
                pos["amount"] = remaining

        logger.info(
            f"📄 Paper позиция закрыта: {symbol} {side.upper()} | {reason} | "
            f"PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)"
        )

        trade_event = TradeEvent(
            type="trade_event",
            trade_id=trade_id,
            symbol=symbol,
            direction=side,
            entry_price=entry_price,
            exit_price=exit_price,
            amount=amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_seconds=holding_seconds,
            outcome=outcome,
            is_opening=False,
            timestamp=utcnow_timestamp(),
        )
        await event_bus.publish(trade_event)

        return {"pnl": pnl, "pnl_pct": pnl_pct, "outcome": outcome, "trade_id": trade_id}

    async def _execute_real_order(self, order_data: dict) -> Order | None:
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

            # ccxt возвращает order["fee"] как dict {"cost": ..., "currency": ...}
            # (или None), а не число — писать его напрямую в DECIMAL-колонку
            # было ошибкой. Аналогично order["price"] у маркет-ордеров обычно
            # None (заполняется только order["average"]).
            fill_price = order.get("average") or order.get("price") or price
            fill_fee = (order.get("fee") or {}).get("cost") or 0

            async with get_session() as session:
                exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
                strategy_db_id = await self._resolve_strategy_id(session, order_data.get("strategy_id"))
                order_obj = Order(
                    exchange_id=exchange_id,
                    symbol_id=symbol_id,
                    strategy_id=strategy_db_id,
                    side=side,
                    order_type=order_data["type"],
                    amount=amount,
                    price=price,
                    status="filled",
                    filled_amount=order["filled"] or amount,
                    filled_price=fill_price,
                    fee=fill_fee,
                    stop_loss=order_data["stop_loss"],
                    take_profit=order_data["take_profit"],
                    order_id_exchange=order["id"],
                    client_order_id=order_data["client_order_id"],
                )
                session.add(order_obj)
                await session.flush()
                order_id = order_obj.id

            # Регистрируем открытую реальную позицию так же, как paper —
            # без этого SL/TP по ней никогда не проверялись бы (см.
            # close_real_position / _check_position_exit в main.py), а
            # закрыть её вручную из дашборда было бы нельзя.
            if side == "buy":
                self.real_positions[symbol] = {
                    "amount": amount,
                    "entry_price": fill_price,
                    "side": "long",
                    "strategy_id": order_data.get("strategy_id"),
                    "stop_loss": order_data.get("stop_loss"),
                    "take_profit": order_data.get("take_profit"),
                    "order_id": order_id,
                    "entry_fee": fill_fee,
                    "opened_at": utcnow(),
                }

            trade_event = TradeEvent(
                type="trade_event",
                trade_id=order_id,
                symbol=symbol,
                direction="long" if side == "buy" else "short",
                entry_price=fill_price,
                exit_price=fill_price,
                amount=amount,
                pnl=0,
                pnl_pct=0,
                holding_seconds=0,
                outcome="pending",
                is_opening=True,
                timestamp=utcnow_timestamp(),
            )
            await event_bus.publish(trade_event)

            return order_obj

        except Exception as e:
            logger.error(f"❌ Ошибка исполнения реального ордера {symbol}: {e}")
            return None

    async def close_real_position(
        self,
        symbol: str,
        side: str,  # long, short
        entry_price: float,
        amount: float,
        reason: str,
        entry_fee: float = 0.0,
        holding_seconds: int = 0,
        strategy_id: str | None = None,
        order_open_id: int | None = None,
        **_ignored,
    ) -> dict | None:
        """
        Закрыть реальную позицию рыночным ордером на бирже: посчитать PnL по
        фактической цене исполнения, записать закрывающий Order + Trade в БД,
        опубликовать закрывающий TradeEvent. Возвращает {"pnl", "pnl_pct",
        "outcome", "trade_id"} или None при ошибке.

        Только long: на споте нет встроенного шорта, а _execute_real_order
        уже не позволяет открыть short-позицию (create_market_sell_order без
        имеющегося актива на споте просто упадёт с ошибкой недостатка
        баланса — исполнение вернёт None и позиция никогда не будет создана).
        """
        if side != "long":
            logger.error(f"close_real_position: закрытие {side}-позиции не поддерживается на споте: {symbol}")
            return None

        try:
            order = await self.exchange.create_market_sell_order(symbol, amount)
        except Exception as e:
            logger.error(f"❌ Не удалось закрыть реальную позицию {symbol}: {e}")
            return None

        exit_price = order.get("average") or order.get("price") or entry_price
        exit_fee = (order.get("fee") or {}).get("cost") or 0

        pnl = (exit_price - entry_price) * amount - entry_fee - exit_fee
        pnl_pct = (pnl / (entry_price * amount) * 100) if entry_price and amount else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "break-even")

        # В отличие от close_paper_position, здесь позиция мутируется ДО
        # записи в БД — намеренно: рыночный ордер на бирже (строкой выше)
        # уже необратимо исполнился с реальными деньгами. Если бы мы
        # отложили обновление self.real_positions до записи в БД и запись
        # упала, следующая проверка увидела бы прежний (ещё не уменьшенный)
        # объём и попыталась бы продать его снова — риск повторной продажи
        # того, чего уже нет на балансе биржи. Реальный "провал записи в
        # БД при уже исполненном на бирже ордере" остаётся редким
        # ручным edge case для расследования по логам, а не автоматически
        # устранимым состоянием.
        #
        # amount может быть частью открытой позиции (частичное закрытие по
        # уровню TP1/TP2) — см. тот же комментарий в close_paper_position.
        pos = self.real_positions.get(symbol)
        if pos is not None:
            remaining = pos["amount"] - amount
            if remaining <= 1e-9:
                self.real_positions.pop(symbol, None)
            else:
                pos["amount"] = remaining

        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            strategy_db_id = await self._resolve_strategy_id(session, strategy_id)

            close_order = Order(
                exchange_id=exchange_id,
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                side="sell",
                order_type="market",
                amount=amount,
                price=exit_price,
                status="filled",
                filled_amount=order["filled"] or amount,
                filled_price=exit_price,
                fee=exit_fee,
                order_id_exchange=order["id"],
                client_order_id=str(uuid.uuid4())[:12],
                notes=f"Real close ({reason})",
            )
            session.add(close_order)
            await session.flush()

            trade = Trade(
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                order_open_id=order_open_id,
                order_close_id=close_order.id,
                direction=side,
                entry_price=entry_price,
                exit_price=exit_price,
                amount=amount,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_seconds=holding_seconds,
                outcome=outcome,
                is_open=False,
                closed_at=utcnow(),
            )
            session.add(trade)
            await session.commit()
            trade_id = trade.id

        logger.info(
            f"💰 Реальная позиция закрыта: {symbol} {side.upper()} | {reason} | "
            f"PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)"
        )

        trade_event = TradeEvent(
            type="trade_event",
            trade_id=trade_id,
            symbol=symbol,
            direction=side,
            entry_price=entry_price,
            exit_price=exit_price,
            amount=amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_seconds=holding_seconds,
            outcome=outcome,
            is_opening=False,
            timestamp=utcnow_timestamp(),
        )
        await event_bus.publish(trade_event)

        return {"pnl": pnl, "pnl_pct": pnl_pct, "outcome": outcome, "trade_id": trade_id}

    def can_execute(self) -> bool:
        """Можно ли исполнить ордер?"""
        return not risk_manager.state.kill_switch_active and not risk_manager.state.paused

    def get_paper_balance(self) -> float:
        """Текущий paper баланс."""
        return self.paper_balance

    def get_paper_positions(self) -> dict:
        """Текущие paper позиции."""
        return dict(self.paper_positions)

    async def get_real_balance(self) -> float | None:
        """Получить реальный баланс в USDT."""
        if not self.exchange:
            return None
        try:
            balance = await self.exchange.fetch_balance()
            return self._extract_usdt_balance(balance)
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None

    @staticmethod
    def _extract_usdt_balance(balance: dict) -> float:
        """
        ccxt fetch_balance() кладёт баланс валюты в ВЛОЖЕННЫЙ словарь:
        balance['free']['USDT'] (плоское число) и/или balance['USDT'] =
        {'free':, 'used':, 'total':} (сам по себе словарь, а НЕ число).
        Старый код фильтровал `isinstance(v, (int, float))` по
        balance.items() верхнего уровня — balance['USDT'] там всегда
        словарь, значит проверка никогда не проходила, и баланс всегда
        читался как 0 независимо от биржи и реального остатка на счёте.
        """
        free = balance.get("free") or {}
        value = free.get("USDT")
        if value is None:
            usdt = balance.get("USDT")
            if isinstance(usdt, dict):
                value = usdt.get("free")
        return float(value or 0.0)


# Глобальный экземпляр
execution_engine = ExecutionEngine()
