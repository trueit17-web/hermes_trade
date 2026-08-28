"""Execution Engine — отправка ордеров, исполнение, трекинг."""
import asyncio
import logging
import uuid
from datetime import UTC

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
                if exchange_id == "bybit":
                    # У Bybit это НЕ то же самое, что set_sandbox_mode.
                    # set_sandbox_mode(True) шлёт запросы на testnet.bybit.com —
                    # отдельная песочница со своей регистрацией и своими
                    # ключами. Ключ, который пользователь создаёт как "демо
                    # счёт" через переключатель Demo Trading в обычном
                    # live-аккаунте (обычный путь для retail), живёт на
                    # api-demo.bybit.com и требует отдельного метода
                    # enable_demo_trading() — иначе тот же самый, реально
                    # валидный ключ отправлялся на testnet, который его не
                    # знает, и Bybit отвечал "API key is invalid" (retCode
                    # 10003). Методы взаимоисключающие: enable_demo_trading()
                    # падает с NotSupported, если до этого уже включён
                    # set_sandbox_mode.
                    self.exchange.enable_demo_trading(True)
                else:
                    # Тот же API-ключ, но запросы идут на demo/testnet-счёт
                    # биржи вместо реального — ccxt сам подменяет нужные
                    # адреса (testnet.binance.vision для Binance, demo-режим
                    # OKX).
                    self.exchange.set_sandbox_mode(True)

            await self.exchange.load_markets()
            logger.info(
                f"🔗 Execution Engine: подключено к {exchange_id}"
                f"{' (демо-счёт)' if settings.use_exchange_sandbox else ' (LIVE, реальные средства)'}"
            )

            await self._warn_if_okx_trade_permission_missing(exchange_id)

            # Восстанавливаем позиции ДО расчёта базы для просадки — иначе
            # (см. ниже) на счету с уже открытыми real-позициями базой для
            # drawdown становился один только свободный кэш.
            await self._restore_real_positions_from_db()
            await self._rearm_stop_loss_orders_after_restart()

            # Проверка баланса
            try:
                balance = await self.exchange.fetch_balance()
                total_usdt = self._extract_usdt_balance(balance)
                self.paper_balance = total_usdt
                logger.info(f"💰 Свободный баланс USDT: {total_usdt:.2f}")
                # Без этого start_balance риск-менеджера оставался на
                # захардкоженном settings.startup_capital_usdt (paper-дефолт)
                # и не совпадал с реальным балансом — сразу после
                # переключения в real это давало ложную просадку (иногда
                # ровно 100%, если баланс к тому же читался как 0 — см.
                # _extract_usdt_balance) и мгновенную паузу торговли.
                #
                # total_usdt — это ТОЛЬКО свободный (неинвестированный) кэш с
                # биржи. Если на счету уже есть открытые real-позиции (обычный
                # случай при каждом рестарте процесса, не только "первое
                # подключение"), это давало обратный перекос: start_balance
                # считался от голого кэша (например $12.79), а
                # _compute_equity() дальше в main.py каждую итерацию считает
                # cash + стоимость позиций (например $7156) — сравнение
                # несопоставимых величин превращало "просадку" в дашборде в
                # гигантский фиктивный "профит" вида -55830%. База для
                # просадки должна быть той же величиной, что и текущий equity:
                # кэш + стоимость уже восстановленных позиций по цене входа
                # (текущая рыночная цена ещё не известна на этом шаге —
                # entry_price здесь так же консервативен, как fallback в
                # main.py._compute_equity).
                positions_value = sum(
                    pos["amount"] * pos["entry_price"]
                    for pos in self.real_positions.values()
                    if pos.get("side") == "long"
                )
                risk_manager.reset_for_real_account(total_usdt + positions_value)
            except Exception as e:
                logger.warning(f"Не удалось получить баланс: {e}")

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
            # На споте нет встроенного шорта — real-режим больше не создаёт
            # такие ордера (см. защиту в _execute_real_order), но старые
            # "осиротевшие" sell-ордера могли остаться в БД от бага,
            # который эту защиту не имел (реальный инцидент: ENA/USDT,
            # bb_strategy — короткий сигнал реально исполнился на бирже,
            # т.к. на счету оказался остаток актива не от этого бота, и
            # "позицию" потом было принципиально невозможно закрыть —
            # close_real_position отдельно отклоняет side != long). Не
            # реконструируем такую позицию заново при каждом рестарте —
            # сам факт её появления уже исправлен, а восстановление только
            # заново засоряло бы логи той же неустранимой ошибкой закрытия.
            if position_side == "short" and not is_paper:
                logger.warning(
                    f"⚠️ Пропущен осиротевший short-ордер {symbol} (id={o.id}) при восстановлении "
                    f"real-позиций — на споте шорт не поддерживается, закрыть такую 'позицию' всё "
                    f"равно невозможно."
                )
                continue
            # Комиссия открытия относится к остатку объёма пропорционально —
            # иначе для частично закрытой позиции она либо задваивалась бы
            # (уже учтена в PnL прошлых частичных закрытий), либо терялась.
            fee = float(o.fee or 0) * (amount / filled_amount) if filled_amount else 0.0
            # В real-режиме комиссия покупки на споте по умолчанию списывается
            # из самого купленного актива (в отличие от paper, где комиссия
            # условная и списывается только с cash-баланса, не уменьшая
            # количество) — без вычета восстановленный после рестарта остаток
            # позиции оказывался больше, чем реально лежит на бирже, и первая
            # же попытка его закрыть падала с "Insufficient balance".
            if position_side == "long" and not is_paper:
                amount = max(0.0, amount - fee)
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

    async def _rearm_stop_loss_orders_after_restart(self):
        """
        После рестарта процесса связь real_positions[symbol]["sl_order_id"] с
        уже выставленным на бирже условным SL-ордером теряется — БД хранит
        только сам stop_loss (цену в Order.stop_loss), а не ID биржевого
        ордера. Чтобы не накапливать дубликаты условных ордеров при каждом
        рестарте, сначала отменяем ВСЕ незакрытые условные ('tpslOrder')
        ордера по символу, затем ставим один новый под актуальный
        остаток/цену. Best-effort, как и вся остальная работа с биржевыми
        SL-ордерами в этом классе — сбой здесь не должен мешать запуску.
        """
        for symbol, pos in list(self.real_positions.items()):
            stop_loss = pos.get("stop_loss")
            amount = pos.get("amount") or 0
            if not stop_loss or amount <= 0:
                continue
            try:
                open_orders = await self.exchange.fetch_open_orders(symbol, params={"orderFilter": "tpslOrder"})
                for o in (open_orders or []):
                    await self._cancel_order_safe(symbol, o.get("id"))
            except Exception as e:
                logger.debug(f"Не удалось получить список условных ордеров {symbol} перед переустановкой SL: {e}")
            await self.sync_stop_loss_order(symbol, amount, stop_loss)

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

    def _market_limits(self, symbol: str) -> dict | None:
        """
        market["limits"] для symbol, если оно есть и имеет ожидаемую форму
        словаря — иначе None. Общий защитный доступ для
        _below_exchange_minimum/reconcile_real_positions: структура
        markets[symbol] не гарантирована (разные биржи, тестовые mock-объекты
        без выставленного .markets), а падать из-за необязательной проверки
        не должны ни отправка ордера, ни сверка позиций.
        """
        try:
            markets = self.exchange.markets if self.exchange else None
            if not isinstance(markets, dict):
                return None
            market = markets.get(symbol)
            if not isinstance(market, dict):
                return None
            limits = market.get("limits")
            return limits if isinstance(limits, dict) else None
        except Exception:
            return None

    def _market_min_amount(self, symbol: str) -> float | None:
        limits = self._market_limits(symbol)
        if not limits:
            return None
        amount_limits = limits.get("amount")
        min_amount = amount_limits.get("min") if isinstance(amount_limits, dict) else None
        return min_amount if isinstance(min_amount, (int, float)) else None

    def _below_exchange_minimum(self, symbol: str, amount: float, price: float | None) -> str | None:
        """
        Проверить объём/стоимость ордера ПРОТИВ биржевых лимитов пары ДО
        отправки запроса — иначе биржа отклоняет ордер (напр. Bybit
        retCode 170140 "Order value exceeded lower limit"), это летит в
        логи как ERROR, будто сломался код, а на деле объём просто
        занижен: посчитанного от текущего доступного баланса (size_pct%
        от него) размера позиции не хватает даже на минимальный
        допустимый на бирже ордер по этой паре — сигнал безопасно
        пропускается, ошибка это ожидаемая при малом остатке средств.

        Это вспомогательная, необязательная проверка: структура markets[symbol]
        не гарантирована (разные биржи, неполные тестовые/тестовые-mock
        объекты) — при любой неожиданности просто не блокируем ордер,
        оставляя решение самой бирже, как и раньше.
        """
        try:
            min_amount = self._market_min_amount(symbol)
            if isinstance(min_amount, (int, float)) and amount < min_amount:
                return f"объём {amount:.8f} {symbol.split('/')[0]} меньше минимального ({min_amount})"
            limits = self._market_limits(symbol) or {}
            cost_limits = limits.get("cost")
            min_cost = cost_limits.get("min") if isinstance(cost_limits, dict) else None
            if isinstance(min_cost, (int, float)) and price and amount * price < min_cost:
                return f"стоимость ордера {amount * price:.4f} USDT меньше минимальной по паре ({min_cost} USDT)"
        except Exception as e:
            logger.debug(f"Не удалось проверить минимальные лимиты биржи для {symbol}: {e}")
        return None

    async def _fetch_fill_details_via_trades(
        self, order_id: str | None, symbol: str, attempts: int = 4, delay: float = 1.0,
    ) -> dict | None:
        """
        Точные детали исполнения (средняя цена, объём, комиссия) через
        историю СДЕЛОК биржи — нужно, когда fetch_order() статус ордера так
        и не подтвердил filled (см. _fetch_confirmed_order/
        _confirm_fill_via_balance). Реальный инцидент: на демо-счету Bybit
        ордер был полностью исполнен (в истории сделок биржи сразу видна
        "Заповнена ціна"/цена и комиссия по факту), но fetch_order по его ID
        ни разу не показал average/price — _confirm_fill_via_balance в этом
        случае подтверждал только ОБЪЁМ (по изменению баланса), а цену брал
        из запрошенной/entry_price как заглушку — PnL любой такой сделки
        считался от несовпадающих цен открытия/закрытия и получался
        околонулевым независимо от реального результата на бирже.

        Пробуем сначала fetch_order_trades (сделки именно этого ордера,
        точнее всего), затем fetch_my_trades с фильтром по order_id. РЕАЛЬНЫЙ
        инцидент (прод, реальный счёт Bybit, SUI/USDT, LINK/USDT, LIT/USDT):
        история сделок биржи иногда ещё не готова к моменту, когда истекло
        окно поллинга fetch_order (~6с) — единственная попытка сразу же
        падала в грубый fallback по балансу (цена/комиссия — оценка, а не
        реальные с биржи). Повторяем несколько раз с паузой — история сделок
        обычно догоняет статус ордера буквально на пару секунд позже. Как и в
        других необязательных сверках с биржей в этом классе, любая
        неожиданность (биржа не поддерживает метод, сеть, неожиданный формат
        ответа) просто означает "детали недоступны этим способом", а не
        падение всего исполнения ордера.
        """
        if not order_id:
            return None
        for attempt in range(attempts):
            result = await self._fetch_fill_details_via_trades_once(order_id, symbol)
            if result is not None:
                return result
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
        return None

    async def _fetch_fill_details_via_trades_once(self, order_id: str, symbol: str) -> dict | None:
        try:
            trades = None
            try:
                trades = await self.exchange.fetch_order_trades(order_id, symbol)
            except Exception:
                trades = None
            if not trades:
                recent = await self.exchange.fetch_my_trades(symbol, limit=10)
                trades = [t for t in (recent or []) if t.get("order") == order_id]
            if not trades:
                return None

            total_amount = sum(float(t.get("amount") or 0) for t in trades)
            if total_amount <= 0:
                return None
            total_cost = sum(
                float(t["cost"]) if t.get("cost") is not None else float(t.get("amount") or 0) * float(t.get("price") or 0)
                for t in trades
            )
            average = total_cost / total_amount
            if not average:
                return None

            fee_cost = 0.0
            fee_currency = None
            for t in trades:
                fee = t.get("fee") or {}
                cost = fee.get("cost")
                if cost:
                    fee_cost += float(cost)
                    fee_currency = fee_currency or fee.get("currency")

            # ID сделки(-ок) на бирже (execId/trade id) — короче и это именно
            # то, что видно как "ID ордера"/"ID транзакції" в истории сделок
            # на самой бирже; parent-ордер (order["id"]) — отдельный,
            # значительно более длинный технический ID, который в интерфейсе
            # биржи нигде не показывается.
            trade_ids = [str(t["id"]) for t in trades if t.get("id")]

            return {
                "amount": total_amount, "average": average,
                "fee": {"cost": fee_cost, "currency": fee_currency},
                "trade_ids": trade_ids,
            }
        except Exception as e:
            logger.debug(f"Не удалось получить детали исполнения через историю сделок {order_id} ({symbol}): {e}")
            return None

    def _resolve_fee(
        self, fee_info: dict | None, filled_amount: float, fill_price: float, side: str, symbol: str,
    ) -> tuple[float, str | None]:
        """
        Комиссию исполнения нужно брать РЕАЛЬНУЮ с биржи (order["fee"] или
        сумма комиссий из истории сделок — см. _fetch_fill_details_via_trades)
        — если её не удалось получить НИОТКУДА (ни один из источников не дал
        cost), считаем по стандартной ставке spot-таксы (то же приближение,
        что и paper_fee_pct в paper-режиме) вместо того, чтобы оставлять 0 —
        иначе PnL был бы завышен на величину реальной, но неучтённой
        комиссии биржи. Валюта комиссии в этом расчётном случае — не
        подтверждённый факт, а стандартное для спота допущение: при покупке
        комиссия обычно удерживается из полученного actива (base), при
        продаже — из полученной quote-валюты.
        """
        fee_info = fee_info or {}
        cost = fee_info.get("cost")
        if cost:
            return float(cost), fee_info.get("currency")
        base_currency, quote_currency = symbol.split("/")
        estimated = filled_amount * fill_price * (settings.paper_fee_pct / 100)
        return estimated, (base_currency if side == "buy" else quote_currency)

    async def _place_stop_loss_order(self, symbol: str, amount: float, stop_loss_price: float) -> str | None:
        """
        Разместить биржевой стоп-ордер (Bybit spot: условный 'tpslOrder' —
        рыночная продажа по достижении stop_loss_price), чтобы защита
        позиции не зависела от того, жив ли процесс бота и успевает ли
        внутренний поллинг цены (_check_position_exit в main.py) её
        отследить. Тейк-профиты (TP1/TP2/TP3) сознательно остаются только
        во внутренней логике: у Bybit spot нет родного OCO-механизма
        частичного выхода по нескольким уровням, один статичный биржевой
        TP-ордер такому сценарию не соответствует, а SL — соответствует
        (единственный уровень, который в момент установки актуален всегда).

        ВАЖНО: Bybit spot НЕ поддерживает stopLoss/takeProfit, прикреплённые
        к самому маркет-ордеру (ccxt бросает InvalidOrder) — это ОТДЕЛЬНЫЙ
        условный ордер, размещаемый уже после того, как позиция открыта.

        Best-effort: любая ошибка (биржа отклонила триггер-цену, не
        поддерживается для этой пары и т.п.) не должна блокировать саму
        позицию — просто логируем и остаёмся под защитой одной внутренней
        проверки, как было раньше.
        """
        if amount <= 0 or not stop_loss_price:
            return None
        # Отслеживаемый объём позиции мог немного разойтись с реальным
        # остатком на бирже — та же причина, что и в close_real_position
        # (комиссии, округление лота, накопленный дрейф за несколько
        # частичных закрытий или рестартов процесса): условный SL-ордер на
        # биржевой остаток, а не на устаревший расчётный объём — иначе
        # биржа отклоняет ЕГО ЦЕЛИКОМ с "Insufficient balance", и позиция
        # остаётся вовсе без биржевой защиты (реальный инцидент: XAUT/USDT,
        # LINK/USDT после нескольких частичных TP).
        try:
            base_currency = symbol.split("/")[0]
            balance = await self.exchange.fetch_balance()
            available = self._extract_currency_balance(balance, base_currency)
            if 0 < available < amount:
                logger.debug(
                    f"SL {symbol}: доступно {available:.8f} {base_currency} < отслеживаемого "
                    f"{amount:.8f} — выставляем на доступный остаток."
                )
                amount = available
        except Exception as e:
            logger.debug(f"Не удалось сверить баланс перед выставлением SL {symbol}: {e}")
        try:
            order = await self.exchange.create_market_sell_order(
                symbol, amount, params={"stopLossPrice": stop_loss_price},
            )
            order_id = order.get("id") if order else None
            if order_id:
                logger.info(
                    f"🛡️ Биржевой SL выставлен: {symbol} sell {amount:.8f} @ триггер "
                    f"{stop_loss_price} (ордер {order_id})"
                )
            return order_id
        except Exception as e:
            logger.warning(
                f"⚠️ Не удалось выставить биржевой SL для {symbol} (триггер {stop_loss_price}): {e} "
                f"— позиция защищена только внутренним поллингом цены."
            )
            return None

    async def _cancel_order_safe(self, symbol: str, order_id: str | None) -> None:
        """Best-effort отмена ордера — он мог уже исполниться или быть отменённым, это не ошибка."""
        if not order_id:
            return
        try:
            await self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.debug(f"Не удалось отменить ордер {order_id} ({symbol}) — возможно, уже неактивен: {e}")

    async def sync_stop_loss_order(self, symbol: str, amount: float, stop_loss_price: float | None) -> None:
        """
        Пересоздать биржевой SL-ордер под текущий остаток/цену позиции —
        нужно после частичного закрытия (TP1/TP2 уменьшают объём) и после
        переноса SL в безубыток (см. _check_position_exit в main.py):
        старый биржевой ордер продавал бы либо неверный объём, либо по
        неверной, уже неактуальной цене. Отменяет прежний отслеживаемый
        SL-ордер (если был) и, если задан stop_loss_price и остаток > 0,
        ставит новый.
        """
        pos = self.real_positions.get(symbol)
        if pos is None:
            return
        await self._cancel_order_safe(symbol, pos.get("sl_order_id"))
        pos["sl_order_id"] = None
        if stop_loss_price and amount > 0:
            pos["sl_order_id"] = await self._place_stop_loss_order(symbol, amount, stop_loss_price)

    async def _confirm_fill_via_balance(
        self, symbol: str, side: str, balance_before: float, expected_amount: float,
    ) -> float | None:
        """
        Второй, независимый от статуса ордера способ подтвердить исполнение —
        нужен, когда _fetch_confirmed_order() так и не увидела filled за всё
        окно поллинга. РЕАЛЬНЫЙ инцидент (демо-счёт Bybit, RLUSD/USDT,
        USDC/USDT, BTC/USDT): биржа исполняла ордер мгновенно (сразу видно в
        истории сделок биржи), но fetch_order по его ID стабильно не
        показывал filled даже после 6 секунд поллинга — увеличение окна
        поллинга не спасает, если сама биржа не успевает обновить статус
        конкретного ордера, а не просто "ещё пара тиков и появится".

        Вместо ожидания статуса ОРДЕРА сверяем остаток БАЗОВОЙ валюты на
        счету до и после попытки: он либо вырос (buy), либо уменьшился
        (sell) — это прямое наблюдение факта "деньги потрачены/актив
        продан", не зависящее от того, обновила ли биржа статус именно
        этого ордера. Возвращает оценку реально исполненного объёма или
        None, если баланс не изменился на сколько-нибудь значимую долю
        запрошенного (защита от шума округления/несвязанной активности на
        счету — тот же счёт может использоваться и вручную).
        """
        try:
            balance = await self.exchange.fetch_balance()
        except Exception as e:
            logger.debug(f"Не удалось сверить баланс {symbol} для второй проверки исполнения: {e}")
            return None
        base_currency = symbol.split("/")[0]
        balance_after = self._extract_currency_balance(balance, base_currency)
        diff = (balance_after - balance_before) if side == "buy" else (balance_before - balance_after)
        if diff >= expected_amount * 0.5:
            return diff
        return None

    async def _execute_real_order(self, order_data: dict) -> Order | None:
        """Реальный ордер через биржу."""
        symbol = order_data["symbol"]
        side = order_data["side"]
        amount = order_data["amount"]
        price = order_data["price"]

        # На споте нет встроенного шорта — этот метод вызывается ТОЛЬКО для
        # ОТКРЫТИЯ новой позиции (create_order -> _execute_real_order);
        # закрытие всегда идёт отдельным путём через close_real_position, а
        # не сюда. Раньше это никак не проверялось: реальный инцидент
        # (ENA/USDT, стратегия bb_strategy) — сигнал side="short" дошёл до
        # сюда и создал market SELL, который на бирже реально ИСПОЛНИЛСЯ
        # (аккаунт держал ENA не через этого бота — предыдущая ручная/чужая
        # активность), молча распродав реальный актив. Ордер записался в БД
        # без какой-либо позиции в памяти (real_positions заполняется только
        # в ветке side=="buy" ниже) — после следующего рестарта он
        # реконструировался из БД как "открытая short-позиция" (см.
        # _load_open_positions_from_db), которую close_real_position
        # принципиально не умеет закрыть (там уже стоит отдельная защита
        # "side != long"), и позиция намертво зависала, засоряя логи той
        # же ошибкой на каждой попытке закрытия.
        if side != "buy":
            logger.error(
                f"❌ Реальный ордер {symbol} отклонён: сторона '{side}' (шорт) не поддерживается "
                f"на споте — на споте нет встроенного шорта, а продажа при наличии баланса актива "
                f"реально исполнилась бы, распродав реальные средства без возможности закрыть "
                f"'позицию' обратно."
            )
            return None

        below_min = self._below_exchange_minimum(symbol, amount, price)
        if below_min:
            logger.warning(
                f"⚠️ Реальный ордер {symbol} пропущен: {below_min} — доступного баланса "
                f"недостаточно для минимального размера ордера по этой паре."
            )
            return None

        balance_before = None
        try:
            snapshot = await self.exchange.fetch_balance()
            balance_before = self._extract_currency_balance(snapshot, symbol.split("/")[0])
        except Exception as e:
            logger.debug(f"Не удалось снять баланс {symbol} до отправки ордера: {e}")

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

            order = await self._fetch_confirmed_order(order, symbol)
            # Bybit возвращает orderId даже для ордера, который потом не
            # исполнился (например, отклонён движком сопоставления) —
            # получение orderId без исключения НЕ значит, что сделка реально
            # произошла на бирже. Раньше это не проверялось: бот регистрировал
            # позицию и списывал "комиссию" для ордера, которого по факту
            # никогда не было — ни самого актива, ни истории операций по нему
            # на бирже, а закрыть такую фантомную позицию невозможно
            # (продавать нечего, "Insufficient balance" на каждой попытке).
            trade_ids: list[str] | None = None
            # Историю сделок биржи (execId, средневзвешенная цена, реальная
            # комиссия) пробуем ВСЕГДА, когда её можно получить — это и есть
            # то, что видно как "Filled Price"/комиссия в истории сделок на
            # самой бирже. order["average"]/order["fee"] из fetch_order
            # нередко отстают: Bybit может подтвердить объём/статус раньше,
            # чем подтянуть в тот же снимок реальную комиссию — раньше эти
            # (потенциально неполные) данные использовались напрямую, из-за
            # чего комиссия/цена в дашборде расходились с биржей.
            trade_fill = await self._fetch_fill_details_via_trades(order.get("id"), symbol)
            if trade_fill:
                order = dict(order)
                order["filled"] = trade_fill["amount"]
                order["average"] = trade_fill["average"]
                order["fee"] = trade_fill["fee"]
                trade_ids = trade_fill.get("trade_ids")
            elif not (order.get("filled") or 0) > 0:
                # История сделок недоступна, и fetch_order так и не показал
                # filled — откатываемся к грубому подтверждению по изменению
                # баланса (объём есть, а цена/комиссия — оценка по
                # запрошенной цене/стандартной ставке ниже).
                confirmed_amount = (
                    await self._confirm_fill_via_balance(symbol, side, balance_before, amount)
                    if balance_before is not None else None
                )
                if confirmed_amount is None:
                    logger.error(
                        f"❌ Ордер {order.get('id')} ({symbol}) не подтверждён как реально "
                        f"исполненный на бирже (filled={order.get('filled')!r}) — позиция НЕ "
                        f"регистрируется, данные должны быть идентичны бирже."
                    )
                    return None
                logger.warning(
                    f"⚠️ Ордер {order.get('id')} ({symbol}) подтверждён по изменению баланса на "
                    f"бирже ({confirmed_amount:.8f} {symbol.split('/')[0]}), хотя fetch_order так и "
                    f"не показал filled — цена исполнения оценивается по запрошенной, не биржевой."
                )
                order = dict(order)
                order["filled"] = confirmed_amount
            logger.info(f"✅ Ордер исполнен на бирже: {order['id']} | {side.upper()} {amount:.4f} {symbol}")

            # ccxt возвращает order["fee"] как dict {"cost": ..., "currency": ...}
            # (или None), а не число — писать его напрямую в DECIMAL-колонку
            # было ошибкой. Аналогично order["price"] у маркет-ордеров обычно
            # None (заполняется только order["average"]).
            fill_price = order.get("average") or order.get("price") or price
            filled_amount = order["filled"] or amount
            fill_fee, fee_currency = self._resolve_fee(order.get("fee"), filled_amount, fill_price, side, symbol)
            fee_info = {"cost": fill_fee, "currency": fee_currency}
            # На споте комиссия обычно списывается из полученного актива:
            # при покупке — из base-валюты (1INCH), а не из USDT. Раньше
            # позиция запоминалась с "amount" = запрошенный объём, без
            # вычета комиссии — реально на бирже оставалось на fill_fee
            # меньше, и попытка закрыть позицию тем же объёмом падала с
            # "Insufficient balance".
            net_amount = filled_amount
            base_currency = symbol.split("/")[0]
            if side == "buy" and fee_info.get("currency") == base_currency:
                net_amount = max(0.0, filled_amount - fill_fee)

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
                    filled_amount=filled_amount,
                    filled_price=fill_price,
                    fee=fill_fee,
                    fee_currency=fee_currency,
                    stop_loss=order_data["stop_loss"],
                    take_profit=order_data["take_profit"],
                    order_id_exchange=",".join(trade_ids) if trade_ids else order["id"],
                    client_order_id=order_data["client_order_id"],
                )
                session.add(order_obj)
                await session.flush()
                order_id = order_obj.id

            # Регистрируем открытую реальную позицию так же, как paper —
            # без этого SL/TP по ней никогда не проверялись бы (см.
            # close_real_position / _check_position_exit в main.py), а
            # закрыть её вручную из дашборда было бы нельзя. "amount" —
            # именно net_amount (за вычетом комиссии из base-валюты, если
            # применимо), т.к. это реально доступный к продаже остаток.
            if side == "buy":
                self.real_positions[symbol] = {
                    "amount": net_amount,
                    "entry_price": fill_price,
                    "side": "long",
                    "strategy_id": order_data.get("strategy_id"),
                    "stop_loss": order_data.get("stop_loss"),
                    "take_profit": order_data.get("take_profit"),
                    "order_id": order_id,
                    "entry_fee": fill_fee,
                    "opened_at": utcnow(),
                    "sl_order_id": None,
                }
                if order_data.get("stop_loss"):
                    await self.sync_stop_loss_order(symbol, net_amount, order_data["stop_loss"])

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

        # Отменяем биржевой SL-ордер (если был) ДО собственной продажи —
        # иначе он остаётся висеть параллельно с этим закрытием (не важно,
        # по какой причине оно происходит — TP, ручное закрытие или сам же
        # SL) и может конфликтовать за один и тот же остаток базовой валюты.
        tracked_pos = self.real_positions.get(symbol)
        if tracked_pos is not None:
            await self._cancel_order_safe(symbol, tracked_pos.get("sl_order_id"))

        # Отслеживаемый объём позиции — оценка (комиссии, округление лота
        # биржей и т.п. могут понемногу расходиться с реальным остатком) —
        # без подстраховки продажа "полного" объёма падает на бирже с
        # "Insufficient balance", и позиция навсегда зависает открытой,
        # хотя реально продать почти всё, что есть, всё равно можно.
        sell_amount = amount
        available = None
        try:
            base_currency = symbol.split("/")[0]
            balance = await self.exchange.fetch_balance()
            available = self._extract_currency_balance(balance, base_currency)
            if 0 < available < amount:
                logger.warning(
                    f"⚠️ Доступный баланс {base_currency} ({available:.8f}) меньше отслеживаемого "
                    f"объёма позиции {symbol} ({amount:.8f}) — продаём доступный остаток."
                )
                sell_amount = available
        except Exception as e:
            logger.debug(f"Не удалось сверить доступный баланс перед закрытием {symbol}: {e}")

        try:
            order = await self.exchange.create_market_sell_order(symbol, sell_amount)
        except Exception as e:
            # available логируется прямо здесь (а не только по debug выше) —
            # без этого "Insufficient balance" от биржи ни разу не говорил,
            # ЧТО именно бот считает доступным по СВОЕЙ проверке: 0 (актив
            # действительно отсутствует — например уже продан вручную, или
            # запрос баланса ушёл не в тот account type биржи) — это
            # принципиально другая причина, чем "чуть меньше из-за
            # комиссии/округления" (эту вторую подстраховка выше уже решает).
            logger.error(
                f"❌ Не удалось закрыть реальную позицию {symbol}: {e} | "
                f"наш учёт: {amount:.8f}, доступно на бирже: "
                f"{available if available is not None else 'не удалось проверить'}"
            )
            # Помимо available == 0 (актива нет вообще), есть ещё два случая
            # той же природы — продать позицию в принципе невозможно, и
            # бесконечный повтор попытки (как раньше было с available == 0)
            # ничего не изменит:
            # 1) доступный остаток положительный, но настолько мал, что
            #    биржа отклоняет ЛЮБОЙ ордер на его продажу как ниже
            #    минимального торгуемого ОБЪЁМА (ccxt/биржа сообщают об этом
            #    словами "precision"/"minimum" в тексте ошибки — например
            #    "amount ... must be greater than minimum amount precision
            #    of 0.001").
            # 2) объёма достаточно (available >= amount), но его СТОИМОСТЬ в
            #    quote-валюте (amount * текущая цена) ниже минимальной для
            #    пары — Bybit отвечает retCode 170140 "Order value exceeded
            #    lower limit" (реальный инцидент: SUI/USDT, ~26 минут подряд
            #    одна и та же ошибка каждые ~70с, available > amount, так что
            #    условие (1) не срабатывало вообще). В отличие от (1), эта
            #    проверка НЕ требует available < amount — деление позиции на
            #    более мелкие ордера её не решает, наоборот, ещё уменьшает
            #    стоимость каждого.
            unsellable_dust = available == 0 or (
                available is not None
                and available < amount
                and any(kw in str(e).lower() for kw in ("precision", "minimum"))
            ) or "lower limit" in str(e).lower()
            if unsellable_dust:
                await self._reconcile_phantom_position(symbol, order_open_id)
            return None

        order = await self._fetch_confirmed_order(order, symbol)
        trade_ids: list[str] | None = None
        # Историю сделок биржи пробуем ВСЕГДА (см. тот же приоритет и
        # обоснование при открытии в _execute_real_order) — это то же самое,
        # что видно как Filled Price/комиссия в истории сделок на самой
        # бирже, и точнее, чем order["average"]/order["fee"] из fetch_order.
        trade_fill = await self._fetch_fill_details_via_trades(order.get("id"), symbol)
        if trade_fill:
            order = dict(order)
            order["filled"] = trade_fill["amount"]
            order["average"] = trade_fill["average"]
            order["fee"] = trade_fill["fee"]
            trade_ids = trade_fill.get("trade_ids")
        elif not (order.get("filled") or 0) > 0:
            # История сделок недоступна, и fetch_order так и не показал
            # filled — второй, независимый от статуса ордера способ
            # подтверждения (см. _confirm_fill_via_balance): available уже
            # снят с биржи чуть выше (до отправки sell), так что здесь не
            # нужен ещё один запрос баланса "до".
            confirmed_amount = (
                await self._confirm_fill_via_balance(symbol, "sell", available, sell_amount)
                if available is not None else None
            )
            if confirmed_amount is None:
                logger.error(
                    f"❌ Закрывающий ордер {order.get('id')} ({symbol}) не подтверждён как "
                    f"реально исполненный на бирже (filled={order.get('filled')!r}) — закрытие "
                    f"НЕ засчитывается, данные должны быть идентичны бирже."
                )
                return None
            logger.warning(
                f"⚠️ Закрывающий ордер {order.get('id')} ({symbol}) подтверждён по изменению "
                f"баланса на бирже ({confirmed_amount:.8f} {symbol.split('/')[0]}), хотя "
                f"fetch_order так и не показал filled — цена исполнения оценивается по цене "
                f"открытия, не биржевой."
            )
            order = dict(order)
            order["filled"] = confirmed_amount
        exit_price = order.get("average") or order.get("price") or entry_price
        exit_filled_amount = order["filled"] or amount
        exit_fee, exit_fee_currency = self._resolve_fee(order.get("fee"), exit_filled_amount, exit_price, "sell", symbol)

        # Комиссия ОТКРЫТИЯ на споте обычно удерживается в BASE-валюте
        # (полученный актив) — например 105.4915 TAC, а не в USDT. PnL ниже
        # считается в QUOTE-валюте (USDT): вычитать base-валютную комиссию
        # как есть означало бы принять "105.4915 TAC" за "105.4915 USDT" —
        # искажение на порядки. Переводим её в USDT-эквивалент по цене
        # входа, если знаем (из fee_currency открывающего Order), что она
        # была в base-валюте; иначе (уже в quote или неизвестно) — как есть.
        entry_fee_quote = entry_fee
        base_currency = symbol.split("/")[0]
        if order_open_id is not None:
            try:
                async with get_session() as session:
                    opening_order = (
                        await session.execute(select(Order).where(Order.id == order_open_id))
                    ).scalar_one_or_none()
                if opening_order is not None and opening_order.fee_currency == base_currency:
                    entry_fee_quote = entry_fee * entry_price
            except Exception as e:
                logger.debug(f"Не удалось определить валюту комиссии открытия для {symbol}: {e}")

        pnl = (exit_price - entry_price) * amount - entry_fee_quote - exit_fee
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
                fee_currency=exit_fee_currency,
                order_id_exchange=",".join(trade_ids) if trade_ids else order["id"],
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

    async def _reconcile_phantom_position(self, symbol: str, order_open_id: int | None):
        """
        Снять с учёта позицию, которую больше невозможно закрыть обычной
        продажей на бирже — независимо согласны два сигнала: наша
        предварительная проверка баланса (available < amount) и сама
        попытка продажи. Два варианта:
        - available == 0: актива нет вообще. Обычно открывающий BUY-ордер
          вернул orderId, но так и не исполнился на бирже (Bybit не
          гарантирует исполнение самим фактом ответа на создание ордера —
          см. проверку в _execute_real_order/close_real_position), а старый
          код регистрировал позицию без проверки реального исполнения.
        - 0 < available < amount и биржа отклоняет продажу available как
          ниже минимального торгуемого объёма ("precision"/"minimum" в
          тексте ошибки): учтённый объём позиции завышен относительно
          реального остатка (например, старый код до фикса фантомных
          позиций регистрировал ЗАПРОШЕННЫЙ объём вместо фактически
          исполненного при частичном филле) — остаток биржевой "пылью"
          продать нельзя, а повторные попытки продать несуществующий
          излишек только зацикливаются и портят расчёт equity/просадки.

        Помечаем исходный открывающий Order как rejected (а не оставляем
        "filled") — без этого при следующем рестарте бота позиция
        реконструировалась бы из БД заново (реконструкция берёт только
        Order.status == "filled", см. _load_open_positions_from_db), и
        зависание повторилось бы. Не создаём фиктивную закрывающую Trade —
        учитывать нечего: либо реального открытия не было, либо то, что
        было, уже неотделимо от текущей биржевой пыли.
        """
        self.real_positions.pop(symbol, None)
        if order_open_id is not None:
            async with get_session() as session:
                await session.execute(
                    update(Order).where(Order.id == order_open_id).values(status="rejected")
                )
                await session.commit()
        # Без этого risk_manager.state.open_positions_count навсегда
        # оставался бы завышенным после каждой реконсиляции (эта функция —
        # единственное место, где позиция снимается с учёта в обход обычных
        # путей закрытия — /positions/close и SL/TP в _check_position_exit,
        # которые сами вызывают on_position_closed) — искусственно
        # ограничивая число новых сделок через check_max_positions(), пока
        # число открытых позиций фактически меньше лимита.
        risk_manager.on_position_closed(symbol)
        logger.warning(
            f"⚠️ Позиция {symbol} снята с учёта: закрыть её обычной продажей на бирже "
            f"невозможно (нет актива или его остаток ниже минимального торгуемого объёма) "
            f"— исходный открывающий ордер помечен как rejected, чтобы не восстанавливать "
            f"позицию заново при следующем рестарте."
        )

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

    async def reconcile_real_positions(self) -> float | None:
        """
        Сверить ВСЕ отслеживаемые реальные позиции с фактическими остатками
        на бирже одним вызовом fetch_balance() и снять с учёта те, которые
        обычной продажей закрыть уже невозможно (актива нет вообще или его
        остаток ниже минимального торгуемого объёма пары) — ДО того, как
        расхождение попадёт в _compute_equity()/просадку в main.py.

        До этого сверка была чисто РЕАКТИВНОЙ: она случалась только в
        момент попытки закрыть позицию (close_real_position), то есть
        только когда сработает SL/TP. Если цена никогда не доходила до
        SL/TP, испорченная позиция (раздутый/не совпадающий с биржей
        amount) могла висеть в real_positions неограниченно долго, каждую
        итерацию искажая equity — так разово раздутый объём AVAX/USDT
        (учтено 416.5, на бирже 0.00046) превратил "Просадку" в дашборде в
        "-220110,7%".

        Возвращает актуальный свободный баланс USDT из ТОГО ЖЕ запроса —
        вызывающий код (main.py, расчёт equity) должен использовать именно
        его вместо отдельного get_real_balance(), чтобы не делать два
        одинаковых запроса к бирже за одну итерацию.
        """
        if not self.exchange:
            return None
        try:
            balance = await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None

        for symbol, pos in list(self.real_positions.items()):
            tracked_amount = pos.get("amount") or 0
            if tracked_amount <= 0:
                continue
            base_currency = symbol.split("/")[0]
            available = self._extract_currency_balance(balance, base_currency)
            if available >= tracked_amount:
                continue
            min_amount = self._market_min_amount(symbol)
            unsellable = available == 0 or (min_amount is not None and available < min_amount)
            if not unsellable:
                continue
            # Прежде чем списывать позицию как фантомную (без PnL), проверяем
            # ДВА способа объяснить исчезновение реальным закрытием на бирже
            # (а не багом/пылью) — по возрастающей специфичности:
            # 1) свой же биржевой SL-ордер сработал сам по себе, без участия
            #    бота (см. sync_stop_loss_order) — знаем точный order id;
            # 2) более общий случай — позиции без выставленного SL-ордера
            #    (SL не настроен, отклонён биржей, или закрыто вручную/через
            #    TP на самой бирже) — ищем недавнюю продажу в истории сделок
            #    биржи по объёму (см. _finalize_via_recent_trade_history).
            # Оба варианта пишут закрытие с настоящими ценой/комиссией/PnL —
            # без них любое закрытие в обход обычного цикла бота (в т.ч.
            # ручное на бирже) молча терялось бы из истории сделок навсегда,
            # а не появлялось бы даже после рестарта.
            if await self._finalize_externally_closed_position(symbol, pos):
                continue
            if await self._finalize_via_recent_trade_history(symbol, pos):
                continue
            logger.warning(
                f"⚠️ Периодическая сверка позиций: {symbol} — учтено {tracked_amount:.8f}, "
                f"на бирже доступно {available:.8f} (продать невозможно) — расхождение поймано "
                f"до попытки закрытия, чтобы не портить equity/просадку."
            )
            await self._reconcile_phantom_position(symbol, pos.get("order_id"))

        return self._extract_usdt_balance(balance)

    async def _finalize_externally_closed_position(self, symbol: str, pos: dict) -> bool:
        """
        Позиция пропала с баланса биржи (available ~0 при периодической
        сверке reconcile_real_positions), и на неё был выставлен биржевой
        SL-ордер (см. sync_stop_loss_order) — скорее всего, именно он
        сработал сам по себе, без участия бота (например, между итерациями
        основного цикла или пока процесс был недоступен). В отличие от
        _reconcile_phantom_position (для позиций, реальное исполнение
        которых объяснить нечем — фантом/пыль), здесь есть конкретный
        биржевой ордер, который можно проверить: если он реально исполнился,
        закрытие записывается с настоящими ценой/комиссией/PnL — как обычное
        закрытие, а не молча теряется из статистики и risk_manager.daily_pnl.

        Возвращает True, если закрытие удалось финализировать (тогда
        _reconcile_phantom_position для этого символа вызывать уже не надо).
        """
        sl_order_id = pos.get("sl_order_id")
        if not sl_order_id:
            return False
        try:
            order = await self.exchange.fetch_order(sl_order_id, symbol)
        except Exception as e:
            logger.debug(f"Не удалось проверить биржевой SL-ордер {sl_order_id} ({symbol}): {e}")
            return False
        status = str(order.get("status") or "").lower()
        if status not in ("closed", "filled"):
            return False

        amount = float(order.get("filled") or pos.get("amount") or 0)
        if amount <= 0:
            return False

        trade_fill = await self._fetch_fill_details_via_trades(str(sl_order_id), symbol)
        if trade_fill:
            exit_price = trade_fill["average"]
            amount = trade_fill["amount"]
            exit_fee = trade_fill["fee"].get("cost") or 0
        else:
            exit_price = order.get("average") or order.get("price")
            if not exit_price:
                return False
            exit_fee, _ = self._resolve_fee(order.get("fee"), amount, exit_price, "sell", symbol)

        await self._record_external_close(
            symbol, pos, exit_price=exit_price, amount=amount, exit_fee=exit_fee,
            order_id_exchange=str(sl_order_id),
            log_note=f"🛡️ Биржевой SL сработал сам по себе (вне цикла бота): {symbol}",
        )
        return True

    async def _finalize_via_recent_trade_history(self, symbol: str, pos: dict) -> bool:
        """
        Общий случай (в отличие от _finalize_externally_closed_position выше,
        здесь НЕТ известного order id для точной проверки): позиция пропала
        с баланса биржи, но у неё либо не было выставленного биржевого
        SL-ордера (SL не задан, отклонён биржей — см. _place_stop_loss_order),
        либо закрытие произошло другим способом — вручную на самой бирже
        или через биржевой TP. Ищем недавнюю ПРОДАЖУ в истории сделок биржи
        по этому символу и сверяем её объём с тем, что мы отслеживаем.

        ВАЖНО: аккаунт биржи может использоваться не только этим ботом
        (например, параллельно работающим независимым ботом на том же
        символе) — доверять чужой сделке с совпадающим объёмом по чистой
        случайности опасно (реальные деньги, неверно приписанный PnL хуже,
        чем отсутствие записи). Поэтому принимаем совпадение, только если:
        (1) сделка произошла ПОСЛЕ открытия нашей позиции, и
        (2) суммарный объём недавних продаж отличается от нашего
        отслеживаемого объёма не более чем на 15% — иначе (в т.ч. если
        историю сделок вообще не удалось получить) не гадаем и оставляем
        обычный фолбэк на _reconcile_phantom_position (без PnL, но без
        риска приписать чужую сделку).
        """
        tracked_amount = pos.get("amount") or 0
        opened_at = pos.get("opened_at")
        if tracked_amount <= 0 or not opened_at:
            return False
        # Как и другие необязательные сверки с биржей в этом классе (см.
        # _fetch_fill_details_via_trades) — вся функция под одним широким
        # try/except: непредвиденный формат ответа биржи/мока (например,
        # fetch_my_trades не поддерживается или неитерируемый результат)
        # должен просто означать "сверить не удалось", а не ронять весь
        # reconcile_real_positions.
        try:
            recent = await self.exchange.fetch_my_trades(symbol, limit=20)
            if not recent:
                return False

            # opened_at — наивный datetime в UTC (см. utcnow); .timestamp()
            # на наивном datetime трактует его как ЛОКАЛЬНОЕ время, а не
            # UTC, и даёт неверный epoch вне контейнеров с TZ=UTC — сначала
            # явно проставляем tzinfo=UTC, как и utcnow_timestamp.
            opened_at_ts = opened_at.replace(tzinfo=UTC).timestamp() * 1000
            sells = [
                t for t in recent
                if str(t.get("side", "")).lower() == "sell" and (t.get("timestamp") or 0) >= opened_at_ts
            ]
            if not sells:
                return False

            total_amount = sum(float(t.get("amount") or 0) for t in sells)
            if total_amount <= 0:
                return False
            if abs(total_amount - tracked_amount) / tracked_amount > 0.15:
                logger.debug(
                    f"Сверка {symbol}: недавние продажи ({total_amount:.8f}) слишком расходятся с "
                    f"отслеживаемым объёмом ({tracked_amount:.8f}) — не гадаем, чей это ордер."
                )
                return False
        except Exception as e:
            logger.debug(f"Не удалось сверить закрытие {symbol} по истории сделок: {e}")
            return False

        total_cost = sum(
            float(t["cost"]) if t.get("cost") is not None else float(t.get("amount") or 0) * float(t.get("price") or 0)
            for t in sells
        )
        exit_price = total_cost / total_amount if total_amount else 0
        if not exit_price:
            return False
        exit_fee = 0.0
        for t in sells:
            fee = t.get("fee") or {}
            cost = fee.get("cost")
            if cost:
                exit_fee += float(cost)
        trade_ids = [str(t["id"]) for t in sells if t.get("id")]

        await self._record_external_close(
            symbol, pos, exit_price=exit_price, amount=total_amount, exit_fee=exit_fee,
            order_id_exchange=",".join(trade_ids) if trade_ids else None,
            log_note=f"🔍 Позиция закрыта вне цикла бота (найдено по истории сделок биржи): {symbol}",
        )
        return True

    async def _record_external_close(
        self, symbol: str, pos: dict, *, exit_price: float, amount: float, exit_fee: float,
        order_id_exchange: str | None, log_note: str,
    ) -> None:
        """
        Общий хвост записи закрытия, обнаруженного вне обычного цикла бота
        (см. _finalize_externally_closed_position и
        _finalize_via_recent_trade_history) — та же запись Order+Trade и
        публикация TradeEvent, что и у close_real_position, но без попытки
        продать (это уже произошло на бирже без нас).
        """
        entry_price = pos.get("entry_price") or 0
        entry_fee = pos.get("entry_fee") or 0
        pnl = (exit_price - entry_price) * amount - entry_fee - exit_fee
        pnl_pct = (pnl / (entry_price * amount) * 100) if entry_price and amount else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "break-even")
        opened_at = pos.get("opened_at")
        holding_seconds = int((utcnow() - opened_at).total_seconds()) if opened_at else 0
        order_open_id = pos.get("order_id")

        self.real_positions.pop(symbol, None)

        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            strategy_db_id = await self._resolve_strategy_id(session, pos.get("strategy_id"))
            close_order = Order(
                exchange_id=exchange_id,
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                side="sell",
                order_type="market",
                amount=amount,
                price=exit_price,
                status="filled",
                filled_amount=amount,
                filled_price=exit_price,
                fee=exit_fee,
                order_id_exchange=order_id_exchange,
                client_order_id=str(uuid.uuid4())[:12],
                notes="Real close (exchange-triggered, outside bot cycle)",
            )
            session.add(close_order)
            await session.flush()

            trade = Trade(
                symbol_id=symbol_id,
                strategy_id=strategy_db_id,
                order_open_id=order_open_id,
                order_close_id=close_order.id,
                direction="long",
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

        risk_manager.on_trade_closed(pnl)
        risk_manager.on_position_closed(symbol)
        logger.warning(f"{log_note} | PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)")

        trade_event = TradeEvent(
            type="trade_event",
            trade_id=trade_id,
            symbol=symbol,
            direction="long",
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

    @staticmethod
    def _extract_usdt_balance(balance: dict) -> float:
        return ExecutionEngine._extract_currency_balance(balance, "USDT")

    @staticmethod
    def _extract_currency_balance(balance: dict, currency: str) -> float:
        """
        ccxt fetch_balance() кладёт баланс валюты в ВЛОЖЕННЫЙ словарь:
        balance['free'][currency] (плоское число) и/или balance[currency] =
        {'free':, 'used':, 'total':} (сам по себе словарь, а НЕ число).
        Старый код фильтровал `isinstance(v, (int, float))` по
        balance.items() верхнего уровня — balance[currency] там всегда
        словарь, значит проверка никогда не проходила, и баланс всегда
        читался как 0 независимо от биржи и реального остатка на счёте.
        """
        free = balance.get("free") or {}
        value = free.get(currency)
        if value is None:
            entry = balance.get(currency)
            if isinstance(entry, dict):
                value = entry.get("free")
        return float(value or 0.0)

    async def _fetch_confirmed_order(
        self, order: dict, symbol: str, attempts: int = 8, delay: float = 0.75,
    ) -> dict:
        """
        Bybit v5 (и потенциально другие биржи) на СОЗДАНИЕ маркет-ордера
        возвращает только orderId — цена/объём/комиссия исполнения туда не
        попадают, т.к. сопоставление на бирже асинхронное и происходит уже
        ПОСЛЕ ответа на запрос создания. order["average"]/["price"] в таком
        ответе всегда None — exit_price/fill_price падали на entry_price
        (комиссия — на 0), из-за чего PnL ЛЮБОЙ закрытой сделки на Bybit
        получался ровно 0 (или ровно -комиссия, если она всё же попадала).
        Раз рыночный ордер исполняется почти мгновенно, короткий поллинг
        fetch_order догоняет реальные данные буквально за один-два тика.

        Реальный инцидент (RLUSD/USDT на демо-счёте Bybit): ~50 подряд
        реальных покупок за час, каждая меньше предыдущей — ордер реально
        исполнялся на бирже (подтверждено историей сделок биржи), но эта
        функция каждый раз считала его неподтверждённым, вызывающий код
        (_execute_real_order) трактовал это как провал и НЕ регистрировал
        позицию — на следующей итерации стратегия видела символ "свободным"
        и открывала его заново на то, что осталось от баланса. Причина:
        условие выхода из цикла проверяло только average/price, хотя
        решение о провале в вызывающем коде принимается по полю filled —
        ответ fetch_order мог уже содержать реальный filled без ещё
        подтянувшихся average/price, и такой промежуточный снимок никогда
        не возвращался: цикл впустую доходил до конца попыток и откатывался
        к исходному, заведомо неактуальному ответу СОЗДАНИЯ ордера
        (filled там всегда None/0 — см. выше). Теперь: (1) выходим раньше и
        по filled тоже, не только по average/price; (2) если так и не
        дождались ни одного из трёх — возвращаем ПОСЛЕДНИЙ полученный от
        биржи снимок, а не исходный ответ создания ордера, который точно
        устарел.
        """
        if order.get("average") or order.get("price") or (order.get("filled") or 0) > 0:
            return order
        order_id = order.get("id")
        if not order_id:
            return order
        latest = order
        for _ in range(attempts):
            await asyncio.sleep(delay)
            try:
                fetched = await self.exchange.fetch_order(order_id, symbol)
            except Exception as e:
                logger.debug(f"Не удалось уточнить исполнение ордера {order_id} ({symbol}): {e}")
                break
            latest = fetched
            if fetched.get("average") or fetched.get("price") or (fetched.get("filled") or 0) > 0:
                return fetched
        logger.warning(
            f"⚠️ Не удалось дождаться подтверждения исполнения ордера {order_id} ({symbol}) "
            f"за {attempts * delay:.1f}с — используем последний полученный от биржи снимок."
        )
        return latest


# Глобальный экземпляр
execution_engine = ExecutionEngine()
