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

# reconcile_real_positions(): минимальный возраст позиции, прежде чем её
# можно списать как "фантомную" из-за расхождения с балансом биржи — см.
# комментарий на месте использования (реальный инцидент: CHZ/MANA/ZRX,
# биржа не успела отразить только что купленный актив в fetch_balance()).
RECONCILE_MIN_AGE_SECONDS = 180


class ExecutionEngine:
    """Движок исполнения ордеров."""

    def __init__(self):
        # Раньше — единственный self.exchange, привязанный к ТЕКУЩЕМУ
        # settings.market_type. Реальные позиции могут существовать
        # одновременно и на споте, и на фьючерсах (пользователь переключил
        # тумблер, пока часть позиций ещё открыта на другом рынке) — тогда
        # нужны ОБА подключения сразу, иначе ведение "чужой" по текущему
        # тумблеру позиции (SL, закрытие) уходит не в тот ccxt-клиент.
        # Реальный инцидент: при переключении на фьючерсы 3 спотовые позиции
        # (MON/RLUSD/USDC) стали обслуживаться так, будто они фьючерсные.
        # self.exchange ниже остаётся как свойство для обратной
        # совместимости — это клиент ИМЕННО текущего рынка (используется
        # везде, где операция не привязана к конкретной позиции: получение
        # тикера/баланса, открытие НОВОЙ позиции). Для операций над
        # конкретной УЖЕ ОТКРЫТОЙ позицией используется _exchange_for(pos),
        # который резолвит клиент по market_type САМОЙ позиции, а не по
        # текущему тумблеру.
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self.exchange_id: str | None = None
        self.is_paper: bool = settings.is_paper
        self.paper_balance: float = settings.startup_capital_usdt
        self.paper_positions: dict[str, dict] = {}
        self.real_positions: dict[str, dict] = {}
        self.last_prices: dict[str, float] = {}
        self.order_counter = 0

    @property
    def exchange(self) -> ccxt.Exchange | None:
        return self._exchanges.get(settings.market_type)

    @exchange.setter
    def exchange(self, value: ccxt.Exchange | None) -> None:
        self._exchanges[settings.market_type] = value

    def _exchange_for(self, pos: dict | None) -> ccxt.Exchange | None:
        """Ccxt-клиент рынка, на котором была открыта КОНКРЕТНАЯ позиция —
        не текущего тумблера settings.market_type (см. комментарий в __init__)."""
        if pos is None:
            return self.exchange
        return self._exchanges.get(pos.get("market_type", "spot"))

    @staticmethod
    def _ccxt_symbol(exchange: ccxt.Exchange | None, symbol: str) -> str:
        """
        Unified-символ ДЛЯ ПРЯМЫХ ВЫЗОВОВ К CCXT (create_order/
        fetch_position/cancel_order/fetch_ticker/...) — НЕ путать с нашим
        собственным каноническим "BASE/QUOTE", которым everywhere else в
        этом классе (real_positions, БД, main.py, дашборд) обозначается
        символ — тот менять не нужно, только то, что летит В exchange.*().

        Реальный инцидент (прод, месяцами): у ccxt спотовый и linear-swap
        (то, что мы называем "futures"/USDT-perpetual) рынки одной и той
        же пары — это ДВА РАЗНЫХ unified-символа: "BCH/USDT" (спот) и
        "BCH/USDT:USDT" (linear swap, суффикс — расчётная валюта через
        двоеточие; см. parse_market в ccxt/bybit.py — `symbol = symbol +
        ':' + settle`). exchange.market(symbol) matches ПО БУКВАЛЬНОМУ
        СОВПАДЕНИЮ СТРОКИ В self.markets — если "BCH/USDT" уже есть как
        ключ (а он есть — это спотовый рынок), метод возвращает СПОТОВЫЙ
        рынок ВСЕГДА, что бы ни было выставлено в options.defaultType/
        defaultSubType. Раз весь код в этом файле годами обращался к
        "futures"-клиенту голым "BASE/QUOTE" (без суффикса), КАЖДЫЙ вызов
        (create_order, fetch_position, cancel_order, set_leverage,
        fetch_open_orders...) на самом деле резолвился в СПОТОВЫЙ рынок —
        отсюда и fetch_position(), падающий retCode 181001 "category only
        support linear or option" (у спота нет позиций), и стабильный
        170131 "Insufficient balance" на закрытии/SL (это не reduceOnly
        закрытие фьючерсного контракта, а спотовая/маржинальная продажа с
        другой семантикой баланса), и висящие на бирже "условные ордера" —
        всё это были спотовые/маржинальные операции, а не настоящие
        linear-perpetual фьючерсы.

        options.defaultType уже корректно проставлен ПРИ ПОДКЛЮЧЕНИИ
        клиента (см. _connect_exchange: "swap" для futures, "spot" для
        spot) — читаем его отсюда же, а не заводим отдельный market_type
        параметр в каждой сигнатуре: то же самое отличие клиента, просто
        доступное напрямую через сам ccxt-объект.
        """
        if exchange is None or ":" in symbol or "/" not in symbol:
            return symbol
        options = exchange.options
        # exchange.options — обычный dict у реального ccxt.Exchange; защитная
        # проверка типа — не только на случай неожиданной биржи без options,
        # но и на тестовые AsyncMock()-заглушки без спека, у которых
        # exchange.options САМ становится AsyncMock (см. unittest.mock:
        # атрибуты AsyncMock по умолчанию рекурсивно тоже AsyncMock) — без
        # неё .get(...) вернул бы корутину вместо значения.
        if not isinstance(options, dict) or options.get("defaultType") != "swap":
            return symbol
        quote = symbol.split("/")[-1]
        return f"{symbol}:{quote}"

    def get_open_positions(self) -> dict:
        """Открытые позиции для текущего режима (paper или real)."""
        return dict(self.paper_positions if self.is_paper else self.real_positions)

    async def get_reference_price(self, symbol: str, market_type: str | None = None) -> float | None:
        """
        Текущая рыночная цена symbol — для сигналов без явной цены входа
        ("Диапазон входа: по рынку" — см. is_market_entry в
        channel_monitor.py и _on_telegram_signal в main.py), которым всё
        равно нужно конкретное число ДО открытия ордера: расчёт объёма
        (position_value / entry) и оценка качества сигнала (score_signal)
        сами по себе требуют цену раньше, чем create_order() успевает
        получить её самостоятельно для маркет-ордера без явной цены (см.
        тот же приём в create_order — здесь вынесен наружу, чтобы
        вызывающий код мог получить цену РАНЬШЕ, чем нужно посчитать amount).
        """
        try:
            if self.is_paper:
                ticker = await self.exchange.fetch_ticker(self._ccxt_symbol(self.exchange, symbol))
            else:
                order_market_type = market_type or settings.market_type
                exchange = await self._ensure_exchange_connected(order_market_type)
                if exchange is None:
                    return None
                ticker = await exchange.fetch_ticker(self._ccxt_symbol(exchange, symbol))
            return ticker["last"] or ticker["bid"] or ticker["ask"]
        except Exception as e:
            logger.warning(f"Не удалось получить рыночную цену {symbol}: {e}")
            return None

    async def initialize(self, exchange_id: str = "binance"):
        """Инициализация подключения к бирже."""
        self.exchange_id = exchange_id

        if self.is_paper:
            logger.info("📄 Execution Engine: Paper Trading режим")
            await self._restore_paper_state_from_db()
            return

        # initialize() может вызываться повторно в УЖЕ РАБОТАЮЩЕМ процессе —
        # без рестарта контейнера, при живом переключении настроек биржи
        # (market_type/active_exchange/use_exchange_sandbox, см.
        # settings_store.apply_settings_update) — старые соединения (ccxt +
        # их aiohttp ClientSession) при этом просто перезаписывались новыми
        # без закрытия. Для рестарта ВСЕГО процесса эта утечка уже была
        # починена (см. _cleanup() в main.py — вызывается перед выходом), но
        # переключение настройки на лету, пока процесс жив, идёт другим
        # путём — реальный инцидент: переключение market_type на проде дало
        # ту же "Unclosed client session"/"Unclosed connector" от aiohttp,
        # что и рестарт до фикса #31. Закрываем ВСЕ подключённые клиенты
        # (не только текущего рынка) — после предыдущего initialize() мог
        # остаться и лениво поднятый клиент второго рынка (см. ниже).
        for market_type, exchange in list(self._exchanges.items()):
            try:
                await exchange.close()
            except Exception as e:
                logger.debug(f"Не удалось закрыть предыдущее соединение с биржей [{market_type}]: {e}")
        self._exchanges = {}

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

            self.exchange = await self._connect_exchange(exchange_id, settings.market_type)
            logger.info(
                f"🔗 Execution Engine: подключено к {exchange_id}"
                f"{' (демо-счёт)' if settings.use_exchange_sandbox else ' (LIVE, реальные средства)'}"
                f"{' [фьючерсы/swap]' if settings.market_type == 'futures' else ' [спот]'}"
            )

            await self._warn_if_okx_trade_permission_missing(exchange_id)

            # Восстанавливаем позиции ДО расчёта базы для просадки — иначе
            # (см. ниже) на счету с уже открытыми real-позициями базой для
            # drawdown становился один только свободный кэш.
            await self._restore_real_positions_from_db()

            # Позиция могла быть открыта на ДРУГОМ рынке, чем текущий
            # тумблер (например, тумблер сейчас на spot, но восстановлена
            # фьючерсная short-позиция, открытая раньше) — без отдельного
            # подключения к тому рынку её SL/закрытие уходили бы в клиент
            # текущего тумблера, привязанный к неверному типу рынка.
            other_markets = {
                pos.get("market_type", "spot") for pos in self.real_positions.values()
            } - {settings.market_type}
            for other_market_type in other_markets:
                try:
                    self._exchanges[other_market_type] = await self._connect_exchange(
                        exchange_id, other_market_type,
                    )
                    logger.info(
                        f"🔗 Execution Engine: дополнительно подключено к {exchange_id} "
                        f"[{other_market_type}] — есть открытые real-позиции на этом рынке."
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Не удалось подключить {exchange_id} [{other_market_type}] для "
                        f"восстановленных позиций этого рынка: {e}"
                    )

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

    async def _connect_exchange(self, exchange_id: str, market_type: str) -> ccxt.Exchange:
        """
        Собрать и подключить один ccxt-клиент для указанного рынка (spot/
        futures) — общая логика между эagerly подключаемым клиентом текущего
        тумблера и лениво подключаемым клиентом для позиций, восстановленных
        на ДРУГОМ рынке (см. initialize()). Ключи и sandbox-настройки
        читаются из settings — они общие для аккаунта независимо от рынка.
        """
        credentials: dict[str, tuple[str | None, str | None, str | None]] = {
            "binance": (settings.binance_api_key, settings.binance_api_secret, None),
            "bybit": (settings.bybit_api_key, settings.bybit_api_secret, None),
            "okx": (settings.okx_api_key, settings.okx_api_secret, settings.okx_passphrase),
        }
        api_key, api_secret, passphrase = credentials.get(exchange_id, (None, None, None))
        exchange_class = getattr(ccxt, exchange_id)

        # market_type=="futures" переключает ccxt на linear-swap рынок
        # (USDT-perpetual) вместо спота — см. комментарий у
        # settings.market_type. defaultSubType нужен, чтобы попасть именно
        # на linear (USDT-margined), а не inverse-контракты.
        market_options = (
            {"defaultType": "swap", "defaultSubType": "linear"}
            if market_type == "futures"
            else {"defaultType": "spot"}
        )
        exchange_config: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": market_options,
        }
        if passphrase:
            exchange_config["password"] = passphrase
        exchange = exchange_class(exchange_config)

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
                exchange.enable_demo_trading(True)
            else:
                # Тот же API-ключ, но запросы идут на demo/testnet-счёт
                # биржи вместо реального — ccxt сам подменяет нужные
                # адреса (testnet.binance.vision для Binance, demo-режим
                # OKX).
                exchange.set_sandbox_mode(True)

        await exchange.load_markets()
        return exchange

    async def _ensure_exchange_connected(self, market_type: str) -> ccxt.Exchange | None:
        """
        Клиент нужного рынка, подключая его ПО ТРЕБОВАНИЮ, если ещё нет —
        в отличие от initialize() (который лениво поднимает второй клиент
        только для рынков, найденных среди позиций, восстановленных при
        СТАРТЕ процесса), это нужно при открытии НОВОЙ позиции на рынке,
        для которого клиента ещё не существует вовсе (например: сигнал
        Telegram-канала, настроенного на futures, а весь текущий тумблер и
        все текущие real-позиции — на споте — на фьючерсы за всё время
        работы процесса ещё не заходили).
        """
        existing = self._exchanges.get(market_type)
        if existing is not None:
            return existing
        if self.exchange_id is None:
            return None
        try:
            exchange = await self._connect_exchange(self.exchange_id, market_type)
        except Exception as e:
            logger.error(f"Не удалось подключиться к {self.exchange_id} [{market_type}] для нового ордера: {e}")
            return None
        self._exchanges[market_type] = exchange
        logger.info(
            f"🔗 Execution Engine: лениво подключено к {self.exchange_id} [{market_type}] — "
            f"сигнал требует этот рынок."
        )
        return exchange

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
            # На фьючерсах short — штатная открытая позиция, и её нужно
            # восстанавливать точно так же, как long, иначе она "теряется"
            # при каждом рестарте бота. Решение берётся из СОБСТВЕННОГО
            # market_type ордера (новое поле, привязка позиции к рынку, на
            # котором она реально была открыта), а не из текущего положения
            # тумблера settings.market_type — иначе восстановление зависело
            # бы от того, где сейчас стоит тумблер, а не от того, где
            # позиция реально была открыта (реальный инцидент: спотовые
            # позиции MON/RLUSD/USDC при переключении тумблера на futures
            # обслуживались так, будто они фьючерсные).
            order_market_type = o.market_type or "spot"
            if position_side == "short" and not is_paper and order_market_type == "spot":
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
            # В real-режиме комиссия покупки на споте ЧАСТО списывается из
            # самого купленного актива (в отличие от paper, где комиссия
            # условная и списывается только с cash-баланса, не уменьшая
            # количество) — без вычета восстановленный после рестарта остаток
            # позиции оказывался больше, чем реально лежит на бирже, и первая
            # же попытка его закрыть падала с "Insufficient balance". НО
            # комиссия не всегда в base-валюте (например Bybit нередко берёт
            # её в USDT) — вычитать её тогда так же неверно, как принять
            # "0.39 USDT" за "0.39 ETH": объём мог уйти в отрицательный и
            # схлопнуться до 0.0 (max(0.0, ...) ниже), хотя на бирже реально
            # лежит весь объём нетронутым — реальный инцидент: ETH/USDT,
            # каждый цикл бот пытался продать 0.0, биржа отклоняла ордер с
            # "Data sent for paramter '' is not valid" бесконечно. Открытие
            # (_execute_real_order) вычитает комиссию по тому же условию —
            # реконструкция при рестарте должна давать тот же результат.
            base_currency = symbol.split("/")[0]
            if position_side == "long" and not is_paper and o.fee_currency == base_currency:
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
                "tp_hit_count": 0, "market_type": order_market_type,
            })
            pos["entry_price"] = (
                (pos["entry_price"] * pos["amount"] + price * amount) / (pos["amount"] + amount)
                if (pos["amount"] + amount) else price
            )
            pos["amount"] += amount
            pos["side"] = position_side
            pos["market_type"] = order_market_type
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
        рестарте, сначала отменяем ВСЕ незакрытые условные ордера по
        символу, затем ставим один новый под актуальный остаток/цену.
        Best-effort, как и вся остальная работа с биржевыми SL-ордерами в
        этом классе — сбой здесь не должен мешать запуску.

        orderFilter зависит от рынка позиции: 'tpslOrder' — для спота
        (ccxt/bybit.py: create_order/cancel_order явно документируют его
        как "Valid for spot only"), 'StopOrder' — для фьючерсов (то же
        значение, которое ccxt сам подставляет по умолчанию для
        триггерных ордеров вне зависимости от рынка — без "spot only"
        оговорки для fetch_open_orders).
        """
        for symbol, pos in list(self.real_positions.items()):
            stop_loss = pos.get("stop_loss")
            amount = pos.get("amount") or 0
            if not stop_loss or amount <= 0:
                continue
            exchange = self._exchange_for(pos)
            if exchange is None:
                continue
            order_filter = "StopOrder" if pos.get("market_type", "spot") == "futures" else "tpslOrder"
            try:
                open_orders = await exchange.fetch_open_orders(
                    self._ccxt_symbol(exchange, symbol), params={"orderFilter": order_filter}
                )
                for o in (open_orders or []):
                    await self._cancel_order_safe(symbol, o.get("id"), exchange)
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
        """Закрыть все подключённые соединения с биржей (spot и/или futures)."""
        if not self._exchanges:
            return
        for market_type, exchange in list(self._exchanges.items()):
            try:
                await exchange.close()
            except Exception as e:
                logger.debug(f"Не удалось закрыть соединение с биржей [{market_type}]: {e}")
        self._exchanges = {}
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
        market_type: str | None = None,
        leverage: float | None = None,
    ) -> Order | None:
        """
        Создать ордер.
        Возвращает Order объект (сохранённый в БД) или None.

        market_type — рынок для НОВОЙ позиции (spot/futures), если задан
        явно (например, настройка конкретного Telegram-канала — см.
        _execute_telegram_signal в main.py) — по умолчанию (None) берётся
        settings.market_type (текущий тумблер в шапке дашборда), как и
        раньше для сигналов стратегий/ручных ордеров.

        leverage — плечо ИМЕННО для этого ордера (например, канал явно
        указал его в тексте сигнала — см. _execute_telegram_signal), если
        задано И рынок фьючерсный — используется вместо глобальной
        settings.futures_leverage (см. _execute_real_order). Игнорируется
        на споте (там плеча не существует) и в paper-режиме (там нет
        реального маржинального механизма — см. _execute_paper_order).
        """
        if risk_manager.state.kill_switch_active:
            logger.warning(f"❌ Попытка создать ордер при активном kill switch: {symbol}")
            return None

        if not self.can_execute():
            logger.warning(f"❌ Исполнение отклонено: {symbol} {side}")
            return None

        client_order_id = str(uuid.uuid4())[:12]
        order_market_type = market_type or settings.market_type

        # Получить цену исполнения
        execution_price = price
        if order_type == "market" and execution_price is None:
            try:
                if self.is_paper:
                    ticker = await self.exchange.fetch_ticker(self._ccxt_symbol(self.exchange, symbol))
                else:
                    # Тикер снимаем с клиента ЦЕЛЕВОГО рынка ордера, а не
                    # текущего тумблера — иначе для канала с market_type,
                    # отличным от тумблера, цена читалась бы не с той пары
                    # (у фьючерсов и спота разный спред/цена).
                    ticker_exchange = await self._ensure_exchange_connected(order_market_type)
                    if ticker_exchange is None:
                        raise RuntimeError(f"нет подключения к бирже для рынка '{order_market_type}'")
                    ticker = await ticker_exchange.fetch_ticker(self._ccxt_symbol(ticker_exchange, symbol))
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
            "market_type": order_market_type,
            "leverage": leverage,
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

    def _market_limits(self, symbol: str, exchange: ccxt.Exchange | None = None) -> dict | None:
        """
        market["limits"] для symbol, если оно есть и имеет ожидаемую форму
        словаря — иначе None. Общий защитный доступ для
        _below_exchange_minimum/reconcile_real_positions: структура
        markets[symbol] не гарантирована (разные биржи, тестовые mock-объекты
        без выставленного .markets), а падать из-за необязательной проверки
        не должны ни отправка ордера, ни сверка позиций.

        exchange по умолчанию — self.exchange (клиент ТЕКУЩЕГО тумблера).
        Для ордера/позиции на рынке, отличном от текущего тумблера (напр.
        Telegram-канал настроен на другой рынок — см. _execute_real_order/
        _reconcile_futures_position/_reconcile_spot_position), нужно
        передать явно клиент ЕЁ рынка — иначе лимиты читались бы со
        structurally другого markets dict чужого рынка.
        """
        try:
            ex = exchange if exchange is not None else self.exchange
            markets = ex.markets if ex else None
            if not isinstance(markets, dict):
                return None
            market = markets.get(self._ccxt_symbol(ex, symbol))
            if not isinstance(market, dict):
                return None
            limits = market.get("limits")
            return limits if isinstance(limits, dict) else None
        except Exception:
            return None

    def _market_min_amount(self, symbol: str, exchange: ccxt.Exchange | None = None) -> float | None:
        limits = self._market_limits(symbol, exchange)
        if not limits:
            return None
        amount_limits = limits.get("amount")
        min_amount = amount_limits.get("min") if isinstance(amount_limits, dict) else None
        return min_amount if isinstance(min_amount, (int, float)) else None

    def _below_exchange_minimum(
        self, symbol: str, amount: float, price: float | None, exchange: ccxt.Exchange | None = None,
    ) -> str | None:
        """
        Проверить объём/стоимость ордера ПРОТИВ биржевых лимитов пары ДО
        отправки запроса — иначе биржа отклоняет ордер (напр. Bybit
        retCode 170140 "Order value exceeded lower limit"), это летит в
        логи как ERROR, будто сломался код, а на деле объём просто
        занижен: посчитанного от текущего доступного баланса (size_pct%
        от него) размера позиции не хватает даже на минимальный
        допустимый на бирже ордер по этой паре — сигнал безопасно
        пропускается, ошибка это ожидаемая при малом остатке средств.

        exchange — клиент РЫНКА ЭТОГО ОРДЕРА (может отличаться от текущего
        тумблера — напр. Telegram-канал настроен на другой рынок, см.
        _execute_real_order); по умолчанию (None) — self.exchange, как и
        раньше для сигналов без явного override.

        Это вспомогательная, необязательная проверка: структура markets[symbol]
        не гарантирована (разные биржи, неполные тестовые/тестовые-mock
        объекты) — при любой неожиданности просто не блокируем ордер,
        оставляя решение самой бирже, как и раньше.
        """
        try:
            min_amount = self._market_min_amount(symbol, exchange)
            if isinstance(min_amount, (int, float)) and amount < min_amount:
                return f"объём {amount:.8f} {symbol.split('/')[0]} меньше минимального ({min_amount})"
            limits = self._market_limits(symbol, exchange) or {}
            cost_limits = limits.get("cost")
            min_cost = cost_limits.get("min") if isinstance(cost_limits, dict) else None
            if isinstance(min_cost, (int, float)) and price and amount * price < min_cost:
                return f"стоимость ордера {amount * price:.4f} USDT меньше минимальной по паре ({min_cost} USDT)"
        except Exception as e:
            logger.debug(f"Не удалось проверить минимальные лимиты биржи для {symbol}: {e}")
        return None

    async def _fetch_fill_details_via_trades(
        self, order_id: str | None, symbol: str, exchange: ccxt.Exchange | None = None,
        attempts: int = 6, delay: float = 1.5,
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
            result = await self._fetch_fill_details_via_trades_once(order_id, symbol, exchange)
            if result is not None:
                return result
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
        return None

    async def _fetch_fill_details_via_trades_once(
        self, order_id: str, symbol: str, exchange: ccxt.Exchange | None = None,
    ) -> dict | None:
        ex = exchange if exchange is not None else self.exchange
        ccxt_symbol = self._ccxt_symbol(ex, symbol)
        try:
            trades = None
            try:
                trades = await ex.fetch_order_trades(order_id, ccxt_symbol)
            except Exception:
                trades = None
            if not trades:
                recent = await ex.fetch_my_trades(ccxt_symbol, limit=10)
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
        amount_requested: float | None = None,
    ) -> tuple[float, str | None]:
        """
        Комиссию исполнения нужно брать РЕАЛЬНУЮ с биржи (order["fee"] или
        сумма комиссий из истории сделок — см. _fetch_fill_details_via_trades,
        которая уже сама повторяет запрос несколько раз с паузой, если
        биржа не успела отдать комиссию сразу) — если её не удалось
        получить НИОТКУДА (ни один из источников не дал cost даже после
        повторов), не оставляем 0 (иначе PnL был бы завышен на величину
        реальной, но неучтённой комиссии биржи), а пробуем по убыванию
        точности:

        1) Разница между запрошенным и фактически исполненным объёмом (в
           base-валюте) — на споте комиссия покупки часто списывается
           биржей из самого актива ДО того, как объём попадает в
           order["filled"]/историю сделок, так что эта разница и есть
           фактическая комиссия, точнее стандартной ставки ниже. Валиден,
           только если запрошено СТРОГО больше исполненного — иначе (объём
           совпал или исполнилось больше, к комиссии отношения не имеющая
           ситуация) сигнала нет.
        2) Стандартная ставка spot-таксы (то же приближение, что и
           paper_fee_pct в paper-режиме). Валюта ЭТОЙ оценки — всегда
           quote, а не "base для покупки, quote для продажи" (как для
           настоящей комиссии с биржи, где такое допущение имеет смысл):
           estimated = filled_amount(base) × fill_price(quote/base) × pct —
           это ЧИСЛО в quote-валюте по построению формулы, независимо от
           side. Реальный инцидент: HYPE/USDT, buy — оценочная комиссия
           0.37 USDT была помечена как "0.37 HYPE" (~31 USDT по факту), из-
           за чего close_real_position (конвертирует комиссию открытия в
           USDT-эквивалент, ТОЛЬКО если её валюта — base) домножила и без
           того неверно про-labeled число ЕЩЁ РАЗ на entry_price — реально
           прибыльный Take Profit 1 показал PnL -13.47 вместо примерно +2.
        """
        fee_info = fee_info or {}
        cost = fee_info.get("cost")
        if cost:
            return float(cost), fee_info.get("currency")
        base_currency, quote_currency = symbol.split("/")
        # Только для buy: комиссия покупки на споте обычно списывается из
        # base-валюты (полученного актива) ДО того, как объём попадает в
        # order["filled"]. Для sell это допущение неверно — там комиссия
        # обычно из quote (полученной от продажи), а не из проданного
        # base-объёма, так что разница запрошенного/исполненного там —
        # просто округление лота, не комиссия.
        if side == "buy" and amount_requested is not None and amount_requested > filled_amount:
            return amount_requested - filled_amount, base_currency
        estimated = filled_amount * fill_price * (settings.paper_fee_pct / 100)
        return estimated, quote_currency

    async def _place_stop_loss_order(
        self, symbol: str, amount: float, stop_loss_price: float,
        exchange: ccxt.Exchange | None = None, side: str = "long", is_futures: bool = False,
    ) -> str | None:
        """
        Разместить биржевой стоп-ордер — условный reduceOnly-ордер
        (params={"stopLossPrice": ...}), рыночный по достижении
        stop_loss_price — чтобы защита позиции не зависела от того, жив ли
        процесс бота и успевает ли внутренний поллинг цены
        (_check_position_exit в main.py) её отследить. Тейк-профиты
        (TP1/TP2/TP3) сознательно остаются только во внутренней логике —
        ни на споте, ни на фьючерсах: у Bybit нет родного OCO-механизма
        частичного выхода по нескольким уровням, один статичный биржевой
        TP-ордер такому сценарию не соответствует, а SL — соответствует
        (единственный уровень, который в момент установки актуален всегда).

        Направление зависит от side: "long" защищается sell-стопом (спот и
        фьючерсы одинаково — единственный вариант на споте), "short" —
        buy-стопом (только фьючерсы, обратный к открытию). ccxt сам
        вычисляет triggerDirection из side+stopLossPrice (подтверждено
        чтением ccxt/bybit.py create_order_request: side="sell" даёт
        triggerDirection=2/fall — верно для long, side="buy" даёт
        triggerDirection=1/rise — верно для short), явно его передавать не
        нужно.

        ВАЖНО: Bybit НЕ поддерживает stopLoss/takeProfit, прикреплённые к
        самому маркет-ордеру (ccxt бросает InvalidOrder) — это ОТДЕЛЬНЫЙ
        условный ордер, размещаемый уже после того, как позиция открыта.

        Best-effort: любая ошибка (биржа отклонила триггер-цену, не
        поддерживается для этой пары и т.п.) не должна блокировать саму
        позицию — просто логируем и остаёмся под защитой одной внутренней
        проверки, как было раньше.
        """
        if amount <= 0 or not stop_loss_price:
            return None
        ex = exchange if exchange is not None else self.exchange
        if ex is None:
            return None
        # Проверка доступного остатка на кошельке — спот-специфична (на
        # фьючерсах позиция не выражается остатком монеты на кошельке, там
        # нечего сверять). Отслеживаемый объём позиции мог немного
        # разойтись с реальным остатком на бирже — та же причина, что и в
        # close_real_position (комиссии, округление лота, накопленный
        # дрейф за несколько частичных закрытий или рестартов процесса):
        # условный SL-ордер на биржевой остаток, а не на устаревший
        # расчётный объём — иначе биржа отклоняет ЕГО ЦЕЛИКОМ с
        # "Insufficient balance", и позиция остаётся вовсе без биржевой
        # защиты (реальный инцидент: XAUT/USDT, LINK/USDT после нескольких
        # частичных TP).
        if not is_futures:
            try:
                base_currency = symbol.split("/")[0]
                balance = await ex.fetch_balance()
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
            params: dict = {"stopLossPrice": stop_loss_price}
            if is_futures:
                # reduceOnly — та же защита от переворота позиции, что и в
                # close_real_position (ccxt и так проставляет его сам в
                # этой ветке, см. докстринг выше, но передаём явно — тот
                # же стиль, что и там).
                params["reduceOnly"] = True
            closing_side = "sell" if side == "long" else "buy"
            ccxt_symbol = self._ccxt_symbol(ex, symbol)
            order = (
                await ex.create_market_sell_order(ccxt_symbol, amount, params=params)
                if closing_side == "sell"
                else await ex.create_market_buy_order(ccxt_symbol, amount, params=params)
            )
            order_id = order.get("id") if order else None
            if order_id:
                logger.info(
                    f"🛡️ Биржевой SL выставлен: {symbol} {closing_side} {amount:.8f} @ триггер "
                    f"{stop_loss_price} (ордер {order_id})"
                )
            return order_id
        except Exception as e:
            logger.warning(
                f"⚠️ Не удалось выставить биржевой SL для {symbol} (триггер {stop_loss_price}): {e} "
                f"— позиция защищена только внутренним поллингом цены."
            )
            return None

    async def _cancel_order_safe(
        self, symbol: str, order_id: str | None, exchange: ccxt.Exchange | None = None,
    ) -> bool:
        """
        Best-effort отмена ордера — он мог уже исполниться или быть отменённым,
        это не ошибка. Возвращает True, если на бирже ордера точно больше нет
        (отмена прошла, или биржа говорит "не найден" — OrderNotFound), и False
        при любой ДРУГОЙ ошибке — тогда ордер мог остаться живым на бирже, и
        вызывающему коду нельзя считать его order_id безопасным для сброса
        (см. sync_stop_loss_order — забыть ID в этом случае значит осиротить
        реальный условный ордер на бирже).
        """
        if not order_id:
            return True
        ex = exchange if exchange is not None else self.exchange
        if ex is None:
            return False
        try:
            await ex.cancel_order(order_id, self._ccxt_symbol(ex, symbol))
            return True
        except ccxt.OrderNotFound:
            return True
        except Exception as e:
            logger.debug(f"Не удалось отменить ордер {order_id} ({symbol}) — возможно, уже неактивен: {e}")
            return False

    async def sync_stop_loss_order(self, symbol: str, amount: float, stop_loss_price: float | None) -> None:
        """
        Пересоздать биржевой SL-ордер под текущий остаток/цену позиции —
        нужно после частичного закрытия (TP1/TP2 уменьшают объём) и после
        переноса SL в безубыток (см. _check_position_exit в main.py):
        старый биржевой ордер продавал бы либо неверный объём, либо по
        неверной, уже неактуальной цене. Отменяет прежний отслеживаемый
        SL-ордер (если был) и, если задан stop_loss_price и остаток > 0,
        ставит новый. Клиент резолвится по СОБСТВЕННОМУ market_type позиции
        (_exchange_for), а не по текущему тумблеру settings.market_type —
        позиция могла быть открыта на рынке, отличном от текущего.

        На фьючерсах SL теперь тоже ставится — _place_stop_loss_order сам
        выбирает направление ордера по side позиции (sell для long, buy
        для short), симметрично споту (см. докстринг _place_stop_loss_order).
        Раньше здесь была защита от реального инцидента (восстановление
        short-позиции ENA/USDT после рестарта на фьючерсах получало
        спотовый "sell"-стоп — семантически неверно для шорта); теперь
        направление определяется явно по side, а не всегда sell, так что
        сама причина того инцидента устранена в _place_stop_loss_order.
        """
        pos = self.real_positions.get(symbol)
        if pos is None:
            return
        exchange = self._exchange_for(pos)
        old_sl_order_id = pos.get("sl_order_id")
        if old_sl_order_id and not await self._cancel_order_safe(symbol, old_sl_order_id, exchange):
            # Отмена не подтверждена — старый условный SL-ордер мог остаться
            # живым на бирже. Забыть его ID здесь (как раньше) означало бы
            # осиротить реальный ордер: он продолжает резервировать
            # объём/маржу на бирже, но бот больше не знает о его
            # существовании и не может ни отменить его, ни обнаружить его
            # срабатывание через _finalize_externally_closed_position.
            # Реальный инцидент (прод): именно так осиротели SL-ордера
            # BCH/USDT и HYPE/USDT после первого же частичного TP — после
            # этого КАЖДАЯ попытка переставить SL или закрыть остаток
            # позиции стабильно падала с "insufficient balance" (Bybit
            # резервирует объём под уже существующий условный ордер), а
            # реконсиляция не могла подобрать позицию, потому что не знала
            # order_id, по которому нужно проверять исполнение.
            logger.warning(
                f"⚠️ Не удалось подтвердить отмену старого SL-ордера {old_sl_order_id} ({symbol}) — "
                f"оставляем его отслеживаемым и не выставляем новый в этом цикле, чтобы не осиротить его."
            )
            return
        pos["sl_order_id"] = None
        if stop_loss_price and amount > 0:
            pos["sl_order_id"] = await self._place_stop_loss_order(
                symbol, amount, stop_loss_price, exchange,
                side=pos.get("side", "long"),
                is_futures=pos.get("market_type", "spot") == "futures",
            )

    async def _confirm_fill_via_balance(
        self, symbol: str, side: str, balance_before: float, expected_amount: float,
        exchange: ccxt.Exchange | None = None,
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
        ex = exchange if exchange is not None else self.exchange
        try:
            balance = await ex.fetch_balance()
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

        # Рынок этого КОНКРЕТНОГО ордера — либо явно передан вызывающим
        # кодом (напр. настройка Telegram-канала, см. create_order/
        # _execute_telegram_signal в main.py), либо (market_type не задан)
        # текущий тумблер settings.market_type — как и раньше для сигналов
        # стратегий/ручных ордеров.
        order_market_type = order_data.get("market_type") or settings.market_type
        is_futures = order_market_type == "futures"
        exchange = await self._ensure_exchange_connected(order_market_type)
        if exchange is None:
            logger.error(
                f"❌ Реальный ордер {symbol} отклонён: нет подключения к бирже для рынка "
                f"'{order_market_type}'."
            )
            return None

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
        # же ошибкой на каждой попытке закрытия. На фьючерсах (ЭТАП 2)
        # side=="sell" — это штатное открытие короткой позиции, а не
        # продажа несуществующего актива, поэтому защита теперь только
        # спотовая.
        if not is_futures and side != "buy":
            logger.error(
                f"❌ Реальный ордер {symbol} отклонён: сторона '{side}' (шорт) не поддерживается "
                f"на споте — на споте нет встроенного шорта, а продажа при наличии баланса актива "
                f"реально исполнилась бы, распродав реальные средства без возможности закрыть "
                f"'позицию' обратно."
            )
            return None

        leverage_to_set = settings.futures_leverage
        if is_futures:
            # Плечо ПО УМОЛЧАНИЮ — глобальная настройка (settings.
            # futures_leverage), но конкретный сигнал (например,
            # Telegram-канал, явно указавший "Кредитное плечо: х35" в
            # тексте — см. _execute_telegram_signal) может задать своё,
            # per-ордерное — используем его, если есть. best-effort:
            # некоторые биржи/версии ccxt бросают исключение, если плечо
            # уже установлено в то же значение ("leverage not modified") —
            # это не ошибка, ордер всё равно можно размещать с уже
            # действующим плечом.
            leverage_to_set = order_data.get("leverage") or settings.futures_leverage
            try:
                await exchange.set_leverage(int(leverage_to_set), self._ccxt_symbol(exchange, symbol))
            except Exception as e:
                logger.debug(f"Не удалось установить плечо {leverage_to_set}x для {symbol}: {e}")

        below_min = self._below_exchange_minimum(symbol, amount, price, exchange)
        if below_min:
            logger.warning(
                f"⚠️ Реальный ордер {symbol} пропущен: {below_min} — доступного баланса "
                f"недостаточно для минимального размера ордера по этой паре."
            )
            return None

        # Баланс базовой валюты на СПОТОВОМ кошельке (для фолбэк-подтверждения
        # исполнения по изменению баланса ниже) не имеет смысла на фьючерсах —
        # там позиция это отдельная сущность (fetch_positions), а не остаток
        # монеты на кошельке; открытие/закрытие меняет маржу в USDT, а не
        # баланс базовой валюты. На фьючерсах фолбэк просто не пробуем.
        balance_before = None
        if not is_futures:
            try:
                snapshot = await exchange.fetch_balance()
                balance_before = self._extract_currency_balance(snapshot, symbol.split("/")[0])
            except Exception as e:
                logger.debug(f"Не удалось снять баланс {symbol} до отправки ордера: {e}")

        try:
            ccxt_symbol = self._ccxt_symbol(exchange, symbol)
            if order_data["type"] == "market":
                if side == "buy":
                    order = await exchange.create_market_buy_order(ccxt_symbol, amount)
                else:
                    order = await exchange.create_market_sell_order(ccxt_symbol, amount)
            elif order_data["type"] == "limit":
                if side == "buy":
                    order = await exchange.create_limit_buy_order(ccxt_symbol, price, amount)
                else:
                    order = await exchange.create_limit_sell_order(ccxt_symbol, price, amount)
            else:
                logger.error(f"Неизвестный тип ордера: {order_data['type']}")
                return None

            order = await self._fetch_confirmed_order(order, symbol, exchange)
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
            trade_fill = await self._fetch_fill_details_via_trades(order.get("id"), symbol, exchange)
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
                    await self._confirm_fill_via_balance(symbol, side, balance_before, amount, exchange)
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
            fill_fee, fee_currency = self._resolve_fee(
                order.get("fee"), filled_amount, fill_price, side, symbol, amount_requested=amount,
            )
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
                    market_type=order_market_type,
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
            # На спот сюда доходит только side=="buy" (short отклонён выше),
            # на фьючерсах — обе стороны: "sell" здесь означает открытие
            # короткой позиции, а не продажу существующего актива.
            self.real_positions[symbol] = {
                "amount": net_amount,
                "entry_price": fill_price,
                "side": "long" if side == "buy" else "short",
                "strategy_id": order_data.get("strategy_id"),
                "stop_loss": order_data.get("stop_loss"),
                "take_profit": order_data.get("take_profit"),
                "order_id": order_id,
                "entry_fee": fill_fee,
                "opened_at": utcnow(),
                "sl_order_id": None,
                # Рынок, на котором позиция РЕАЛЬНО открыта — привязка к
                # order_market_type ЭТОГО ордера (текущий тумблер по
                # умолчанию, либо явный override вызывающего кода — см.
                # create_order), а не постоянная ссылка на текущий тумблер:
                # дальнейшее ведение позиции (SL, закрытие) использует
                # именно это поле через _exchange_for(pos), даже если
                # тумблер потом переключат на другой рынок.
                "market_type": order_market_type,
                # Плечо, реально запрошенное у биржи для ЭТОГО ордера (см.
                # leverage_to_set выше — per-сигнал или глобальный дефолт).
                # Только для отображения сразу после открытия — следующий
                # цикл сверки (_reconcile_futures_position) перезапишет
                # его подтверждённым биржей значением из fetch_position();
                # на споте плеча не бывает, оставляем None.
                "leverage": leverage_to_set if is_futures else None,
            }
            # Биржевой SL теперь ставится для обоих рынков —
            # sync_stop_loss_order сама выбирает направление ордера по
            # side позиции (уже зарегистрирована выше).
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

        На споте — только long: там нет встроенного шорта, а
        _execute_real_order уже не позволяет открыть short-позицию
        (create_market_sell_order без имеющегося актива на споте просто
        упадёт с ошибкой недостатка баланса — исполнение вернёт None и
        позиция никогда не будет создана). На фьючерсах поддержаны обе
        стороны: long закрывается продажей, short — покупкой (buy to
        cover), обе — с reduceOnly, чтобы рассинхрон объёма не открыл
        встречную позицию вместо закрытия текущей.

        Рынок и ccxt-клиент резолвятся по СОБСТВЕННОМУ market_type
        отслеживаемой позиции (_exchange_for), а не по текущему положению
        тумблера settings.market_type — позиция могла быть открыта на
        рынке, отличном от текущего (реальный инцидент: спотовые позиции
        MON/RLUSD/USDC при переключении тумблера на futures обслуживались
        так, будто они фьючерсные).
        """
        tracked_pos = self.real_positions.get(symbol)
        market_type = tracked_pos.get("market_type", "spot") if tracked_pos is not None else settings.market_type
        is_futures = market_type == "futures"
        exchange = self._exchange_for(tracked_pos)
        if not is_futures and side != "long":
            logger.error(f"close_real_position: закрытие {side}-позиции не поддерживается на споте: {symbol}")
            return None
        closing_side = "sell" if side == "long" else "buy"

        # Отслеживаемый объём уже 0 (или отрицательный из-за накопленной
        # погрешности) — продавать нечего, а create_market_sell_order с
        # нулевым количеством биржа отклоняет ("Data sent for paramter ''
        # is not valid" у Bybit) — без этой проверки любой источник такого
        # объёма (реальный инцидент: ошибка реконструкции позиции при
        # рестарте — см. _load_open_positions_from_db) заставлял бота
        # повторять один и тот же провальный ордер на каждой итерации цикла
        # бесконечно, вместо того чтобы один раз снять позицию с учёта.
        if amount <= 0:
            logger.warning(
                f"⚠️ Закрытие {symbol} пропущено: отслеживаемый объём уже {amount:.8f} — "
                f"нечего продавать, снимаем позицию с учёта без ордера на бирже."
            )
            await self._reconcile_phantom_position(symbol, order_open_id)
            return None

        # Отменяем биржевой SL-ордер (если был) ДО собственной продажи —
        # иначе он остаётся висеть параллельно с этим закрытием (не важно,
        # по какой причине оно происходит — TP, ручное закрытие или сам же
        # SL) и может конфликтовать за один и тот же остаток базовой валюты.
        # Если отмена не подтверждена (ambiguous-ошибка, не "ордера уже
        # нет") — собственная продажа заведомо конфликтовала бы с ещё живым
        # SL-ордером на тот же объём и упала бы с той же ошибкой биржи
        # (реальный инцидент: CHIP/USDT — SL остался живым и отслеживаемым,
        # но КАЖДАЯ последующая попытка close_real_position всё равно
        # стабильно падала insufficient balance, потому что сама попытка
        # отмены здесь тоже не проходила, а код раньше не проверял её
        # результат и лез продавать поверх всё ещё активного SL). Лучше
        # пропустить попытку сейчас и повторить на следующем цикле, чем
        # штамповать заведомо провальные ордера.
        if tracked_pos is not None:
            old_sl_order_id = tracked_pos.get("sl_order_id")
            if old_sl_order_id and not await self._cancel_order_safe(symbol, old_sl_order_id, exchange):
                logger.warning(
                    f"⚠️ Закрытие {symbol} отложено: не удалось подтвердить отмену биржевого "
                    f"SL-ордера {old_sl_order_id} — собственная продажа сейчас неизбежно "
                    f"конфликтовала бы с ним же. Попробуем снова на следующем цикле."
                )
                return None

        if exchange is None:
            logger.error(
                f"❌ Не удалось закрыть реальную позицию {symbol}: нет подключения к бирже "
                f"для рынка '{market_type}'."
            )
            return None

        # Отслеживаемый объём позиции — оценка (комиссии, округление лота
        # биржей и т.п. могут понемногу расходиться с реальным остатком) —
        # без подстраховки продажа "полного" объёма падает на бирже с
        # "Insufficient balance", и позиция навсегда зависает открытой,
        # хотя реально продать почти всё, что есть, всё равно можно.
        # Спот-специфично: на фьючерсах не нужно "владеть" монетой, чтобы
        # закрыть контракт — сверять тут нечего, closing_amount = amount.
        closing_amount = amount
        available = None
        if not is_futures:
            try:
                base_currency = symbol.split("/")[0]
                balance = await exchange.fetch_balance()
                available = self._extract_currency_balance(balance, base_currency)
                if 0 < available < amount:
                    logger.warning(
                        f"⚠️ Доступный баланс {base_currency} ({available:.8f}) меньше отслеживаемого "
                        f"объёма позиции {symbol} ({amount:.8f}) — продаём доступный остаток."
                    )
                    closing_amount = available
            except Exception as e:
                logger.debug(f"Не удалось сверить доступный баланс перед закрытием {symbol}: {e}")

        try:
            ccxt_symbol = self._ccxt_symbol(exchange, symbol)
            if is_futures:
                # reduceOnly — защита от переворота позиции вместо закрытия,
                # если объём вдруг разошёлся с тем, что реально открыто на бирже.
                params = {"reduceOnly": True}
                order = (
                    await exchange.create_market_sell_order(ccxt_symbol, closing_amount, params=params)
                    if closing_side == "sell"
                    else await exchange.create_market_buy_order(ccxt_symbol, closing_amount, params=params)
                )
            else:
                order = await exchange.create_market_sell_order(ccxt_symbol, closing_amount)
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
            #    of 0.001"). Реальный инцидент: USDC/USDT, наш учёт всего
            #    0.00717323 при доступных на бирже 14092.25 — available
            #    здесь заведомо НЕ меньше amount (сам отслеживаемый объём —
            #    пыль, а не нехватка баланса), поэтому проверка НЕ требует
            #    available < amount: продать меньше минимального объёма
            #    нельзя вне зависимости от того, сколько ещё есть на бирже.
            # 2) объёма достаточно (available >= amount), но его СТОИМОСТЬ в
            #    quote-валюте (amount * текущая цена) ниже минимальной для
            #    пары — Bybit отвечает retCode 170140 "Order value exceeded
            #    lower limit" (реальный инцидент: SUI/USDT, ~26 минут подряд
            #    одна и та же ошибка каждые ~70с). Деление позиции на более
            #    мелкие ордера её не решает, наоборот, ещё уменьшает
            #    стоимость каждого.
            # Дуст-списание (_reconcile_phantom_position) — спот-специфичная
            # концепция (биржевые лимиты на минимальный ОБЪЁМ базовой
            # валюты на кошельке), на фьючерсах контрактов "пыль" в этом
            # смысле не бывает. Для фьючерсов просто оставляем позицию
            # отслеживаемой — следующая проверка SL/TP (main.py) попробует
            # закрыть её снова, вместо необратимого списания без PnL.
            unsellable_dust = not is_futures and (
                available == 0
                or any(kw in str(e).lower() for kw in ("precision", "minimum"))
                or "lower limit" in str(e).lower()
            )
            if unsellable_dust:
                await self._reconcile_phantom_position(symbol, order_open_id)
            return None

        order = await self._fetch_confirmed_order(order, symbol, exchange)
        trade_ids: list[str] | None = None
        # Историю сделок биржи пробуем ВСЕГДА (см. тот же приоритет и
        # обоснование при открытии в _execute_real_order) — это то же самое,
        # что видно как Filled Price/комиссия в истории сделок на самой
        # бирже, и точнее, чем order["average"]/order["fee"] из fetch_order.
        trade_fill = await self._fetch_fill_details_via_trades(order.get("id"), symbol, exchange)
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
                await self._confirm_fill_via_balance(symbol, closing_side, available, closing_amount, exchange)
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
        exit_fee, exit_fee_currency = self._resolve_fee(
            order.get("fee"), exit_filled_amount, exit_price, closing_side, symbol, amount_requested=closing_amount,
        )

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

        # Для linear USDT-perp плечо не входит в формулу PnL напрямую (оно
        # влияет только на требуемую маржу под позицию, не на сам PnL) —
        # эта же формула (с точностью до знака по стороне) верна и для
        # фьючерсов. Ветка для short зеркальна paper-версии
        # (close_paper_position: pnl = (entry - exit) * amount - fees).
        if side == "long":
            pnl = (exit_price - entry_price) * amount - entry_fee_quote - exit_fee
        else:
            pnl = (entry_price - exit_price) * amount - entry_fee_quote - exit_fee
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
                side=closing_side,
                order_type="market",
                amount=amount,
                price=exit_price,
                status="filled",
                filled_amount=order["filled"] or amount,
                filled_price=exit_price,
                fee=exit_fee,
                fee_currency=exit_fee_currency,
                market_type=market_type,
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

    async def recalculate_closed_trade(self, trade_id: int) -> dict | None:
        """
        Перезапросить у биржи реальные цену/объём/комиссию для открывающего
        ордера и КАЖДОЙ "ноги" закрытия позиции (Trade.order_open_id может
        быть общим для нескольких строк Trade — частичные закрытия по
        уровням TP1/TP2/TP3, см. GET /trades/{id}/detail) и пересчитать
        PnL каждой — ручной способ подтянуть точные данные постфактум для
        сделки, изначально записанной по оценке (биржа не успела вовремя
        отдать комиссию/цену — см. _resolve_fee, "не подтверждён по
        filled" и т.п.), не дожидаясь следующего похожего инцидента.
        Только для real — в paper реальных данных с биржи для сверки нет
        вообще.

        Возвращает {"updated": bool, "pnl", "pnl_pct", "outcome"} —
        суммарные PnL/PnL% по всей группе (updated False, если у биржи
        так и не нашлось ничего нового ни по одному из ордеров — как и
        остальные best-effort сверки с биржей в этом классе) или None,
        если сделка/её открывающий ордер не найдены.
        """
        if self.is_paper:
            return None

        async with get_session() as session:
            trade = await session.get(Trade, trade_id)
            if trade is None or trade.order_open_id is None:
                return None
            symbol_row = await session.get(Symbol, trade.symbol_id)
            symbol = symbol_row.symbol if symbol_row else None
            opening_order = await session.get(Order, trade.order_open_id)
            if not symbol or opening_order is None:
                return None

            legs = (
                await session.execute(select(Trade).where(Trade.order_open_id == trade.order_open_id))
            ).scalars().all()
            closing_orders = {}
            for leg in legs:
                if leg.order_close_id is not None:
                    closing_orders[leg.id] = await session.get(Order, leg.order_close_id)

            refreshed = False
            for order in [opening_order, *closing_orders.values()]:
                if order is None or not order.order_id_exchange:
                    continue
                # order_id_exchange хранит либо ID самого ордера (обычный
                # случай, когда история сделок биржи не нашлась сразу — см.
                # _execute_real_order/close_real_position), либо список ID
                # СДЕЛОК через запятую, если она уже нашлась тогда же —
                # пересчитывать в этом случае уже нечего (данные и так
                # точные), fetch_order_trades по ID сделки просто ничего не
                # найдёт, что здесь безопасно эквивалентно "нового нет".
                order_ref = order.order_id_exchange.split(",")[0]
                fill = await self._fetch_fill_details_via_trades(order_ref, symbol)
                if fill is None:
                    continue
                order.filled_price = fill["average"]
                order.filled_amount = fill["amount"]
                fee = fill.get("fee") or {}
                order.fee = fee.get("cost")
                order.fee_currency = fee.get("currency")
                if fill.get("trade_ids"):
                    order.order_id_exchange = ",".join(fill["trade_ids"])
                refreshed = True

            if not refreshed:
                return {"updated": False}

            base_currency = symbol.split("/")[0]
            total_amount = 0.0
            total_pnl = 0.0
            for leg in legs:
                closing_order = closing_orders.get(leg.id)
                if closing_order is None or opening_order.filled_price is None or closing_order.filled_price is None:
                    total_amount += float(leg.amount)
                    total_pnl += float(leg.pnl)
                    continue
                entry_price = float(opening_order.filled_price)
                exit_price = float(closing_order.filled_price)
                # closing_order соответствует ровно этой "ноге" один к
                # одному — берём именно его объём как источник истины (мог
                # чуть отличаться от ранее записанного Trade.amount), а не
                # оставляем устаревшее значение.
                amount = float(closing_order.filled_amount) if closing_order.filled_amount else float(leg.amount)
                entry_fee = (
                    float(opening_order.fee or 0) * (amount / float(opening_order.filled_amount))
                    if opening_order.filled_amount else 0.0
                )
                exit_fee = float(closing_order.fee or 0)
                # Та же конвертация комиссии открытия в USDT-эквивалент,
                # что и в close_real_position — иначе base-валютная
                # комиссия ("105.49 TAC") считалась бы quote-валютной.
                entry_fee_quote = (
                    entry_fee * entry_price if opening_order.fee_currency == base_currency else entry_fee
                )

                leg.amount = amount
                leg.entry_price = entry_price
                leg.exit_price = exit_price
                pnl = (exit_price - entry_price) * amount - entry_fee_quote - exit_fee
                leg.pnl = pnl
                leg.pnl_pct = (pnl / (entry_price * amount) * 100) if entry_price and amount else 0.0
                leg.outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "break-even")
                total_amount += amount
                total_pnl += pnl

            await session.commit()

        total_pnl_pct = (total_pnl / (float(opening_order.filled_price) * total_amount) * 100) if total_amount else 0.0
        outcome = "win" if total_pnl > 0 else ("loss" if total_pnl < 0 else "break-even")
        logger.info(
            f"🔄 Сделка #{trade_id} ({symbol}) пересчитана по данным биржи: "
            f"PnL {total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
        )
        return {"updated": True, "pnl": total_pnl, "pnl_pct": total_pnl_pct, "outcome": outcome}

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

        Также отменяем биржевой SL-ордер позиции (если был выставлен, см.
        sync_stop_loss_order) — без этого он навсегда остаётся висеть на
        бирже: мы больше никогда не вернёмся к этому symbol, чтобы его
        закрыть/отменить обычным путём, а сам ордер продолжает держать часть
        актива в "used"-балансе. Реальный симптом (прод): в новом списке
        балансов дашборда число валют с ненулевым "в ордерах" оказалось
        БОЛЬШЕ числа фактически открытых позиций — это и есть накопленные за
        время работы бота осиротевшие SL-ордера от прошлых списаний.
        """
        pos = self.real_positions.pop(symbol, None)
        if pos is not None:
            await self._cancel_order_safe(symbol, pos.get("sl_order_id"), self._exchange_for(pos))
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
        Сверить ВСЕ отслеживаемые реальные позиции (спот и фьючерсы, каждая
        через клиент СВОЕГО рынка — см. _exchange_for) с фактическим
        состоянием на бирже и снять с учёта те, которые обычным закрытием
        уже невозможно закрыть — ДО того, как расхождение попадёт в
        _compute_equity()/просадку в main.py.

        До этого сверка была чисто РЕАКТИВНОЙ: она случалась только в
        момент попытки закрыть позицию (close_real_position), то есть
        только когда сработает SL/TP. Если цена никогда не доходила до
        SL/TP, испорченная позиция (раздутый/не совпадающий с биржей
        amount) могла висеть в real_positions неограниченно долго, каждую
        итерацию искажая equity — так разово раздутый объём AVAX/USDT
        (учтено 416.5, на бирже 0.00046) превратил "Просадку" в дашборде в
        "-220110,7%".

        Возвращает актуальный свободный баланс USDT ТЕКУЩЕГО тумблера —
        вызывающий код (main.py, расчёт equity) должен использовать именно
        его вместо отдельного get_real_balance(), чтобы не делать лишний
        запрос к бирже за одну итерацию. Агрегация баланса по ОБОИМ рынкам
        сразу — намеренно вне рамок (то же упрощение, что и в Этапе 3).
        """
        if not self.exchange:
            return None
        try:
            balance = await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None

        # Спот и фьючерсы сверяются РАЗНЫМИ способами (остаток монеты на
        # кошельке vs размер открытой позиции-контракта) — маршрутизация по
        # market_type КАЖДОЙ позиции, а не по текущему тумблеру: обе могут
        # быть отслеживаемы одновременно (см. _exchange_for/Этап 3).
        # spot_balance резолвится лениво и переиспользуется на все спотовые
        # позиции за один проход — `balance` выше уже spot, если текущий
        # тумблер сам на споте, иначе нужен отдельный запрос к spot-клиенту.
        spot_balance = balance if settings.market_type == "spot" else None
        for symbol, pos in list(self.real_positions.items()):
            if pos.get("market_type", "spot") == "futures":
                await self._reconcile_futures_position(symbol, pos)
                continue
            if spot_balance is None:
                spot_exchange = self._exchanges.get("spot")
                if spot_exchange is None:
                    continue
                try:
                    spot_balance = await spot_exchange.fetch_balance()
                except Exception as e:
                    logger.debug(f"Не удалось получить spot-баланс для сверки позиций: {e}")
                    continue
            await self._reconcile_spot_position(symbol, pos, spot_balance)

        return self._extract_usdt_balance(balance)

    async def _reconcile_spot_position(self, symbol: str, pos: dict, balance: dict) -> None:
        """
        Спотовая часть reconcile_real_positions — сверяет отслеживаемый
        объём с остатком БАЗОВОЙ ВАЛЮТЫ НА КОШЕЛЬКЕ (balance — снимок
        СПОТОВОГО клиента этой позиции, не обязательно текущего тумблера).
        """
        tracked_amount = pos.get("amount") or 0
        if tracked_amount <= 0:
            return
        base_currency = symbol.split("/")[0]
        available = self._extract_currency_balance(balance, base_currency)
        if available >= tracked_amount:
            return
        min_amount = self._market_min_amount(symbol, self._exchanges.get("spot"))
        unsellable = available == 0 or (min_amount is not None and available < min_amount)
        if not unsellable:
            return
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
        # а не появлялось бы даже после рестарта. Проверяем их ДО
        # grace-периода ниже — SL/TP вполне может сработать по-настоящему
        # уже через несколько секунд после открытия, и такое реальное
        # закрытие нужно распознать сразу, а не откладывать.
        if await self._finalize_externally_closed_position(symbol, pos):
            return
        if await self._finalize_via_recent_trade_history(symbol, pos):
            return
        # Свежеоткрытая позиция, для которой не нашлось объяснения через
        # SL-ордер/историю сделок, — биржа иногда не успевает отразить
        # только что купленный актив в ответе fetch_balance() сразу
        # (реальный инцидент, прод: CHZ/MANA/ZRX — ~90с после открытия
        # и подтверждения ордера как исполненного, fetch_balance() всё
        # ещё показывал available≈0; позицию ошибочно списали как
        # фантомную, а стратегия тут же открыла ДУБЛИРУЮЩУЮ новую на тот
        # же символ — реальные деньги от первой позиции повисли на
        # бирже без SL/TP и без отслеживания). Даём бирже время
        # догнать состояние вместо немедленного списания — если
        # расхождение реальное, а не задержка репликации баланса,
        # следующий цикл сверки поймает его снова, когда позиция
        # перестанет быть "свежей".
        opened_at = pos.get("opened_at")
        if opened_at and (utcnow() - opened_at).total_seconds() < RECONCILE_MIN_AGE_SECONDS:
            return
        logger.warning(
            f"⚠️ Периодическая сверка позиций: {symbol} — учтено {tracked_amount:.8f}, "
            f"на бирже доступно {available:.8f} (продать невозможно) — расхождение поймано "
            f"до попытки закрытия, чтобы не портить equity/просадку."
        )
        await self._reconcile_phantom_position(symbol, pos.get("order_id"))

    async def _reconcile_futures_position(self, symbol: str, pos: dict) -> None:
        """
        Фьючерсная часть reconcile_real_positions — прямая параллель
        _reconcile_spot_position, но вместо остатка базовой валюты на
        кошельке сравнивает отслеживаемый объём с фактическим размером
        ПОЗИЦИИ на бирже (fetch_position — контракт, а не монета на
        балансе; позиция не выражается остатком монеты на фьючерсах).

        fetch_position() у ccxt/bybit всегда возвращает dict (даже для
        закрытой позиции — с contracts=0), не бросает исключение и не
        возвращает пустой список — единственный способ отличить "позиции
        нет" — проверить актуальный объём контракта.

        Попутно (без дополнительного запроса к бирже — используем тот же
        ответ) кэширует leverage/margin_usdt на pos для отображения на
        дашборде (Этап 6): settings.futures_leverage — глобальная
        настройка, могла измениться ПОСЛЕ открытия конкретной позиции, так
        что единственный надёжный источник факта — сама биржа.
        """
        tracked_amount = pos.get("amount") or 0
        if tracked_amount <= 0:
            return
        exchange = self._exchange_for(pos)
        if exchange is None:
            return
        # Расширенная диагностика (временно): _ccxt_symbol (см. коммит про
        # суффикс :QUOTE для linear-swap) не решил проблему — 181001
        # "category only support linear or option" продолжает падать даже
        # на позиции, открытой УЖЕ ПОСЛЕ этого фикса. Логируем сам
        # запрашиваемый ccxt_symbol и то, что exchange.market() реально о
        # нём думает (type/linear/inverse/id) — ДО вызова fetch_position,
        # чтобы это попало в лог даже если сам fetch_position упадёт:
        # без этого невозможно отличить "market() резолвит не тот рынок"
        # от "запрос вообще ушёл с другим набором параметров".
        ccxt_symbol = self._ccxt_symbol(exchange, symbol)
        try:
            resolved = exchange.market(ccxt_symbol)
            market_info = (
                f"market={resolved.get('symbol')!r} type={resolved.get('type')!r} "
                f"linear={resolved.get('linear')!r} inverse={resolved.get('inverse')!r} "
                f"id={resolved.get('id')!r}"
            )
        except Exception as market_err:
            market_info = f"exchange.market({ccxt_symbol!r}) упал: {type(market_err).__name__}: {market_err}"
        options_info = exchange.options if isinstance(exchange.options, dict) else repr(exchange.options)
        try:
            position = await exchange.fetch_position(ccxt_symbol)
            actual_amount = float(position.get("contracts") or 0)
        except Exception as e:
            # Поднято с debug до warning намеренно, временно (диагностика):
            # реальный инцидент (прод) — эта ветка стабильно проваливается
            # часами подряд на нескольких символах, из-за чего leverage/
            # margin_usdt никогда не подтягиваются и реконсиляция ни разу
            # не может поймать реально исчезнувшую/осиротевшую позицию
            # (BCH/USDT, HYPE/USDT) — но САМА причина невидима на
            # debug-уровне через /logs дашборда. type(e).__name__ — на
            # случай, если текст исключения сам по себе неинформативен.
            logger.warning(
                f"⚠️ Не удалось сверить фьючерсную позицию {symbol}: {type(e).__name__}: {e} | "
                f"ccxt_symbol={ccxt_symbol!r} | {market_info} | "
                f"exchange.options.defaultType={options_info.get('defaultType') if isinstance(options_info, dict) else options_info!r}"
            )
            return
        pos["leverage"] = position.get("leverage")
        pos["margin_usdt"] = position.get("initialMargin")
        if actual_amount >= tracked_amount:
            return
        min_amount = self._market_min_amount(symbol, exchange)
        unsellable = actual_amount == 0 or (min_amount is not None and actual_amount < min_amount)
        if not unsellable:
            return
        # Та же цепочка объяснений исчезновения, что и на споте (см.
        # _reconcile_spot_position) — оба _finalize_* уже рынко-осознанны
        # (резолвят клиент по market_type позиции, Этап 3), изменений не
        # требуют.
        if await self._finalize_externally_closed_position(symbol, pos):
            return
        if await self._finalize_via_recent_trade_history(symbol, pos):
            return
        opened_at = pos.get("opened_at")
        if opened_at and (utcnow() - opened_at).total_seconds() < RECONCILE_MIN_AGE_SECONDS:
            return
        logger.warning(
            f"⚠️ Периодическая сверка фьючерсных позиций: {symbol} — учтено {tracked_amount:.8f}, "
            f"на бирже открыто {actual_amount:.8f} контрактов — расхождение поймано до попытки "
            f"закрытия, чтобы не портить equity/просадку."
        )
        await self._reconcile_phantom_position(symbol, pos.get("order_id"))

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
        exchange = self._exchange_for(pos)
        if exchange is None:
            return False
        try:
            order = await exchange.fetch_order(sl_order_id, self._ccxt_symbol(exchange, symbol))
        except Exception as e:
            logger.debug(f"Не удалось проверить биржевой SL-ордер {sl_order_id} ({symbol}): {e}")
            return False
        status = str(order.get("status") or "").lower()
        if status not in ("closed", "filled"):
            return False

        amount = float(order.get("filled") or pos.get("amount") or 0)
        if amount <= 0:
            return False

        trade_fill = await self._fetch_fill_details_via_trades(str(sl_order_id), symbol, exchange)
        if trade_fill:
            exit_price = trade_fill["average"]
            amount = trade_fill["amount"]
            exit_fee = trade_fill["fee"].get("cost") or 0
            exit_fee_currency = trade_fill["fee"].get("currency")
        else:
            exit_price = order.get("average") or order.get("price")
            if not exit_price:
                return False
            closing_side = "sell" if pos.get("side", "long") == "long" else "buy"
            exit_fee, exit_fee_currency = self._resolve_fee(order.get("fee"), amount, exit_price, closing_side, symbol)

        await self._record_external_close(
            symbol, pos, exit_price=exit_price, amount=amount, exit_fee=exit_fee,
            exit_fee_currency=exit_fee_currency,
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

        Реальный инцидент (прод, WIF/USDT short на фьючерсах): closing_side
        раньше был жёстко захардкожен как "sell" — верно только для LONG
        (закрытие long = продажа). Закрытие SHORT-позиции на бирже — это
        BUY (откуп), и такая сделка тут никогда не находилась: фильтр
        молча возвращал пустой список, метод отдавал False, и любая
        short-позиция, закрывшаяся вне цикла бота (сработавший SL,
        ликвидация, ручное закрытие на бирже), проваливалась дальше в
        _reconcile_phantom_position — терялась без PnL, хотя реальная
        закрывающая сделка была в истории и её было чем найти.
        """
        tracked_amount = pos.get("amount") or 0
        opened_at = pos.get("opened_at")
        if tracked_amount <= 0 or not opened_at:
            return False
        exchange = self._exchange_for(pos)
        if exchange is None:
            return False
        # Как и другие необязательные сверки с биржей в этом классе (см.
        # _fetch_fill_details_via_trades) — вся функция под одним широким
        # try/except: непредвиденный формат ответа биржи/мока (например,
        # fetch_my_trades не поддерживается или неитерируемый результат)
        # должен просто означать "сверить не удалось", а не ронять весь
        # reconcile_real_positions.
        try:
            recent = await exchange.fetch_my_trades(self._ccxt_symbol(exchange, symbol), limit=20)
            if not recent:
                return False

            # opened_at — наивный datetime в UTC (см. utcnow); .timestamp()
            # на наивном datetime трактует его как ЛОКАЛЬНОЕ время, а не
            # UTC, и даёт неверный epoch вне контейнеров с TZ=UTC — сначала
            # явно проставляем tzinfo=UTC, как и utcnow_timestamp.
            opened_at_ts = opened_at.replace(tzinfo=UTC).timestamp() * 1000
            closing_side = "sell" if pos.get("side", "long") == "long" else "buy"
            closing_trades = [
                t for t in recent
                if str(t.get("side", "")).lower() == closing_side and (t.get("timestamp") or 0) >= opened_at_ts
            ]
            if not closing_trades:
                return False

            total_amount = sum(float(t.get("amount") or 0) for t in closing_trades)
            if total_amount <= 0:
                return False
            if abs(total_amount - tracked_amount) / tracked_amount > 0.15:
                logger.debug(
                    f"Сверка {symbol}: недавние {closing_side} ({total_amount:.8f}) слишком расходятся с "
                    f"отслеживаемым объёмом ({tracked_amount:.8f}) — не гадаем, чей это ордер."
                )
                return False
        except Exception as e:
            logger.debug(f"Не удалось сверить закрытие {symbol} по истории сделок: {e}")
            return False

        total_cost = sum(
            float(t["cost"]) if t.get("cost") is not None else float(t.get("amount") or 0) * float(t.get("price") or 0)
            for t in closing_trades
        )
        exit_price = total_cost / total_amount if total_amount else 0
        if not exit_price:
            return False
        # Несколько сделок закрытия могут быть с РАЗНОЙ валютой комиссии
        # (например, часть — в USDT, часть — в base-валюте, если бирже
        # хватило базового актива не на весь объём) — суммировать "как
        # есть" означало бы складывать разноразмерные числа. Конвертируем
        # base-валютную комиссию каждой сделки в USDT по её же цене
        # исполнения (передаём итоговую валюту как None — валюта после
        # суммирования уже смешанная/нормализованная в quote, а не одна
        # известная валюта конкретной сделки).
        base_currency = symbol.split("/")[0]
        exit_fee = 0.0
        for t in closing_trades:
            fee = t.get("fee") or {}
            cost = fee.get("cost")
            if not cost:
                continue
            cost = float(cost)
            if fee.get("currency") == base_currency:
                fill_price = float(t.get("price") or exit_price)
                cost *= fill_price
            exit_fee += cost
        trade_ids = [str(t["id"]) for t in closing_trades if t.get("id")]

        await self._record_external_close(
            symbol, pos, exit_price=exit_price, amount=total_amount, exit_fee=exit_fee,
            order_id_exchange=",".join(trade_ids) if trade_ids else None,
            log_note=f"🔍 Позиция закрыта вне цикла бота (найдено по истории сделок биржи): {symbol}",
        )
        return True

    async def _record_external_close(
        self, symbol: str, pos: dict, *, exit_price: float, amount: float, exit_fee: float,
        order_id_exchange: str | None, log_note: str,
        exit_fee_currency: str | None = None,
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
        side = pos.get("side", "long")
        order_open_id = pos.get("order_id")

        # Комиссии часто удержаны в BASE-валюте, а не в USDT (см. _resolve_fee
        # — на споте комиссия покупки обычно списывается из полученного
        # актива). Вычитать такую комиссию из PnL как есть означало бы
        # принять, например, "105.4915 TAC" за "105.4915 USDT" — искажение
        # на порядки (тот же класс бага, что уже исправлен в
        # close_real_position — см. её комментарий про инцидент HYPE/USDT).
        # Этот путь (закрытие ВНЕ цикла бота) раньше вообще не делал такую
        # конвертацию — считал entry_fee/exit_fee уже в USDT независимо от
        # реальной валюты списания.
        entry_fee_quote = entry_fee
        exit_fee_quote = exit_fee
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
        if exit_fee_currency == base_currency:
            exit_fee_quote = exit_fee * exit_price

        # Фьючерсная позиция, закрытая ВНЕ цикла бота, могла быть short —
        # раньше здесь всегда считалась long-формула и direction="long"
        # (правильно только для спота, где short не бывает).
        if side == "long":
            pnl = (exit_price - entry_price) * amount - entry_fee_quote - exit_fee_quote
        else:
            pnl = (entry_price - exit_price) * amount - entry_fee_quote - exit_fee_quote
        pnl_pct = (pnl / (entry_price * amount) * 100) if entry_price and amount else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "break-even")
        opened_at = pos.get("opened_at")
        holding_seconds = int((utcnow() - opened_at).total_seconds()) if opened_at else 0

        self.real_positions.pop(symbol, None)
        # Позиция закрылась ЧУЖИМ путём (сработал сам SL/найдено по истории
        # сделок), а не через close_real_position — тот отменяет
        # sl_order_id ДО своей продажи, здесь этого шага никогда не было.
        # Реальный симптом (прод): ASTER/QTUM/TIA годами держали часть
        # баланса заблокированной ("used" в /balances) — осиротевший
        # условный SL-ордер от давно закрытой (этим путём) позиции никогда
        # не отменялся, продолжая резервировать актив на бирже. Та же
        # отмена, что уже сделана для _reconcile_phantom_position.
        await self._cancel_order_safe(symbol, pos.get("sl_order_id"), self._exchange_for(pos))

        async with get_session() as session:
            exchange_id, symbol_id = await self._resolve_symbol_id(session, symbol)
            strategy_db_id = await self._resolve_strategy_id(session, pos.get("strategy_id"))
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
                fee_currency=exit_fee_currency,
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

        risk_manager.on_trade_closed(pnl)
        risk_manager.on_position_closed(symbol)
        logger.warning(f"{log_note} | PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)")

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

    @staticmethod
    def _extract_usdt_balance(balance: dict) -> float:
        return ExecutionEngine._extract_currency_balance(balance, "USDT")

    @staticmethod
    def _extract_currency_balance(balance: dict, currency: str, field: str = "free") -> float:
        """
        ccxt fetch_balance() кладёт баланс валюты в ВЛОЖЕННЫЙ словарь:
        balance[field][currency] (плоское число) и/или balance[currency] =
        {'free':, 'used':, 'total':} (сам по себе словарь, а НЕ число).
        Старый код фильтровал `isinstance(v, (int, float))` по
        balance.items() верхнего уровня — balance[currency] там всегда
        словарь, значит проверка никогда не проходила, и баланс всегда
        читался как 0 независимо от биржи и реального остатка на счёте.

        field — "free" (по умолчанию, как и раньше), "used" или "total" —
        тот же вложенный формат под любым из этих трёх ключей.
        """
        top = balance.get(field) or {}
        value = top.get(currency)
        if value is None:
            entry = balance.get(currency)
            if isinstance(entry, dict):
                value = entry.get(field)
        return float(value or 0.0)

    async def get_all_balances(self) -> list[dict] | None:
        """
        Все ненулевые балансы аккаунта на бирже — все валюты, а не только
        котируемая (USDT), которую показывает get_real_balance(). Нужно для
        дашборда: бот может держать актив по любой открытой позиции (и
        пыль, оставшуюся после _reconcile_phantom_position), а пользователь
        иначе видел только один агрегированный USDT-баланс и не мог свериться
        с тем, что реально лежит на счету биржи.
        """
        if not self.exchange:
            return None
        try:
            balance = await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Ошибка получения балансов: {e}")
            return None
        # Валюты берём и из плоских словарей free/used/total, и из формата,
        # где balance[currency] сам по себе словарь {'free':,'used':,'total':}
        # (см. _extract_currency_balance) — иначе биржи/сборки ccxt, которые
        # отдают ТОЛЬКО второй формат (без верхнеуровневых free/used/total),
        # всегда возвращали бы пустой список балансов.
        reserved_keys = {"info", "timestamp", "datetime", "free", "used", "total"}
        currencies = set(balance.get("free") or {}) | set(balance.get("used") or {}) | set(balance.get("total") or {})
        currencies |= {k for k, v in balance.items() if k not in reserved_keys and isinstance(v, dict)}
        result = []
        for currency in currencies:
            total = self._extract_currency_balance(balance, currency, "total")
            if total <= 0:
                continue
            result.append({
                "currency": currency,
                "free": self._extract_currency_balance(balance, currency, "free"),
                "used": self._extract_currency_balance(balance, currency, "used"),
                "total": total,
            })

        async def _fill_usdt_value(item: dict) -> None:
            if item["currency"] == "USDT":
                item["usdt_value"] = item["total"]
                return
            try:
                pair = f"{item['currency']}/USDT"
                ticker = await self.exchange.fetch_ticker(self._ccxt_symbol(self.exchange, pair))
                price = ticker.get("last") or ticker.get("bid") or ticker.get("ask")
                item["usdt_value"] = item["total"] * price if price else None
            except Exception:
                # Валюта без прямой пары к USDT на бирже (или временная ошибка
                # тикера) — не критично для списка балансов в целом, просто
                # эта строка показывает эквивалент как "—" на дашборде.
                item["usdt_value"] = None

        await asyncio.gather(*(_fill_usdt_value(item) for item in result))
        result.sort(key=lambda b: b["currency"])
        return result

    async def sweep_balances_to_usdt(self) -> dict:
        """
        Продать по рынку ВСЕ ненулевые остатки (кроме самого USDT) обратно
        в USDT — ручная разовая операция через дашборд (кнопка "Продать
        все остатки в USDT" рядом с /balances), а не автоматическое
        действие бота. Всегда через СПОТОВЫЙ клиент независимо от текущего
        тумблера market_type — это обычные держания базовой валюты на
        аккаунте (в т.ч. пыль, оставшаяся после _reconcile_phantom_position/
        _record_external_close), а не фьючерсные контракты.

        Перед продажей КАЖДОЙ валюты, у которой есть заблокированный
        ("used") остаток, сначала отменяет любые открытые ордера по её
        паре к USDT (обычные и условные/tpsl — тип заранее не известен) —
        иначе такой остаток не попадёт в fetch_balance()["free"] и его
        нельзя будет продать. Реальный симптом (прод): у ASTER/QTUM/TIA
        часть баланса годами оставалась в used — осиротевшие условные
        SL-ордера, ни разу не отменённые (см. фикс в _record_external_close).

        Best-effort по каждой валюте отдельно — ошибка по одной не должна
        прерывать обработку остальных. Возвращает
        {"sold": [...], "skipped": [...], "errors": [...]}.
        """
        exchange = await self._ensure_exchange_connected("spot")
        if exchange is None:
            return {
                "sold": [], "skipped": [],
                "errors": [{"currency": None, "reason": "нет подключения к спотовому рынку"}],
            }

        try:
            balance = await exchange.fetch_balance()
        except Exception as e:
            return {
                "sold": [], "skipped": [],
                "errors": [{"currency": None, "reason": f"не удалось получить баланс: {e}"}],
            }

        reserved_keys = {"info", "timestamp", "datetime", "free", "used", "total"}
        currencies = set(balance.get("free") or {}) | set(balance.get("used") or {}) | set(balance.get("total") or {})
        currencies |= {k for k, v in balance.items() if k not in reserved_keys and isinstance(v, dict)}

        sold: list[dict] = []
        skipped: list[dict] = []
        errors: list[dict] = []

        for currency in sorted(currencies):
            if currency == "USDT":
                continue
            total = self._extract_currency_balance(balance, currency, "total")
            if total <= 0:
                continue
            symbol = f"{currency}/USDT"

            if symbol not in (exchange.markets or {}):
                skipped.append({"currency": currency, "reason": f"нет пары {symbol} на бирже"})
                continue

            used = self._extract_currency_balance(balance, currency, "used")
            if used > 0:
                for order_filter in (None, "tpslOrder", "StopOrder"):
                    try:
                        params = {"orderFilter": order_filter} if order_filter else {}
                        open_orders = await exchange.fetch_open_orders(symbol, params=params)
                        for o in (open_orders or []):
                            await self._cancel_order_safe(symbol, o.get("id"), exchange)
                    except Exception as e:
                        logger.debug(
                            f"Не удалось получить/отменить открытые ордера {symbol} ({order_filter}): {e}"
                        )

            try:
                fresh_balance = await exchange.fetch_balance()
                free = self._extract_currency_balance(fresh_balance, currency, "free")
            except Exception as e:
                errors.append({"currency": currency, "reason": f"не удалось перепроверить баланс: {e}"})
                continue

            if free <= 0:
                skipped.append({
                    "currency": currency,
                    "reason": "нет доступного остатка после отмены ордеров" if used > 0 else "нет свободного остатка",
                })
                continue

            try:
                ticker = await exchange.fetch_ticker(symbol)
                price = ticker.get("last") or ticker.get("bid") or ticker.get("ask")
            except Exception as e:
                errors.append({"currency": currency, "reason": f"не удалось получить цену {symbol}: {e}"})
                continue

            below_min = self._below_exchange_minimum(symbol, free, price, exchange)
            if below_min:
                skipped.append({"currency": currency, "reason": f"пыль — {below_min}"})
                continue

            try:
                order = await exchange.create_market_sell_order(symbol, free)
            except Exception as e:
                errors.append({"currency": currency, "reason": str(e)})
                continue

            sold.append({
                "currency": currency, "amount": free,
                "order_id": order.get("id") if isinstance(order, dict) else None,
            })
            logger.warning(f"💱 Продано {free:.8f} {currency} -> USDT (ручная конвертация остатков через дашборд)")

        if sold:
            # Продажа сразу многих валют резко меняет реальный баланс USDT,
            # никак не отражая торговый результат — это одноразовая
            # консолидация РАНЕЕ НЕотслеживаемых остатков (в т.ч. пыль
            # после _reconcile_phantom_position/_record_external_close), а
            # не прибыль/убыток от сделок. Без пересчёта базы
            # risk_manager.state.start_balance остаётся устаревшим (маленьким)
            # числом с последнего рестарта, и просадка считается против
            # резко выросшего текущего баланса — реальный симптом (прод):
            # total_drawdown_pct показал -1943% ("прибыль" по текущему
            # знаку, но по факту защита от РЕАЛЬНОЙ будущей просадки
            # (max_drawdown_pct) в таком состоянии становится
            # нечувствительной — даже 90%-й убыток от НОВОГО баланса ещё
            # долго не приблизит просадку к порогу, посчитанному от старой
            # заниженной базы). Пересчитываем базу так же, как это делает
            # initialize() при подключении к реальному аккаунту (см.
            # reset_for_real_account).
            try:
                final_balance = await exchange.fetch_balance()
                total_usdt = self._extract_currency_balance(final_balance, "USDT", "total")
                positions_value = sum(
                    pos["amount"] * pos["entry_price"]
                    for pos in self.real_positions.values()
                    if pos.get("side") == "long"
                )
                risk_manager.reset_for_real_account(total_usdt + positions_value)
            except Exception as e:
                logger.warning(f"Не удалось пересчитать базу просадки после конвертации остатков: {e}")

        return {"sold": sold, "skipped": skipped, "errors": errors}

    async def _fetch_confirmed_order(
        self, order: dict, symbol: str, exchange: ccxt.Exchange | None = None,
        attempts: int = 8, delay: float = 0.75,
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
        ex = exchange if exchange is not None else self.exchange
        ccxt_symbol = self._ccxt_symbol(ex, symbol)
        latest = order
        for _ in range(attempts):
            await asyncio.sleep(delay)
            try:
                fetched = await ex.fetch_order(order_id, ccxt_symbol)
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
