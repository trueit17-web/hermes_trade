"""
Симуляция исхода исторического сигнала канала по свечам биржи — второй
этап плана (после history_backfill.py): даёт РЕАЛЬНУЮ метку win/loss/
break-even для будущей ML-модели качества сигнала, а не только сырой
текст сигнала.

Сравнивает ТОЛЬКО close каждой часовой свечи — так же, как и живой бот
(_check_position_exit в main.py получает current_price из последней уже
ЗАКРЫТОЙ 1h-свечи, а не из high/low внутри неё) — поэтому не нужно решать
неоднозначность "SL и TP задеты в одной и той же свече": в любой момент
сравнивается ровно одна цена (close), а не диапазон.

Использует ту же логику частичных TP (1/N исходного объёма на каждом
уровне, кроме последнего закрывающего остаток) и ступенчатого SL (TP1ой ->
безубыток, TPn (n>=2) -> уровень TP(n-1)), что и _check_position_exit —
продублирована здесь как чистая, тестируемая без сети функция
(simulate_signal_against_candles), а не импортирована из main.py, чтобы не
тянуть в этот модуль весь TradingBot и его зависимости.
"""
import logging
from datetime import UTC

import ccxt.async_support as ccxt
from sqlalchemy import select

from src.config import settings
from src.db.models import HistoricalSignal
from src.db.session import get_session
from src.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _tp_levels(entry_price: float, tp: float | None, take_profits: list[float] | None) -> list[float]:
    """
    Копия ветки strategy_id=="telegram_signal" из TradingBot._tp_levels
    (main.py) — см. её докстринг там для полного обоснования (реальные
    цели канала используются напрямую и все, без ограничения тремя;
    линейная интерполяция на 3 уровня — только если канал явно указал
    единственный tp). Продублирована, а не импортирована — main.py тянет
    слишком много посторонних зависимостей для этого модуля.
    """
    if take_profits:
        return list(take_profits)
    if not tp:
        return []
    tp1 = entry_price + (tp - entry_price) / 3
    tp2 = entry_price + (tp - entry_price) * 2 / 3
    return [tp1, tp2, tp]


def simulate_signal_against_candles(
    side: str, entry: float, sl: float | None, tp: float | None,
    take_profits: list[float] | None, closes: list[float],
) -> dict:
    """
    closes — цены закрытия часовых свечей ПОСЛЕ времени сигнала, в
    хронологическом порядке (см. докстринг модуля про "почему только close").

    Возвращает {"outcome": "win"/"loss"/"break-even"/"unresolved",
    "pnl_pct": float | None, "exit_reason": str | None, "tp_hit_count": int}.

    pnl_pct — взвешенная по объёму каждого уровня доходность в процентах
    (без комиссий — исторических данных по ним нет), тот же принцип, что у
    _check_position_exit: 1/N исходного объёма на каждый частичный уровень,
    остаток — на финальном закрытии (последний TP или SL).

    Если свечи закончились, а позиция ещё не закрыта полностью, но хотя бы
    один TP уже сработал — итог помечается "win": ступенчатый SL держит
    остаток минимум в безубытке, поэтому общий результат уже не может стать
    отрицательным, даже если финальная цель ещё не достигнута в пределах
    загруженных свечей. Если ни один уровень не сработал вообще —
    "unresolved" (недостаточно данных, чтобы судить).
    """
    tp_levels = _tp_levels(entry, tp, take_profits)
    n_levels = len(tp_levels)
    if not sl or n_levels == 0 or not closes:
        return {"outcome": "unresolved", "pnl_pct": None, "exit_reason": None, "tp_hit_count": 0}

    def price_return_pct(price: float) -> float:
        return (price - entry) / entry * 100 if side == "long" else (entry - price) / entry * 100

    def outcome_from_pnl(pnl_pct: float) -> str:
        return "win" if pnl_pct > 0 else ("loss" if pnl_pct < 0 else "break-even")

    current_sl = sl
    tp_hit_count = 0
    realized_pnl_pct = 0.0

    for close in closes:
        sl_triggered = close <= current_sl if side == "long" else close >= current_sl
        if sl_triggered:
            remaining_weight = 1 - tp_hit_count / n_levels
            realized_pnl_pct += remaining_weight * price_return_pct(close)
            return {
                "outcome": outcome_from_pnl(realized_pnl_pct), "pnl_pct": realized_pnl_pct,
                "exit_reason": "stop_loss", "tp_hit_count": tp_hit_count,
            }

        # Цена могла перепрыгнуть сразу несколько уровней между свечами —
        # берём самый дальний ещё не достигнутый (тот же приём, что и в
        # живом _check_position_exit), а не первый по порядку.
        level_hit = None
        for level in range(n_levels - 1, tp_hit_count - 1, -1):
            target = tp_levels[level]
            reached = close >= target if side == "long" else close <= target
            if reached:
                level_hit = level
                break
        if level_hit is None:
            continue

        is_final = level_hit == n_levels - 1
        weight = (1 - tp_hit_count / n_levels) if is_final else (1 / n_levels)
        realized_pnl_pct += weight * price_return_pct(close)
        if is_final:
            return {
                "outcome": outcome_from_pnl(realized_pnl_pct), "pnl_pct": realized_pnl_pct,
                "exit_reason": f"take_profit_{level_hit + 1}", "tp_hit_count": tp_hit_count + 1,
            }
        tp_hit_count += 1
        current_sl = entry if level_hit == 0 else tp_levels[level_hit - 1]

    if tp_hit_count > 0:
        return {
            "outcome": "win", "pnl_pct": realized_pnl_pct,
            "exit_reason": "unresolved_after_partial_tp", "tp_hit_count": tp_hit_count,
        }
    return {"outcome": "unresolved", "pnl_pct": None, "exit_reason": None, "tp_hit_count": 0}


def _ccxt_symbol_for_market(symbol: str, market_type: str) -> str:
    """
    Та же суффиксация unified-символа, что и ExecutionEngine._ccxt_symbol
    (executor.py) — на linear swap ("futures" в терминах этого бота) ccxt
    резолвит пару под символом с суффиксом ":QUOTE" ("BTC/USDT:USDT"), а
    не голым "BTC/USDT" (тот матчится на СПОТОВЫЙ рынок той же пары).
    Продублирована как чистая функция без ccxt.Exchange — здесь клиент
    создаётся заново под конкретный market_type, а не переиспользует
    состояние execution_engine.
    """
    if market_type != "futures":
        return symbol
    quote = symbol.split("/")[-1]
    return f"{symbol}:{quote}"


async def simulate_channel_signal_outcomes(
    db_channel_id: int, market_type: str, limit: int = 50, exchange_id: str | None = None,
) -> dict:
    """
    Симулировать исход до `limit` ещё не симулированных распознанных
    исторических сигналов канала (parse_status="parsed", entry/SL заданы,
    simulated_outcome IS NULL) — тянет часовые свечи биржи (ccxt, публичные
    market-data, ключи не нужны) от времени сигнала вперёд и прогоняет
    simulate_signal_against_candles; пишет результат в те же строки.

    exchange_id по умолчанию — settings.active_exchange (та же биржа, что
    и реальная торговля бота).
    """
    async with get_session() as session:
        rows = (
            await session.execute(
                select(HistoricalSignal)
                .where(
                    HistoricalSignal.channel_id == db_channel_id,
                    HistoricalSignal.parse_status == "parsed",
                    HistoricalSignal.simulated_outcome.is_(None),
                    HistoricalSignal.parsed_entry.is_not(None),
                    HistoricalSignal.parsed_sl.is_not(None),
                    HistoricalSignal.parsed_side.is_not(None),
                    HistoricalSignal.parsed_pair.is_not(None),
                )
                .order_by(HistoricalSignal.message_date.asc())
                .limit(limit)
            )
        ).scalars().all()

    if not rows:
        return {"simulated": 0, "skipped_no_data": 0, "unresolved": 0}

    resolved_exchange_id = (exchange_id or settings.active_exchange).lower()
    exchange_cls = getattr(ccxt, resolved_exchange_id, None)
    if exchange_cls is None:
        return {"error": f"Неизвестная биржа {resolved_exchange_id}"}

    exchange = exchange_cls({
        "enableRateLimit": True,
        "options": {"defaultType": "swap" if market_type == "futures" else "spot"},
    })

    simulated = skipped_no_data = unresolved = 0
    try:
        try:
            await exchange.load_markets()
        except Exception as e:
            return {"error": f"Не удалось загрузить рынки {resolved_exchange_id}: {e}"}

        for row in rows:
            ccxt_symbol = _ccxt_symbol_for_market(row.parsed_pair, market_type)
            try:
                since_ms = int(row.message_date.replace(tzinfo=UTC).timestamp() * 1000)
                ohlcv = await exchange.fetch_ohlcv(ccxt_symbol, timeframe="1h", since=since_ms, limit=1000)
            except Exception as e:
                logger.debug(f"Не удалось получить свечи {ccxt_symbol} для симуляции сигнала {row.id}: {e}")
                ohlcv = None
            if not ohlcv:
                skipped_no_data += 1
                continue

            closes = [float(c[4]) for c in ohlcv]
            result = simulate_signal_against_candles(
                row.parsed_side,
                float(row.parsed_entry),
                float(row.parsed_sl),
                float(row.parsed_tp) if row.parsed_tp else None,
                [float(x) for x in row.parsed_take_profits] if row.parsed_take_profits else None,
                closes,
            )

            async with get_session() as session:
                db_row = await session.get(HistoricalSignal, row.id)
                db_row.simulated_outcome = result["outcome"]
                db_row.simulated_pnl_pct = result["pnl_pct"]
                db_row.simulated_exit_reason = result["exit_reason"]
                db_row.simulated_tp_hit_count = result["tp_hit_count"]
                db_row.simulated_at = utcnow()
                await session.commit()

            if result["outcome"] == "unresolved":
                unresolved += 1
            else:
                simulated += 1
    finally:
        await exchange.close()

    return {"simulated": simulated, "skipped_no_data": skipped_no_data, "unresolved": unresolved}
