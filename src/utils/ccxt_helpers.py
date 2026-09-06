"""Общие ccxt-хелперы для перевода нашего канонического символа "BASE/QUOTE"
в unified-символ, который ожидает сам ccxt — нужны и execution-клиенту
(src/execution/executor.py), и market-data ingest'у
(src/data_ingest/market_data.py), поэтому вынесены сюда, а не продублированы."""
from __future__ import annotations

from typing import Any


def ccxt_symbol(exchange: Any, symbol: str) -> str:
    """
    Unified-символ ДЛЯ ПРЯМЫХ ВЫЗОВОВ К CCXT (create_order/fetch_position/
    cancel_order/fetch_ticker/fetch_ohlcv/...) — НЕ путать с нашим
    собственным каноническим "BASE/QUOTE", которым обозначается символ
    everywhere else (real_positions, БД, main.py, дашборд) — тот менять не
    нужно, только то, что летит В exchange.*().

    Реальный инцидент (прод, месяцами): у ccxt спотовый и linear-swap (то,
    что мы называем "futures"/USDT-perpetual) рынки одной и той же пары —
    это ДВА РАЗНЫХ unified-символа: "BCH/USDT" (спот) и "BCH/USDT:USDT"
    (linear swap, суффикс — расчётная валюта через двоеточие; см.
    parse_market в ccxt/bybit.py — `symbol = symbol + ':' + settle`).
    exchange.market(symbol) matches ПО БУКВАЛЬНОМУ СОВПАДЕНИЮ СТРОКИ в
    self.markets — если "BCH/USDT" уже есть как ключ (а он есть — это
    спотовый рынок), метод возвращает СПОТОВЫЙ рынок ВСЕГДА, что бы ни было
    выставлено в options.defaultType/defaultSubType. Для пар, вообще не
    имеющих спотового листинга (например TAO/USDT на Bybit — только
    linear-swap), голый "BASE/QUOTE" не резолвится ВООБЩЕ ни во что — ccxt
    отвечает "does not have market symbol" даже без коллизии с спотом.

    options.defaultType уже корректно проставлен ПРИ ПОДКЛЮЧЕНИИ клиента
    ("swap" для futures, "spot" для spot) — читаем его отсюда же, а не
    заводим отдельный market_type параметр в каждой сигнатуре: то же самое
    отличие клиента, просто доступное напрямую через сам ccxt-объект.
    """
    if exchange is None or ":" in symbol or "/" not in symbol:
        return symbol
    options = exchange.options
    # exchange.options — обычный dict у реального ccxt.Exchange; защитная
    # проверка типа — не только на случай неожиданной биржи без options, но
    # и на тестовые AsyncMock()-заглушки без спека, у которых
    # exchange.options САМ становится AsyncMock (см. unittest.mock:
    # атрибуты AsyncMock по умолчанию рекурсивно тоже AsyncMock) — без неё
    # .get(...) вернул бы корутину вместо значения.
    if not isinstance(options, dict) or options.get("defaultType") != "swap":
        return symbol
    quote = symbol.split("/")[-1]
    return f"{symbol}:{quote}"
