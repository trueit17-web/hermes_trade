"""Небольшие расчёты, общие для executor.py и main.py (не зависят от
инфраструктуры — вынесены отдельно, чтобы обе стороны не могли разойтись
в реализации одной и той же формулы, как уже случалось, см. docstring
breakeven_stop_price)."""
import math


def breakeven_stop_price(entry_price: float, side: str, entry_fee_rate: float) -> float:
    """
    Цена SL "в безубыток" после первого частичного TP — entry_price,
    отодвинутый на буфер, покрывающий комиссии полного круга (вход + выход).

    SL ровно на entry_price не учитывает комиссии: если цена всего лишь
    вернётся к отметке входа, сделка всё равно закроется в минус на пару
    комиссий при нулевом фактическом движении цены. Реальный инцидент
    (прод, ONDO/USDT): TP1 сработал через 26с после входа, SL остатка
    переставился ровно на entry_price, откат цены к той же отметке
    зафиксировал -16.49 USDT убытка чисто на комиссиях.

    entry_fee_rate — ФАКТИЧЕСКАЯ ставка комиссии входа для этой позиции
    (entry_fee / (entry_price * amount)), а не общий paper-параметр
    настроек: в real-режиме ставка приходит с биржи и может отличаться от
    paper-оценки. На выходе предполагаем ту же ставку (комиссии
    тейкера/мейкера обычно симметричны на одном аккаунте) — отсюда
    удвоение при расчёте буфера.
    """
    if entry_price <= 0 or entry_fee_rate <= 0:
        return entry_price
    buffer = entry_price * entry_fee_rate * 2
    return entry_price + buffer if side == "long" else entry_price - buffer


def halfway_to_entry_stop_price(current_sl: float, entry_price: float) -> float:
    """
    Цена SL на полпути от ТЕКУЩЕГО значения к цене входа — по явному
    запросу пользователя используется вместо переноса В безубыток после
    первого частичного TP (TP1): полный перенос в безубыток сразу после
    самого первого (обычно ближайшего и наименее значимого) уровня цели
    отдавал сделке слишком мало места для обычного шума цены.

    ВАЖНО: в отличие от breakeven_stop_price выше, эта цена НЕ гарантирует
    неотрицательный итог сделки после TP1 — SL остатка всё ещё на
    "убыточной" стороне от входа, просто ближе к нему, чем был до
    срабатывания TP1. Это осознанный выбор пользователя (больше свободы
    движению цены), а не баг.

    Работает одинаково для обеих сторон сделки: результат — линейная
    интерполяция между current_sl и entry_price, знак направления не важен.
    """
    return current_sl + (entry_price - current_sl) / 2


def stop_distance_fraction(entry: float, stop_loss: float) -> float:
    """Расстояние от входа до SL в долях цены входа (0.05 = 5%)."""
    if not entry:
        return 0.0
    return abs(entry - stop_loss) / entry


def fit_leverage_to_stop(entry: float, stop_loss: float, max_sl_pct_of_margin: float, leverage: float) -> float:
    """
    Наибольшее целое плечо (не выше исходного, не ниже 1), при котором
    убыток на SL не превышает max_sl_pct_of_margin % маржи. Альтернатива
    урезанию SL: SL канала сохраняется, а ликвидация гарантированно
    остаётся дальше него.
    """
    distance = stop_distance_fraction(entry, stop_loss)
    if distance <= 0 or leverage <= 1:
        return max(1.0, float(leverage or 1))
    max_leverage = (max_sl_pct_of_margin / 100) / distance
    return float(max(1, math.floor(min(leverage, max_leverage))))


def risk_based_size_pct(entry: float, stop_loss: float, risk_pct: float, max_size_pct: float) -> tuple[float, bool]:
    """
    Размер позиции (% баланса), при котором срабатывание SL стоит
    risk_pct % баланса: объём = риск / расстояние до SL. Возвращает
    (размер, упёрся_ли_в_потолок max_size_pct).
    """
    distance = stop_distance_fraction(entry, stop_loss)
    if distance <= 0:
        return max_size_pct, True
    size = risk_pct / distance
    if max_size_pct > 0 and size > max_size_pct:
        return max_size_pct, True
    return size, False
