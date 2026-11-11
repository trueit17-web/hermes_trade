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


def reward_risk_ratio(
    entry: float, stop_loss: float, take_profits: list[float], side: str,
    weights: list[float] | None = None,
) -> float | None:
    """
    Соотношение прибыль/риск сигнала: СРЕДНЯЯ (взвешенная долями фиксации)
    дистанция до целей, делённая на дистанцию до стопа. Именно среднюю, а
    не дальнюю цель: бот фиксирует позицию частями на каждом уровне (см.
    _check_position_exit в main.py), так что ожидаемая прибыль при
    достижении ВСЕХ целей — это среднее, а не максимум.

    Цель, уже пройденная ценой входа (канал дал вход по рынку позже своих
    же первых целей), прибыли не даёт — считается за 0, а не за минус.
    None — посчитать нельзя (нет целей, нет стопа или стоп на цене входа).
    weights — доли фиксации канала (см. extract_tp_weights); при несовпадении
    длины со списком целей используется равное распределение, как и при
    фактическом закрытии.
    """
    if not entry or not stop_loss or not take_profits:
        return None
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None
    rewards = [max((tp - entry) if side == "long" else (entry - tp), 0.0) for tp in take_profits]
    if weights and len(weights) == len(rewards) and sum(weights) > 0:
        reward = sum(r * w for r, w in zip(rewards, weights, strict=True)) / sum(weights)
    else:
        reward = sum(rewards) / len(rewards)
    return reward / risk


def signal_filter_reject_reason(
    entry: float, stop_loss: float | None, take_profits: list[float], side: str,
    max_sl_distance_pct: float, min_reward_risk: float,
    weights: list[float] | None = None,
) -> str | None:
    """
    Причина отклонения сигнала фильтром «стоп слишком далеко / прибыль не
    окупает риск» — None, если сигнал проходит (или оба порога выключены, 0).

    Мотив (прод, канал «Сигналы Фьючерсы (AI)», 2026-10/11): часть сигналов
    идёт со стопом 40–70% от входа и единственной целью около +10% — семь
    таких позиций висели от 5 до 26 дней с суммарным убытком при
    срабатывании стопов втрое больше всей реализованной прибыли счёта.
    Оба порога по умолчанию выключены: на той же истории закрытые сделки с
    далёким стопом были и самыми прибыльными, поэтому включать и подбирать
    значения — осознанное решение пользователя, а не дефолт бота.

    Без стопа фильтр не применяется (считать нечего); без целей — только
    проверка дистанции до стопа.
    """
    if not entry or not stop_loss:
        return None
    if max_sl_distance_pct and max_sl_distance_pct > 0:
        distance_pct = stop_distance_fraction(entry, stop_loss) * 100
        if distance_pct > max_sl_distance_pct:
            return f"стоп слишком далеко: {distance_pct:.1f}% от входа (лимит {max_sl_distance_pct:g}%)"
    if min_reward_risk and min_reward_risk > 0:
        ratio = reward_risk_ratio(entry, stop_loss, take_profits, side, weights)
        if ratio is not None and ratio < min_reward_risk:
            return f"прибыль не окупает риск: соотношение {ratio:.2f} (минимум {min_reward_risk:g})"
    return None


def holding_limit_exceeded(opened_at, now, max_holding_hours: float) -> bool:
    """
    Позиция открыта дольше лимита (см. position_max_holding_hours в
    config.py). False, если лимит выключен (0) или время открытия неизвестно.
    """
    if not max_holding_hours or max_holding_hours <= 0 or opened_at is None:
        return False
    return (now - opened_at).total_seconds() > max_holding_hours * 3600
