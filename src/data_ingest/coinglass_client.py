"""CoinGlass API клиент — получение аналитических данных для ML и стратегий."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from src.config import settings
from src.utils.logging import logger

logger = logging.getLogger(__name__)


class CoinGlassClient:
    """
    Клиент для CoinGlass API v4.
    Базовый URL: https://open-api-v4.coinglass.com

    Возвращает данные по:
    - Open Interest (OI)
    - Funding Rate
    - Long/Short Ratio
    - Liquidations
    - Order Book (large limit orders)
    - Taker Buy/Sell
    - Fear & Greed Index
    - ETF flows
    - Индикаторы (RSI, MACD, BB, MA, EMA)
    """

    BASE_URL = "https://open-api-v4.coinglass.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.coinglass_api_key
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
            follow_redirects=True,
        )
        self._cache: dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 час

    async def close(self):
        """Закрыть HTTP клиент."""
        await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        """Заголовки для запроса."""
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-coinglass-api-key"] = self.api_key
        return headers

    async def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Выполнить GET запрос с кэшированием."""
        cache_key = f"{endpoint}:{str(params)}"
        cached = self._cache.get(cache_key)
        if cached:
            cached_time = cached.get("_cached_at", 0)
            if datetime.now().timestamp() - cached_time < self._cache_ttl:
                return cached.get("data")

        try:
            response = await self.client.get(
                endpoint,
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            self._cache[cache_key] = {
                "data": data,
                "_cached_at": datetime.now().timestamp(),
            }
            return data
        except httpx.HTTPStatusError as e:
            logger.warning(f"CoinGlass API error [{endpoint}]: {e.response.status_code} {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"CoinGlass request failed [{endpoint}]: {e}")
            return None

    # === Futures Market ===

    async def get_supported_coins(self, exchange: Optional[str] = None) -> Optional[dict]:
        """Получить список поддерживаемых монет для фьючерсов."""
        return await self._get("/api/futures/supported-coins", {"exchange": exchange} if exchange else None)

    async def get_coins_markets(self, symbol: Optional[str] = None) -> Optional[dict]:
        """Рыночные данные по монетам."""
        return await self._get("/api/futures/coins-markets", {"symbol": symbol} if symbol else None)

    async def get_price_history(
        self,
        symbol: str,
        timeframe: str = "30m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[dict]:
        """История цен (OHLC)."""
        params = {
            "symbol": symbol,
            "type": timeframe,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._get("/api/futures/price/history", params)

    # === Open Interest ===

    async def get_open_interest_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> Optional[dict]:
        """История Open Interest (OHLC)."""
        params = {
            "symbol": symbol,
            "type": timeframe,
            "limit": limit,
        }
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/open-interest/history", params)

    async def get_aggregated_oi_history(self, limit: int = 100) -> Optional[dict]:
        """Агрегированная история OI по всем рынкам."""
        return await self._get("/api/futures/open-interest/aggregated-history", {"limit": limit})

    # === Funding Rate ===

    async def get_funding_rate_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """История фандинг-рейта."""
        params = {"symbol": symbol, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/funding-rate/history", params)

    async def get_funding_rate_arbitrage(self, symbol: Optional[str] = None) -> Optional[dict]:
        """Возможности арбитража фандинг-рейта."""
        return await self._get("/api/futures/funding-rate/arbitrage", {"symbol": symbol} if symbol else None)

    # === Long/Short Ratio ===

    async def get_global_long_short_ratio(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """Глобальное соотношение Long/Short (по аккаунтам)."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/global-long-short-account-ratio/history", params)

    async def get_top_long_short_position_ratio(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """Соотношение позиций top-трейдеров (Long/Short)."""
        params = {"symbol": symbol, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/top-long-short-position-ratio/history", params)

    async def get_net_position_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """Сетевая позиция (net position) по символу."""
        params = {"symbol": symbol, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/net-position/history", params)

    # === Liquidations ===

    async def get_liquidation_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """История ликвидаций по символу."""
        params = {"symbol": symbol, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/liquidation/history", params)

    async def get_liquidation_aggregated_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """Агрегированная история ликвидаций по монете."""
        params: dict = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/api/futures/liquidation/aggregated-history", params)

    async def get_liquidation_heatmap(
        self,
        symbol: str,
        model: int = 1,
    ) -> Optional[dict]:
        """Тепловая карта ликвидаций (model1, model2, model3)."""
        return await self._get(
            f"/api/futures/liquidation/heatmap/model{model}",
            {"symbol": symbol},
        )

    # === Order Book (Large Limit Orders) ===

    async def get_large_limit_orders(
        self,
        symbol: str,
        exchange: Optional[str] = None,
    ) -> Optional[dict]:
        """Крупные лимитные ордера (whale orders)."""
        params: dict = {"symbol": symbol}
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/orderbook/large-limit-order", params)

    async def get_orderbook_ask_bids_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> Optional[dict]:
        """История асков и байдов (с указанием диапазона)."""
        params = {
            "symbol": symbol,
            "type": timeframe,
            "limit": limit,
        }
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/orderbook/ask-bids-history", params)

    # === Taker Buy/Sell ===

    async def get_taker_buy_sell_volume_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> Optional[dict]:
        """История объёма taker buy/sell по символу."""
        params = {
            "symbol": symbol,
            "type": timeframe,
            "limit": limit,
        }
        if exchange:
            params["exchange"] = exchange
        return await self._get("/api/futures/v2/taker-buy-sell-volume/history", params)

    async def get_aggregated_cvd_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[dict]:
        """Агрегированная история CVD (Cumulative Volume Delta)."""
        params: dict = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/api/futures/aggregated-cvd/history", params)

    # === Индикаторы (уже рассчитанные) ===

    async def get_rsi(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[dict]:
        """RSI по символу."""
        return await self._get(
            "/api/futures/indicators/rsi",
            {"symbol": symbol, "type": timeframe, "limit": limit},
        )

    async def get_macd(
        self,
        symbol: str,
        timeframe: str = "1h",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        limit: int = 100,
    ) -> Optional[dict]:
        """MACD по символу."""
        return await self._get(
            "/api/futures/indicators/macd",
            {
                "symbol": symbol,
                "type": timeframe,
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "limit": limit,
            },
        )

    async def get_boll(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[dict]:
        """Bollinger Bands по символу."""
        return await self._get(
            "/api/futures/indicators/boll",
            {"symbol": symbol, "type": timeframe, "limit": limit},
        )

    async def get_ma(self, symbol: str, timeframe: str = "1h", ma_type: str = " SMA", limit: int = 100) -> Optional[dict]:
        """Moving Average по символу."""
        return await self._get(
            "/api/futures/indicators/ma",
            {"symbol": symbol, "type": timeframe, "maType": ma_type, "limit": limit},
        )

    async def get_ema(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[dict]:
        """Exponential Moving Average по символу."""
        return await self._get(
            "/api/futures/indicators/ema",
            {"symbol": symbol, "type": timeframe, "limit": limit},
        )

    async def get_atr(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[dict]:
        """Average True Range по символу."""
        return await self._get(
            "/api/futures/indicators/avg-true-range",
            {"symbol": symbol, "type": timeframe, "limit": limit},
        )

    # === Индексы и макро ===

    async def get_fear_greed_history(self, limit: int = 100) -> Optional[dict]:
        """История Crypto Fear & Greed Index."""
        return await self._get("/api/index/fear-greed-history", {"limit": limit})

    async def get_btc_dominance(self, limit: int = 100) -> Optional[dict]:
        """Bitcoin dominance."""
        return await self._get("/api/index/bitcoin-dominance", {"limit": limit})

    async def get_etf_flow_history(
        self,
        asset: str = "btc",
        limit: int = 100,
    ) -> Optional[dict]:
        """История ETF потоков (btc, eth, sol, xrp)."""
        return await self._get(f"/api/etf/{asset}/flow-history", {"limit": limit})

    async def get_stablecoin_marketcap_history(self, limit: int = 100) -> Optional[dict]:
        """История рыночной капитализации стейблкоинов."""
        return await self._get(
            "/api/index/stableCoin-marketCap-history",
            {"limit": limit},
        )

    async def get_futures_spot_volume_ratio(self, limit: int = 100) -> Optional[dict]:
        """Отношение фьючерсного объёма к спотовому."""
        return await self._get("/api/futures_spot_volume_ratio", {"limit": limit})


# Глобальный экземпляр (создаётся лениво)
_cg_client: Optional[CoinGlassClient] = None


def get_coinglass_client() -> CoinGlassClient:
    """Получить глобальный экземпляр CoinGlass клиента."""
    global _cg_client
    if _cg_client is None:
        _cg_client = CoinGlassClient()
    return _cg_client
