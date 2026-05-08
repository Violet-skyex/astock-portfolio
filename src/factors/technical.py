"""
Technical / price-based factors.

All functions take a prices/volume/returns DataFrame and return a pd.Series
indexed by ts_code, ready for cross-sectional z-scoring in scorer.py.

A-share nuances:
  - Short-term reversal (1M) is often negative alpha in A-shares — user can
    choose to use 1M momentum as a CONTRARIAN signal (flip sign) via `reverse_1m`.
  - Turnover rate (换手率) is a key liquidity factor unique to A-shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 244


def momentum(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Simple price momentum over `lookback` trading days.
    Returns: price[-1] / price[-lookback] - 1
    """
    if len(prices) < lookback:
        return pd.Series(dtype=float)
    return prices.iloc[-1] / prices.iloc[-lookback] - 1


def momentum_1m(prices: pd.DataFrame, reverse: bool = False) -> pd.Series:
    """1-month (21-day) momentum. Set reverse=True for contrarian use."""
    m = momentum(prices, 21)
    return -m if reverse else m


def momentum_3m(prices: pd.DataFrame) -> pd.Series:
    return momentum(prices, 63)


def momentum_6m(prices: pd.DataFrame) -> pd.Series:
    return momentum(prices, 126)


def momentum_12m(prices: pd.DataFrame) -> pd.Series:
    """12-month momentum, skipping the most recent month (standard Jegadeesh-Titman)."""
    if len(prices) < 252:
        return pd.Series(dtype=float)
    return prices.iloc[-21] / prices.iloc[-252] - 1


def volatility_20d(returns: pd.DataFrame) -> pd.Series:
    """Annualised 20-day realised volatility. Lower is better (sign-flip in scorer)."""
    return returns.tail(20).std() * np.sqrt(TRADING_DAYS_YEAR)


def avg_turnover_20d(turnover: pd.DataFrame) -> pd.Series:
    """
    Average daily turnover rate (换手率) over 20 days.
    Higher turnover = higher liquidity and trading activity.
    """
    return turnover.tail(20).mean()


def volume_ratio(volume: pd.DataFrame, short: int = 20, long: int = 60) -> pd.Series:
    """
    Ratio of short-window to long-window average volume.
    > 1 means accelerating interest (same as existing US ETF flow proxy).
    """
    avg_short = volume.tail(short).mean()
    avg_long  = volume.tail(long).mean()
    return avg_short / avg_long.replace(0, np.nan)
