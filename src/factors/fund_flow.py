"""
Capital flow factors derived from fund_flow data.

north_net_buy : 北向资金净买入 (z-scored, higher = more foreign buying)
large_order_net: 大单净流入 (z-scored, higher = more institutional buying)

These are particularly effective in A-shares due to the strong influence of
northbound capital (沪深港通) and institutional herding behaviour.
"""

from __future__ import annotations

import pandas as pd

from ..scorer import zscore


def north_money_factor(fund_flow_df: pd.DataFrame) -> pd.Series:
    """
    Z-scored northbound net buy amount over rolling window.
    Positive = more foreign buying, which is a bullish signal.
    """
    return zscore(fund_flow_df["north_net_buy"].dropna())


def large_order_factor(fund_flow_df: pd.DataFrame) -> pd.Series:
    """
    Z-scored large-order net inflow.
    Positive = institutional accumulation signal.
    """
    return zscore(fund_flow_df["large_order_net"].dropna())


def fund_flow_composite(fund_flow_df: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted composite of north money + large order signals.
    Returns a Series indexed by ts_code.
    """
    z_north = north_money_factor(fund_flow_df)
    z_large = large_order_factor(fund_flow_df)
    combined = (z_north + z_large) / 2
    return combined.reindex(fund_flow_df.index).fillna(0.0)
