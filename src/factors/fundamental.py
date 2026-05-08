"""
Fundamental factors derived from financial statement data.

Value factors  : low PE/PB/PS = good (sign-flip applied in scorer)
Quality factors: high ROE/gross margin = good, low debt ratio = good
Growth factors : high revenue/profit growth = good

All functions take the fundamental table (from data/fundamentals.py) and return
a pd.Series indexed by ts_code.
"""

from __future__ import annotations

import pandas as pd


# ── Value ──────────────────────────────────────────────────────────────────────

def pe_factor(df: pd.DataFrame) -> pd.Series:
    """PE TTM. Lower is better — scorer will invert sign."""
    return df["pe_ttm"]


def pb_factor(df: pd.DataFrame) -> pd.Series:
    """P/B ratio. Lower is better — scorer will invert sign."""
    return df["pb"]


def ps_factor(df: pd.DataFrame) -> pd.Series:
    """P/S TTM. Lower is better — scorer will invert sign."""
    return df["ps_ttm"]


# ── Quality ────────────────────────────────────────────────────────────────────

def roe_factor(df: pd.DataFrame) -> pd.Series:
    """ROE TTM. Higher is better."""
    return df["roe_ttm"]


def gross_margin_factor(df: pd.DataFrame) -> pd.Series:
    """Gross margin. Higher is better."""
    return df["gross_margin"]


def debt_ratio_factor(df: pd.DataFrame) -> pd.Series:
    """Asset-liability ratio. Lower is better — scorer will invert sign."""
    return df["debt_ratio"]


# ── Growth ─────────────────────────────────────────────────────────────────────

def revenue_growth_factor(df: pd.DataFrame) -> pd.Series:
    """YoY revenue growth rate. Higher is better."""
    return df["revenue_yoy"]


def profit_growth_factor(df: pd.DataFrame) -> pd.Series:
    """YoY net profit growth rate. Higher is better."""
    return df["profit_yoy"]


# ── Composite sub-scores ───────────────────────────────────────────────────────

def value_composite(df: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted combination of PE, PB, PS z-scores (all inverted).
    Returns a single value score — higher is cheaper.
    """
    from .utils import zscore
    z_pe = -zscore(pe_factor(df).dropna())
    z_pb = -zscore(pb_factor(df).dropna())
    z_ps = -zscore(ps_factor(df).dropna())
    return (z_pe + z_pb + z_ps).reindex(df.index) / 3


def quality_composite(df: pd.DataFrame) -> pd.Series:
    """Equal-weighted ROE + gross margin - debt ratio."""
    from .utils import zscore
    z_roe    =  zscore(roe_factor(df).dropna())
    z_margin =  zscore(gross_margin_factor(df).dropna())
    z_debt   = -zscore(debt_ratio_factor(df).dropna())
    return (z_roe + z_margin + z_debt).reindex(df.index) / 3


def growth_composite(df: pd.DataFrame) -> pd.Series:
    """Equal-weighted revenue + profit growth."""
    from .utils import zscore
    z_rev    = zscore(revenue_growth_factor(df).dropna())
    z_profit = zscore(profit_growth_factor(df).dropna())
    return (z_rev + z_profit).reindex(df.index) / 2
