"""
Global constants and environment-based configuration.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) first, then os.environ / .env."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


# ── API credentials ────────────────────────────────────────────────────────────
TUSHARE_TOKEN: str = _get_secret("TUSHARE_TOKEN")
BAIDU_API_KEY: str = _get_secret("BAIDU_API_KEY")
BAIDU_SECRET_KEY: str = _get_secret("BAIDU_SECRET_KEY")

# ── Market constants ───────────────────────────────────────────────────────────
TRADING_DAYS_YEAR = 244          # A-share average trading days per year
RISK_FREE_RATE    = 0.02         # CNY 7-day repo rate proxy (annualised)
TRANSACTION_COST  = 0.001        # 0.1% one-way (stamp duty + commission)

# ── Universe filter defaults ───────────────────────────────────────────────────
ETF_MIN_AUM_CNY_M    = 500       # AUM floor in million CNY (5亿)
ETF_MIN_VOL_CNY_M    = 20        # 20-day avg daily volume floor in million CNY (2000万)

# ── Factor defaults ────────────────────────────────────────────────────────────
DEFAULT_FACTOR_WEIGHTS = {
    "momentum":    0.20,
    "value":       0.20,
    "quality":     0.15,
    "growth":      0.15,
    "fund_flow":   0.15,
    "liquidity":   0.15,
}

# ── Portfolio defaults ─────────────────────────────────────────────────────────
DEFAULT_MAX_STOCKS    = 8
DEFAULT_MAX_ETFS      = 4
DEFAULT_MIN_WEIGHT    = 0.05     # 5% floor — positions below this become cash
DEFAULT_MAX_WEIGHT    = 0.30     # 30% cap per position

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARK_INDICES = {
    "沪深300": "000300.SH",
    "中证500":  "000905.SH",
    "上证50":   "000016.SH",
}

# ── Index constituent codes ────────────────────────────────────────────────────
HS300_INDEX  = "000300.SH"
ZZ500_INDEX  = "000905.SH"
KC50_INDEX   = "000688.SH"   # 科创50
