"""
A-share universe management.

Stock pool : 沪深300 ∪ 中证500 ∪ 科创50  (~800 unique stocks after dedup)
ETF pool   : all exchange-listed ETFs filtered by AUM and avg daily volume
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from .config import ETF_MIN_AUM_CNY_M, ETF_MIN_VOL_CNY_M

logger = logging.getLogger(__name__)

# CSI index codes → human label
_INDEX_MAP = {
    "000300": "HS300",
    "000905": "ZZ500",
    "000688": "KC50",
}


# ── ts_code helpers ────────────────────────────────────────────────────────────

def code_to_ts_code(code: str, exchange: str | None = None) -> str:
    """
    Map a 6-digit A-share code to ts_code format (e.g. '600519.SH').

    Exchange detection order:
      1. If `exchange` arg is provided (from AkShare 交易所 column), use it.
      2. Otherwise infer from code prefix:
         688xxx          → .SH (STAR market, Shanghai)
         6xxxxx, 5xxxxx  → .SH (Shanghai main + SH-listed ETFs)
         everything else → .SZ (Shenzhen)
    """
    code = str(code).strip().zfill(6)

    if exchange is not None:
        exch_upper = str(exchange).upper()
        if "上" in exch_upper or "SH" in exch_upper or "SSE" in exch_upper:
            return f"{code}.SH"
        return f"{code}.SZ"

    # Infer from prefix
    if code.startswith("688") or code.startswith("689"):
        return f"{code}.SH"
    if code.startswith(("6", "5", "11")):
        return f"{code}.SH"
    return f"{code}.SZ"


def ts_code_to_symbol(ts_code: str) -> str:
    """Strip the exchange suffix: '600519.SH' → '600519'."""
    return ts_code.split(".")[0]


# ── Stock universe ─────────────────────────────────────────────────────────────

def fetch_index_constituents(index_sym: str) -> pd.DataFrame:
    """
    Fetch current constituents for a CSI index (e.g. '000300').
    Returns DataFrame with columns: [ts_code, name, market_label]
    """
    import akshare as ak

    try:
        df = ak.index_stock_cons_csindex(symbol=index_sym)
    except Exception as e:
        logger.error("Failed to fetch index %s: %s", index_sym, e)
        return pd.DataFrame(columns=["ts_code", "name", "market_label"])

    # Column names from AkShare: 成分券代码, 成分券名称, 交易所
    code_col  = "成分券代码"
    name_col  = "成分券名称"
    exch_col  = "交易所"

    if code_col not in df.columns:
        logger.error("Unexpected columns from index_stock_cons_csindex: %s", df.columns.tolist())
        return pd.DataFrame(columns=["ts_code", "name", "market_label"])

    exchange = df[exch_col] if exch_col in df.columns else None

    result = pd.DataFrame({
        "ts_code": [
            code_to_ts_code(c, exchange.iloc[i] if exchange is not None else None)
            for i, c in enumerate(df[code_col])
        ],
        "name":         df[name_col].values if name_col in df.columns else "",
        "market_label": _INDEX_MAP.get(index_sym, index_sym),
    })
    return result


def get_stock_universe() -> pd.DataFrame:
    """
    Return deduplicated stock universe (HS300 ∪ ZZ500 ∪ KC50).
    Columns: ts_code, name, markets  (comma-separated index memberships)
    """
    dfs: dict[str, pd.DataFrame] = {}
    for sym, label in _INDEX_MAP.items():
        logger.info("Fetching %s constituents…", label)
        dfs[sym] = fetch_index_constituents(sym)
        time.sleep(0.3)   # polite delay between AkShare calls

    combined = pd.concat(dfs.values(), ignore_index=True)

    # Aggregate: one row per ts_code, list all markets it belongs to
    grouped = (
        combined.groupby("ts_code")
        .agg(name=("name", "first"), markets=("market_label", lambda x: ",".join(sorted(set(x)))))
        .reset_index()
    )
    return grouped.sort_values("ts_code").reset_index(drop=True)


# ── ETF universe ───────────────────────────────────────────────────────────────

def fetch_etf_universe() -> pd.DataFrame:
    """
    Fetch all A-share listed ETFs from East Money spot data.

    AkShare confirmed columns include:
      代码, 名称, 成交额 (daily amount in CNY元), 流通市值 (float market cap ≈ AUM in CNY元)

    Returns DataFrame with columns: [ts_code, name, aum_bn, amount_bn]
      aum_bn    = 流通市值 in CNY亿元 (AUM proxy)
      amount_bn = 成交额 in CNY亿元 (daily trading amount)
    """
    import akshare as ak

    try:
        df = ak.fund_etf_spot_em()
    except Exception as e:
        logger.error("fund_etf_spot_em failed: %s", e)
        return pd.DataFrame(columns=["ts_code", "name", "aum_bn", "amount_bn"])

    code_col = "代码"
    name_col = "名称"

    if code_col not in df.columns:
        logger.error("Unexpected fund_etf_spot_em columns: %s", df.columns.tolist())
        return pd.DataFrame(columns=["ts_code", "name", "aum_bn", "amount_bn"])

    # Amounts are in CNY元 — convert to 亿元 (÷ 1e8)
    aum_col    = "流通市值"
    amount_col = "成交额"

    result = pd.DataFrame({
        "ts_code":   df[code_col].apply(code_to_ts_code),
        "name":      df[name_col] if name_col in df.columns else "",
        "aum_bn":    pd.to_numeric(df.get(aum_col),    errors="coerce") / 1e8,
        "amount_bn": pd.to_numeric(df.get(amount_col), errors="coerce") / 1e8,
    })
    return result.dropna(subset=["ts_code"])


def filter_etf_universe(
    etf_df: pd.DataFrame,
    min_aum_bn: float | None = None,         # ETF_MIN_AUM_CNY_M / 100 to convert 百万→亿
    min_amount_bn: float | None = None,      # ETF_MIN_VOL_CNY_M / 100 to convert 百万→亿
    exclude_leveraged: bool = True,
) -> list[str]:
    """
    Filter ETFs by AUM and daily trading amount.

    min_aum_bn    : minimum AUM in CNY亿元.    Default: 5亿 (ETF_MIN_AUM_CNY_M / 100).
    min_amount_bn : minimum daily amount in CNY亿元. Default: 0.2亿 (2000万).
    """
    if min_aum_bn is None:
        min_aum_bn = ETF_MIN_AUM_CNY_M / 100       # 500百万 → 5亿

    if min_amount_bn is None:
        min_amount_bn = ETF_MIN_VOL_CNY_M / 100     # 20百万  → 0.2亿

    mask = (
        etf_df["aum_bn"].fillna(0) >= min_aum_bn
    ) & (
        etf_df["amount_bn"].fillna(0) >= min_amount_bn
    )
    if exclude_leveraged:
        mask &= ~etf_df["name"].str.contains("杠杆|反向|做空|2倍|3倍", na=False)

    return etf_df[mask]["ts_code"].tolist()
