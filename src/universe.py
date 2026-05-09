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
    Fetch all A-share listed ETFs via Tushare (globally accessible).
    Falls back to AkShare fund_etf_spot_em (requires China/HK IP).

    Returns DataFrame with columns: [ts_code, name, aum_bn, amount_bn]
      aum_bn    = amount_bn (no direct AUM in Tushare free tier, use volume as proxy)
      amount_bn = recent daily trading amount in CNY亿元
    """
    from .config import TUSHARE_TOKEN

    if TUSHARE_TOKEN:
        try:
            return _fetch_etf_universe_tushare()
        except Exception as e:
            logger.warning("Tushare ETF universe failed (%s), trying AkShare…", e)

    # AkShare fallback (geo-restricted from US)
    import akshare as ak
    try:
        df = ak.fund_etf_spot_em()
    except Exception as e:
        logger.error("fund_etf_spot_em failed: %s", e)
        return pd.DataFrame(columns=["ts_code", "name", "aum_bn", "amount_bn"])

    if "代码" not in df.columns:
        return pd.DataFrame(columns=["ts_code", "name", "aum_bn", "amount_bn"])

    result = pd.DataFrame({
        "ts_code":   df["代码"].apply(code_to_ts_code),
        "name":      df.get("名称", ""),
        "aum_bn":    pd.to_numeric(df.get("流通市值"), errors="coerce") / 1e8,
        "amount_bn": pd.to_numeric(df.get("成交额"),   errors="coerce") / 1e8,
    })
    return result.dropna(subset=["ts_code"])


def _fetch_etf_universe_tushare() -> pd.DataFrame:
    """
    Build ETF universe via Tushare fund_basic + fund_daily (no IP restriction).

    fund_basic  : list of all exchange-listed active ETFs (name, type)
    fund_daily  : most recent trading day's amount in 千元 → convert to 亿元 (÷1e5)
    aum_bn proxy: same as amount_bn (no free-tier AUM endpoint; volume correlates with AUM)
    """
    import tushare as ts
    from .config import TUSHARE_TOKEN

    pro = ts.pro_api(TUSHARE_TOKEN)

    # All active exchange-listed funds
    basics = pro.fund_basic(market="E", status="L",
                            fields="ts_code,name,fund_type,list_date")
    if basics is None or basics.empty:
        raise RuntimeError("fund_basic returned empty")

    # Keep equity/hybrid ETFs; exclude bond/money-market
    keep_types = {"股票型", "混合型", "指数型", "ETF", ""}
    basics = basics[
        basics["fund_type"].fillna("").apply(lambda x: any(t in x for t in keep_types) or x == "")
    ]

    # Recent daily volume — find a valid recent trading day
    recent_date = None
    for delta in range(10):
        candidate = (pd.Timestamp.today() - pd.Timedelta(days=delta)).strftime("%Y%m%d")
        if pd.Timestamp(candidate).weekday() < 5:
            try:
                vol_df = pro.fund_daily(trade_date=candidate,
                                        fields="ts_code,amount")
                if vol_df is not None and not vol_df.empty:
                    recent_date = candidate
                    break
            except Exception:
                pass
        time.sleep(0.1)

    if recent_date is None or vol_df.empty:
        # No volume data — return basics with zero amount
        basics["amount_bn"] = 0.0
    else:
        # amount in Tushare fund_daily is in 千元; convert to 亿元 (÷1e5)
        vol_df["amount_bn"] = pd.to_numeric(vol_df["amount"], errors="coerce") / 1e5
        basics = basics.merge(vol_df[["ts_code", "amount_bn"]], on="ts_code", how="left")
        basics["amount_bn"] = basics["amount_bn"].fillna(0.0)

    result = pd.DataFrame({
        "ts_code":   basics["ts_code"],
        "name":      basics["name"],
        "aum_bn":    basics["amount_bn"],   # volume as AUM proxy
        "amount_bn": basics["amount_bn"],
    })
    return result.dropna(subset=["ts_code"]).reset_index(drop=True)


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
