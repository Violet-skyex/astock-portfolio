"""
Fundamental financial data for A-share stocks.

Two-layer strategy:
  Primary   — Tushare Pro `daily_basic` + `fina_indicator`
              One API call returns all stocks' metrics at once (fast).
  Fallback  — AkShare per-stock calls (slow, ~0.5s each, use for small subsets).

Tushare required fields:
  daily_basic  : ts_code, trade_date, pe_ttm, pb, ps_ttm, total_mv
  fina_indicator: ts_code, end_date, roe, grossprofit_margin, debt_to_assets,
                  revenue_yoy, netprofit_yoy

AkShare confirmed column names:
  stock_a_indicator_lg → pe, pe_ttm, pb, ps, total_mv
  stock_financial_analysis_indicator → 净资产收益率(%), 销售毛利率(%), 资产负债率(%)
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import TUSHARE_TOKEN

logger = logging.getLogger(__name__)


# ── Tushare bulk fetch (preferred) ────────────────────────────────────────────

def _get_tushare_pro():
    """Return a Tushare Pro API instance, or None if token is missing."""
    if not TUSHARE_TOKEN:
        return None
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        return ts.pro_api()
    except Exception as e:
        logger.warning("Tushare init failed: %s", e)
        return None


def fetch_valuation_tushare(trade_date: str | None = None) -> pd.DataFrame:
    """
    Fetch all stocks' PE/PB/PS/市值 for a given trade date via Tushare.
    trade_date: 'YYYYMMDD', defaults to the most recent trading day.
    Returns DataFrame indexed by ts_code with columns: [pe_ttm, pb, ps_ttm, total_mv_m]
    total_mv_m = total market cap in CNY万元.
    """
    pro = _get_tushare_pro()
    if pro is None:
        logger.warning("Tushare token not set — skipping bulk valuation fetch.")
        return pd.DataFrame()

    fields = "ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv"

    # Resolve trade_date: if not provided, try recent weekdays until we get data
    if not trade_date:
        today = pd.Timestamp.today()
        for delta in range(7):
            candidate = (today - pd.Timedelta(days=delta)).strftime("%Y%m%d")
            if pd.Timestamp(candidate).weekday() < 5:   # Mon–Fri
                trade_date = candidate
                break

    try:
        df = pro.daily_basic(trade_date=trade_date, fields=fields)
    except Exception as e:
        logger.error("Tushare daily_basic failed: %s", e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Keep only the single latest date (guard against multi-date responses)
    if "trade_date" in df.columns:
        df = df[df["trade_date"] == df["trade_date"].max()]

    df = df.rename(columns={"total_mv": "total_mv_m"})
    df = df[["ts_code", "pe_ttm", "pb", "ps_ttm", "total_mv_m"]].copy()
    for col in ["pe_ttm", "pb", "ps_ttm", "total_mv_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["pe_ttm"] < 0, "pe_ttm"] = float("nan")

    result = df.set_index("ts_code")
    return result[~result.index.duplicated(keep="last")]


def fetch_financial_indicators_tushare(
    ts_codes: list[str],
    period: str | None = None,
) -> pd.DataFrame:
    """
    Fetch ROE, gross margin, debt ratio, revenue/profit YoY growth via Tushare.
    period: end date of financial report, 'YYYYMMDD' (e.g. '20231231').
            Defaults to the most recently available period.

    Returns DataFrame indexed by ts_code.
    """
    pro = _get_tushare_pro()
    if pro is None:
        return pd.DataFrame()

    fields = "ts_code,end_date,roe,grossprofit_margin,debt_to_assets,revenue_yoy,netprofit_yoy"
    rows = []
    # Tushare fina_indicator supports batching via ts_code list in some versions;
    # fall back to per-stock calls if needed.
    for code in ts_codes:
        try:
            df = pro.fina_indicator(
                ts_code=code,
                period=period or "",
                fields=fields,
            )
            if df is not None and not df.empty:
                latest = df.sort_values("end_date", ascending=False).iloc[0]
                rows.append({
                    "ts_code":          code,
                    "roe_ttm":          pd.to_numeric(latest.get("roe"), errors="coerce"),
                    "gross_margin":     pd.to_numeric(latest.get("grossprofit_margin"), errors="coerce"),
                    "debt_ratio":       pd.to_numeric(latest.get("debt_to_assets"), errors="coerce"),
                    "revenue_yoy":      pd.to_numeric(latest.get("revenue_yoy"), errors="coerce"),
                    "profit_yoy":       pd.to_numeric(latest.get("netprofit_yoy"), errors="coerce"),
                })
        except Exception as e:
            logger.debug("fina_indicator failed for %s: %s", code, e)
        time.sleep(0.05)   # Tushare rate limit: ~200 calls/min on free tier

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ts_code")


# ── AkShare fallback (per-stock, slow) ────────────────────────────────────────

def fetch_valuation_akshare(ts_code: str) -> dict:
    """
    Fetch latest PE/PB/PS for a single stock via AkShare.
    AkShare function: stock_a_indicator_lg (renamed from stock_a_lg_indicator).
    Confirmed columns: pe, pe_ttm, pb, ps, total_mv
    """
    import akshare as ak

    symbol = ts_code.split(".")[0]
    row = {"ts_code": ts_code, "pe_ttm": float("nan"),
           "pb": float("nan"), "ps_ttm": float("nan"), "total_mv_m": float("nan")}
    try:
        # Try new name first, then old name
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
        except AttributeError:
            df = ak.stock_a_lg_indicator(symbol=symbol)

        if df is not None and not df.empty:
            latest = df.iloc[-1]
            row["pe_ttm"]    = pd.to_numeric(latest.get("pe_ttm",  latest.get("pe")), errors="coerce")
            row["pb"]        = pd.to_numeric(latest.get("pb"),      errors="coerce")
            row["ps_ttm"]    = pd.to_numeric(latest.get("ps"),      errors="coerce")
            row["total_mv_m"] = pd.to_numeric(latest.get("total_mv"), errors="coerce")
    except Exception as e:
        logger.debug("AkShare valuation failed for %s: %s", ts_code, e)
    return row


def fetch_financial_akshare(ts_code: str) -> dict:
    """
    Fetch latest ROE, gross margin, debt ratio for a single stock via AkShare.
    AkShare confirmed columns (long-format pivot): 净资产收益率(%), 销售毛利率(%), 资产负债率(%)
    """
    import akshare as ak

    symbol = ts_code.split(".")[0]
    row = {"ts_code": ts_code, "roe_ttm": float("nan"),
           "gross_margin": float("nan"), "debt_ratio": float("nan"),
           "revenue_yoy": float("nan"), "profit_yoy": float("nan")}
    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2022")
        if df is None or df.empty:
            return row
        # AkShare returns this in a wide format where each metric is a column
        latest = df.iloc[0]   # most recent period is first
        row["roe_ttm"]     = _safe_float(latest, "净资产收益率(%)")
        row["gross_margin"] = _safe_float(latest, "销售毛利率(%)")
        row["debt_ratio"]  = _safe_float(latest, "资产负债率(%)")
    except Exception as e:
        logger.debug("AkShare financial failed for %s: %s", ts_code, e)
    return row


def _safe_float(row, key: str) -> float:
    try:
        return float(str(row.get(key, "nan")).replace("%", "").strip())
    except (ValueError, TypeError):
        return float("nan")


# ── Public interface ───────────────────────────────────────────────────────────

def build_fundamental_table(
    ts_codes: list[str],
    trade_date: str | None = None,
) -> pd.DataFrame:
    """
    Build cross-sectional fundamental table for the stock universe.
    Returns DataFrame indexed by ts_code.

    Columns: pe_ttm, pb, ps_ttm, total_mv_m  — from Tushare daily_basic (one bulk call, fast)
             roe_ttm, gross_margin, debt_ratio, revenue_yoy, profit_yoy — NaN
             (fina_indicator requires per-stock calls: 0.9s × 800 = 12 min; skipped)
    """
    val_df = fetch_valuation_tushare(trade_date)

    if val_df.empty:
        logger.info("Tushare daily_basic unavailable — falling back to AkShare per-stock.")
        val_rows = [fetch_valuation_akshare(code) for code in ts_codes]
        val_df = pd.DataFrame(val_rows).set_index("ts_code")

    val_df = val_df.reindex(ts_codes)

    # Quality/growth factors left as NaN — z-score becomes 0 → factor effectively neutral.
    for col in ["roe_ttm", "gross_margin", "debt_ratio", "revenue_yoy", "profit_yoy"]:
        val_df[col] = float("nan")

    return val_df
