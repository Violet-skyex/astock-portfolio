"""
Capital flow data for A-share stocks via AkShare.

  Large orders : 大单净流入 via stock_individual_fund_flow
                 AkShare confirmed columns include:
                   日期, 大单净流入-净额, 主力净流入-净额, 超大单净流入-净额

  North money  : 北向资金 per stock via stock_hsgt_individual_em
                 Column names vary — detected dynamically.

Both signals return ~100 days of history per call. We take a rolling sum
over the last `window` trading days.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

_MARKET_MAP = {
    "SH": "sh",
    "SZ": "sz",
}


# ── Large order flow ───────────────────────────────────────────────────────────

def _fetch_large_order_one(ts_code: str, window: int = 20) -> dict:
    """
    Fetch rolling `window`-day large order net inflow for one stock.
    AkShare confirmed columns: 大单净流入-净额 (万元)
    """
    symbol = ts_code.split(".")[0]
    market = _MARKET_MAP.get(ts_code.split(".")[-1], "sh")
    result = {"ts_code": ts_code, "large_order_net": float("nan"),
              "main_force_net": float("nan")}
    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if df is None or df.empty:
            return result

        df = df.tail(window)

        # 大单净流入-净额
        large_col = next((c for c in df.columns if "大单净流入" in c and "净额" in c), None)
        if large_col:
            result["large_order_net"] = pd.to_numeric(df[large_col], errors="coerce").sum()

        # 主力净流入-净额 (includes 超大单 + 大单)
        main_col = next((c for c in df.columns if "主力净流入" in c and "净额" in c), None)
        if main_col:
            result["main_force_net"] = pd.to_numeric(df[main_col], errors="coerce").sum()

    except Exception as e:
        logger.debug("fund_flow failed for %s: %s", ts_code, e)
    return result


# ── Northbound capital ─────────────────────────────────────────────────────────

def _fetch_north_money_one(ts_code: str, window: int = 20) -> dict:
    """
    Fetch rolling `window`-day northbound net buy for one stock.
    AkShare function: stock_hsgt_individual_em

    Column detection is dynamic because AkShare naming is inconsistent across versions.
    We look for a column containing '净买入' or '净额'.
    """
    symbol = ts_code.split(".")[0]
    result = {"ts_code": ts_code, "north_net_buy": float("nan")}
    try:
        df = ak.stock_hsgt_individual_em(symbol=symbol)
        if df is None or df.empty:
            return result

        df = df.tail(window)

        net_col = next(
            (c for c in df.columns if "净买入" in c or ("净额" in c and "买" in c)), None
        )
        if net_col is None:
            # Try any column with 净 that looks like a numeric amount
            net_col = next((c for c in df.columns if "净" in c), None)

        if net_col:
            result["north_net_buy"] = pd.to_numeric(df[net_col], errors="coerce").sum()

    except Exception as e:
        logger.debug("northbound failed for %s: %s", ts_code, e)
    return result


# ── Batch fetch ────────────────────────────────────────────────────────────────

def _batch_fetch(
    ts_codes: list[str],
    fetch_fn,
    max_workers: int = 5,
    delay: float = 0.5,
    progress_callback=None,
) -> list[dict]:
    """Generic parallel batch wrapper with polite rate limiting."""
    results = []
    done = 0
    total = len(ts_codes)

    def _worker(code):
        time.sleep(delay)
        return fetch_fn(code)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, code): code for code in ts_codes}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                code = futures[fut]
                logger.debug("batch fetch error for %s: %s", code, e)
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return results


def build_fund_flow_table(
    ts_codes: list[str],
    window: int = 20,
    max_workers: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Build cross-sectional fund flow table for the given stock universe.

    Returns DataFrame indexed by ts_code with columns:
      large_order_net  — rolling `window`-day large order net inflow (万元)
      main_force_net   — rolling `window`-day main force net inflow (万元)
      north_net_buy    — rolling `window`-day northbound net buy (万元)

    Note: AkShare's northbound per-stock data may not be available for all stocks
    (only 沪港通/深港通 eligible stocks have northbound data).
    Missing values are filled with 0 before factor scoring.
    """
    # Large order flow
    logger.info("Fetching large order flow for %d stocks…", len(ts_codes))
    flow_rows = _batch_fetch(
        ts_codes,
        lambda code: _fetch_large_order_one(code, window),
        max_workers=max_workers,
        delay=0.4,
        progress_callback=progress_callback,
    )
    flow_df = pd.DataFrame(flow_rows).set_index("ts_code")

    # Northbound capital (only SH+SZ eligible, ~1000+ stocks)
    # Limit to avoid excessive API calls — skip this for very large universes
    north_eligible = [c for c in ts_codes if not c.startswith("688")]  # STAR市场无北向
    logger.info("Fetching northbound data for %d stocks…", len(north_eligible))
    north_rows = _batch_fetch(
        north_eligible,
        lambda code: _fetch_north_money_one(code, window),
        max_workers=max_workers,
        delay=0.5,
    )
    north_df = (
        pd.DataFrame(north_rows).set_index("ts_code")
        if north_rows else pd.DataFrame(columns=["north_net_buy"])
    )

    combined = flow_df.join(north_df[["north_net_buy"]], how="left")
    return combined.reindex(ts_codes).fillna(0.0)
