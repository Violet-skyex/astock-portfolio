"""
Capital flow data for A-share stocks.

Primary  : Tushare Pro `moneyflow` — batch, globally accessible.
           Fields: buy_lg_amount, sell_lg_amount (大单), net_mf_amount (主力净流入)
Fallback : AkShare per-stock (requires China/HK IP, slow).

North money (北向资金) per-stock is not available in Tushare free batch form;
the north_net_buy column is left as 0.0 and the factor is effectively disabled.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logger = logging.getLogger(__name__)


# ── Tushare moneyflow (primary) ────────────────────────────────────────────────

def _fetch_moneyflow_tushare(
    ts_codes: list[str],
    window: int = 20,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Batch-fetch large-order / main-force net inflow via Tushare moneyflow.
    Returns DataFrame indexed by ts_code with columns:
      large_order_net — cumulative (buy_lg - sell_lg) over last `window` trading days
      main_force_net  — cumulative net_mf_amount over last `window` trading days
    """
    import tushare as ts
    from ..config import TUSHARE_TOKEN

    pro = ts.pro_api(TUSHARE_TOKEN)

    end_dt   = pd.Timestamp.today()
    # fetch ~2× window calendar days to cover weekends / holidays
    start_dt = end_dt - pd.Timedelta(days=int(window * 2.2))
    end_date   = end_dt.strftime("%Y%m%d")
    start_date = start_dt.strftime("%Y%m%d")

    fields = "ts_code,trade_date,buy_lg_amount,sell_lg_amount,net_mf_amount"
    ts_set = set(ts_codes)
    parts: list[pd.DataFrame] = []

    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i : i + batch_size]
        for attempt in range(3):
            try:
                df = pro.moneyflow(
                    ts_code=",".join(batch),
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                )
                if df is not None and not df.empty:
                    parts.append(df[df["ts_code"].isin(ts_set)])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    logger.warning("Tushare moneyflow batch failed: %s", e)
        time.sleep(0.3)

    if not parts:
        return pd.DataFrame()

    raw = pd.concat(parts, ignore_index=True)
    for col in ["buy_lg_amount", "sell_lg_amount", "net_mf_amount"]:
        raw[col] = pd.to_numeric(raw.get(col), errors="coerce").fillna(0)

    raw["large_order_net"] = raw["buy_lg_amount"] - raw["sell_lg_amount"]

    # Keep only the most recent `window` trading days per stock
    raw = raw.sort_values("trade_date")
    raw = raw.groupby("ts_code").tail(window)

    result = raw.groupby("ts_code").agg(
        large_order_net=("large_order_net", "sum"),
        main_force_net=("net_mf_amount",    "sum"),
    )
    return result


# ── AkShare fallback (per-stock, China/HK IP only) ────────────────────────────

import akshare as ak

_MARKET_MAP = {"SH": "sh", "SZ": "sz"}


def _fetch_large_order_one(ts_code: str, window: int = 20) -> dict:
    symbol = ts_code.split(".")[0]
    market = _MARKET_MAP.get(ts_code.split(".")[-1], "sh")
    result = {"ts_code": ts_code, "large_order_net": 0.0, "main_force_net": 0.0}
    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if df is None or df.empty:
            return result
        df = df.tail(window)
        large_col = next((c for c in df.columns if "大单净流入" in c and "净额" in c), None)
        main_col  = next((c for c in df.columns if "主力净流入" in c and "净额" in c), None)
        if large_col:
            result["large_order_net"] = pd.to_numeric(df[large_col], errors="coerce").sum()
        if main_col:
            result["main_force_net"]  = pd.to_numeric(df[main_col],  errors="coerce").sum()
    except Exception as e:
        logger.debug("AkShare fund_flow failed for %s: %s", ts_code, e)
    return result


def _batch_fetch_akshare(
    ts_codes: list[str],
    window: int,
    max_workers: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    results = []
    done = 0

    def _worker(code):
        time.sleep(0.4)
        return _fetch_large_order_one(code, window)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, code): code for code in ts_codes}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                pass
            done += 1
            if progress_callback:
                progress_callback(done, len(ts_codes))

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index("ts_code")


# ── Public interface ───────────────────────────────────────────────────────────

def build_fund_flow_table(
    ts_codes: list[str],
    window: int = 20,
    max_workers: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Build cross-sectional fund flow table.
    Returns DataFrame indexed by ts_code with columns:
      large_order_net, main_force_net, north_net_buy (always 0 — no per-stock source)
    """
    from ..config import TUSHARE_TOKEN

    flow_df = pd.DataFrame()

    if TUSHARE_TOKEN:
        try:
            logger.info("Fetching fund flow via Tushare moneyflow (%d stocks)…", len(ts_codes))
            flow_df = _fetch_moneyflow_tushare(ts_codes, window=window)
        except Exception as e:
            logger.warning("Tushare moneyflow failed: %s", e)

    if flow_df.empty:
        logger.info("Tushare unavailable — trying AkShare fund flow (requires China IP)…")
        flow_df = _batch_fetch_akshare(ts_codes, window,
                                       max_workers=max_workers,
                                       progress_callback=progress_callback)

    if flow_df.empty:
        flow_df = pd.DataFrame(0.0, index=ts_codes,
                               columns=["large_order_net", "main_force_net"])

    flow_df["north_net_buy"] = 0.0
    return flow_df.reindex(ts_codes).fillna(0.0)
