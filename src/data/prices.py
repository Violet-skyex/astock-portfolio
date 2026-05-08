"""
Daily price and volume data for A-share stocks and ETFs.

Data source routing (auto):
  Tushare Pro   — globally accessible; used when TUSHARE_TOKEN is set (stocks only)
  AkShare       — fallback; stock_zh_a_hist requires China/HK IP (East Money geo-restriction)

AkShare confirmed column names:
  stock_zh_a_hist    → 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
  fund_etf_hist_em   → 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
  stock_zh_index_daily → date, open, high, low, close, volume  (English columns)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

_STOCK_COLS = {
    "日期":   "trade_date",
    "收盘":   "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover_rate",
}


# ── Single-ticker fetch ────────────────────────────────────────────────────────

def _fetch_one_stock(
    ts_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV for a single stock.
    Returns DataFrame indexed by trade_date (datetime), columns = ts_code multiindex.
    Returns None on failure.

    Note: stock_zh_a_hist uses East Money's securities API — requires China/HK IP.
    """
    symbol = ts_code.split(".")[0]
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                logger.debug("stock_zh_a_hist failed for %s after %d attempts: %s", ts_code, retries, e)
                return None

    if df is None or df.empty:
        return None

    df = df.rename(columns=_STOCK_COLS)
    df = df[df.columns.intersection(list(_STOCK_COLS.values()))].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")

    # Ensure numeric
    for col in ["close", "volume", "amount", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = pd.MultiIndex.from_tuples([(ts_code, c) for c in df.columns])
    return df


def _fetch_one_etf(
    ts_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch daily data for a single A-share ETF."""
    symbol = ts_code.split(".")[0]
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                logger.debug("fund_etf_hist_em failed for %s after %d attempts: %s", ts_code, retries, e)
                return None

    if df is None or df.empty:
        return None

    df = df.rename(columns=_STOCK_COLS)
    df = df[df.columns.intersection(list(_STOCK_COLS.values()))].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")

    for col in ["close", "volume", "amount", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = pd.MultiIndex.from_tuples([(ts_code, c) for c in df.columns])
    return df


# ── Batch fetch ────────────────────────────────────────────────────────────────

def fetch_prices_batch(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    is_etf: bool = False,
    max_workers: int = 8,
    delay_per_worker: float = 0.2,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch daily close, volume, amount, turnover for a list of A-share tickers in parallel.

    Parameters
    ----------
    ts_codes         : list of ts_code strings, e.g. ['600519.SH', '000858.SZ']
    start_date       : 'YYYY-MM-DD'
    end_date         : 'YYYY-MM-DD'
    is_etf           : True = use fund_etf_hist_em, False = use stock_zh_a_hist
    max_workers      : ThreadPoolExecutor concurrency limit
    delay_per_worker : seconds to sleep in each worker thread to avoid rate limiting
    progress_callback: callable(done: int, total: int) for Streamlit progress bar

    Returns
    -------
    (prices, volume, amount, turnover) — each a DataFrame [trade_date × ts_code]
    Missing tickers are silently skipped; dates are the union of all available dates.
    """
    fetch_fn = _fetch_one_etf if is_etf else _fetch_one_stock
    frames: dict[str, pd.DataFrame] = {}
    total = len(ts_codes)
    done  = 0

    def _worker(code: str) -> tuple[str, pd.DataFrame | None]:
        time.sleep(delay_per_worker)
        return code, fetch_fn(code, start_date, end_date)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, code): code for code in ts_codes}
        for fut in as_completed(futures):
            code, result = fut.result()
            if result is not None:
                frames[code] = result
            done += 1
            if progress_callback:
                progress_callback(done, total)

    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    combined = pd.concat(frames.values(), axis=1).sort_index()

    def _extract(field: str) -> pd.DataFrame:
        try:
            return combined.xs(field, axis=1, level=1)
        except KeyError:
            return pd.DataFrame(index=combined.index)

    prices   = _extract("close")
    volume   = _extract("volume")
    amount   = _extract("amount")
    turnover = _extract("turnover_rate")

    # Forward-fill up to 5 days to handle suspension periods (A-share 停牌)
    prices   = prices.ffill(limit=5)
    turnover = turnover.fillna(0)   # suspended stocks have 0 turnover

    return prices, volume, amount, turnover


# ── Tushare price fetching (global access) ────────────────────────────────────

def _quarterly_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Return (YYYYMMDD, YYYYMMDD) quarterly chunks covering [start_date, end_date]."""
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        q_month = ((cur.month - 1) // 3 + 1) * 3
        q_end = pd.Timestamp(year=cur.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
        q_end = min(q_end, e)
        chunks.append((cur.strftime("%Y%m%d"), q_end.strftime("%Y%m%d")))
        cur = q_end + pd.Timedelta(days=1)
    return chunks


def _fetch_one_stock_tushare(
    ts_code: str,
    start_date: str,
    end_date: str,
    pro,
    retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch single stock daily data via Tushare Pro (close, vol, amount)."""
    sd = start_date.replace("-", "")
    ed = end_date.replace("-", "")

    for attempt in range(retries):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=sd,
                end_date=ed,
                fields="ts_code,trade_date,close,vol,amount",
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                logger.debug("Tushare daily failed for %s: %s", ts_code, e)
                return None

    if df is None or df.empty:
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()

    for col in ["close", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["close", "vol", "amount"]].copy()
    df.columns = pd.MultiIndex.from_tuples([(ts_code, c) for c in df.columns])
    return df


def _fetch_turnover_tushare(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    pro,
    date_index: pd.Index,
) -> pd.DataFrame:
    """Fetch turnover_rate from Tushare daily_basic in quarterly chunks."""
    ts_set = set(ts_codes)
    parts: list[pd.DataFrame] = []

    for sd, ed in _quarterly_chunks(start_date, end_date):
        try:
            db = pro.daily_basic(
                start_date=sd,
                end_date=ed,
                fields="ts_code,trade_date,turnover_rate",
            )
            if db is not None and not db.empty:
                parts.append(db[db["ts_code"].isin(ts_set)])
        except Exception as e:
            logger.warning("Tushare daily_basic failed (%s-%s): %s", sd, ed, e)
        time.sleep(0.5)

    if not parts:
        return pd.DataFrame(0.0, index=date_index, columns=ts_codes)

    raw = pd.concat(parts, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["turnover_rate"] = pd.to_numeric(raw["turnover_rate"], errors="coerce")

    pivot = raw.pivot_table(index="trade_date", columns="ts_code", values="turnover_rate")
    return pivot.reindex(index=date_index, columns=ts_codes).fillna(0.0)


def fetch_prices_tushare(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    max_workers: int = 5,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch price/volume data via Tushare Pro (globally accessible, no IP restriction).

    Returns (prices, volume, amount, turnover) — same shape as fetch_prices_batch.
    Requires TUSHARE_TOKEN to be set in config.
    """
    import tushare as ts
    from ..config import TUSHARE_TOKEN

    if not TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN not configured — cannot use Tushare price source")

    pro = ts.pro_api(TUSHARE_TOKEN)
    frames: dict[str, pd.DataFrame] = {}
    total = len(ts_codes)
    done = 0

    def _worker(code: str) -> tuple[str, pd.DataFrame | None]:
        time.sleep(0.2)
        return code, _fetch_one_stock_tushare(code, start_date, end_date, pro)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, code): code for code in ts_codes}
        for fut in as_completed(futures):
            code, result = fut.result()
            if result is not None:
                frames[code] = result
            done += 1
            if progress_callback:
                progress_callback(done, total)

    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    combined = pd.concat(frames.values(), axis=1).sort_index()

    def _extract(field: str) -> pd.DataFrame:
        try:
            return combined.xs(field, axis=1, level=1)
        except KeyError:
            return pd.DataFrame(index=combined.index)

    prices = _extract("close")
    volume = _extract("vol")
    amount = _extract("amount")
    prices = prices.ffill(limit=5)

    turnover = _fetch_turnover_tushare(ts_codes, start_date, end_date, pro, prices.index)

    return prices, volume, amount, turnover


def fetch_prices_auto(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    is_etf: bool = False,
    max_workers: int = 8,
    delay_per_worker: float = 0.2,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Auto-routing price fetcher.

    For stocks: tries Tushare Pro first (works globally), falls back to AkShare
                (requires China/HK IP).
    For ETFs  : uses AkShare fund_etf_hist_em directly (Tushare fund_daily less reliable).
    """
    from ..config import TUSHARE_TOKEN

    if not is_etf and TUSHARE_TOKEN:
        try:
            logger.info("Fetching prices via Tushare Pro (%d tickers)…", len(ts_codes))
            return fetch_prices_tushare(
                ts_codes, start_date, end_date,
                max_workers=min(max_workers, 5),
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.warning("Tushare fetch failed, falling back to AkShare: %s", e)

    return fetch_prices_batch(
        ts_codes, start_date, end_date,
        is_etf=is_etf,
        max_workers=max_workers,
        delay_per_worker=delay_per_worker,
        progress_callback=progress_callback,
    )


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from close prices. First row is dropped."""
    return prices.pct_change().dropna(how="all")


# ── Benchmark index ────────────────────────────────────────────────────────────

def fetch_index_daily(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Fetch daily close for a benchmark index.
    index_code format: '000300.SH' or '000905.SH'

    AkShare symbol format: 'sh000300' (Shanghai) or 'sz000905' (Shenzhen).
    Returns Series indexed by datetime, named with index_code.
    """
    sym = index_code.split(".")[0]
    prefix = "sh" if index_code.endswith(".SH") else "sz"
    ak_sym = f"{prefix}{sym}"

    try:
        df = ak.stock_zh_index_daily(symbol=ak_sym)
    except Exception as e:
        logger.error("stock_zh_index_daily failed for %s: %s", ak_sym, e)
        return pd.Series(dtype=float, name=index_code)

    # English column names: date, open, high, low, close, volume
    if "date" not in df.columns or "close" not in df.columns:
        logger.error("Unexpected index_daily columns: %s", df.columns.tolist())
        return pd.Series(dtype=float, name=index_code)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    close = pd.to_numeric(df["close"], errors="coerce").rename(index_code)

    # Slice to requested date range
    mask = (close.index >= pd.to_datetime(start_date)) & (close.index <= pd.to_datetime(end_date))
    return close[mask]
