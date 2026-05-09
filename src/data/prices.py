"""
Daily price and volume data for A-share stocks and ETFs.

Data source routing (fetch_prices_auto):
  1. yfinance  — primary; free, no API key, globally accessible.
                 A-share symbol mapping: 600519.SH → 600519.SS, 000858.SZ → 000858.SZ
  2. AkShare   — fallback (China/HK IP only); stock_zh_a_hist is East Money geo-restricted.

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
    is_etf: bool = False,
    progress_callback=None,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bulk quarterly batch fetch via Tushare Pro (globally accessible).

    Sends batches of `batch_size` codes per quarterly date chunk, so 800 stocks
    over 1 year = ~32 API calls instead of 800 individual calls.
    Works for both stocks (pro.daily) and ETFs (pro.fund_daily).
    """
    import tushare as ts
    from ..config import TUSHARE_TOKEN

    if not TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN not configured")

    pro = ts.pro_api(TUSHARE_TOKEN)
    api_fn   = pro.fund_daily if is_etf else pro.daily
    fields   = "ts_code,trade_date,close,vol,amount"
    ts_set   = set(ts_codes)

    date_chunks  = _quarterly_chunks(start_date, end_date)
    code_batches = [ts_codes[i : i + batch_size] for i in range(0, len(ts_codes), batch_size)]
    total_calls  = len(code_batches) * len(date_chunks)
    done         = 0

    daily_parts: list[pd.DataFrame] = []

    for sd, ed in date_chunks:
        for batch in code_batches:
            batch_str = ",".join(batch)
            for attempt in range(3):
                try:
                    df = api_fn(ts_code=batch_str, start_date=sd, end_date=ed, fields=fields)
                    if df is not None and not df.empty:
                        daily_parts.append(df)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                    else:
                        logger.warning(
                            "Tushare batch failed (%s…%s %s-%s): %s",
                            batch[0], batch[-1], sd, ed, e,
                        )
            done += 1
            if progress_callback:
                progress_callback(done, total_calls)
            time.sleep(0.35)   # stay within free-tier rate limit

    if not daily_parts:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    raw = pd.concat(daily_parts, ignore_index=True)
    raw = raw[raw["ts_code"].isin(ts_set)].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    for col in ["close", "vol", "amount"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    def _pivot(col: str) -> pd.DataFrame:
        if col not in raw.columns:
            return pd.DataFrame()
        return raw.pivot_table(index="trade_date", columns="ts_code",
                               values=col, aggfunc="last")

    prices = _pivot("close").sort_index()
    volume = _pivot("vol").sort_index()
    amount = _pivot("amount").sort_index()

    if prices.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    prices = prices.ffill(limit=5)

    # Turnover via daily_basic bulk would fetch ALL ~5000 A-shares (no ts_code filter
    # in Tushare bulk mode), causing multi-minute hangs. Skip it; scorer handles
    # zero turnover gracefully (liquidity factor = 0 instead of computed score).
    turnover = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    return prices, volume, amount, turnover


# ── yfinance price fetching (primary, global, free) ───────────────────────────

def _ts_to_yf(ts_code: str) -> str:
    """Convert ts_code to yfinance symbol: 600519.SH → 600519.SS, 000858.SZ → 000858.SZ"""
    code, exch = ts_code.split(".")
    return f"{code}.{'SS' if exch == 'SH' else 'SZ'}"


def fetch_prices_yfinance(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Batch download via yfinance (Yahoo Finance).
    Free, no API key, globally accessible. Covers all A-share stocks and ETFs.
    Returns (prices, volume, amount=empty, turnover=zeros).
    """
    import yfinance as yf

    yf_syms  = [_ts_to_yf(c) for c in ts_codes]
    sym_map  = {_ts_to_yf(c): c for c in ts_codes}   # yf_sym → ts_code

    if progress_callback:
        progress_callback(0, 1)

    try:
        raw = yf.download(
            yf_syms,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error("yfinance download failed: %s", e)
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    if progress_callback:
        progress_callback(1, 1)

    if raw.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    # MultiIndex columns: (field, yf_sym) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close_df  = raw["Close"].rename(columns=sym_map)
        volume_df = raw["Volume"].rename(columns=sym_map)
    else:
        # Single ticker — flat columns
        single = ts_codes[0]
        close_df  = raw[["Close"]].rename(columns={"Close": single})
        volume_df = raw[["Volume"]].rename(columns={"Volume": single})

    close_df  = close_df.sort_index().ffill(limit=5)
    volume_df = volume_df.sort_index()

    # Drop columns that are entirely NaN (yfinance couldn't find the symbol)
    close_df  = close_df.dropna(axis=1, how="all")
    volume_df = volume_df.reindex(columns=close_df.columns).fillna(0)

    amount_df   = pd.DataFrame(index=close_df.index)
    turnover_df = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)

    return close_df, volume_df, amount_df, turnover_df


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

    Primary : yfinance — free, global, no API key required.
    Fallback : AkShare — requires China/HK IP (East Money geo-restriction).
    """
    try:
        logger.info("Fetching %s prices via yfinance (%d tickers)…",
                    "ETF" if is_etf else "stock", len(ts_codes))
        result = fetch_prices_yfinance(ts_codes, start_date, end_date,
                                       progress_callback=progress_callback)
        prices = result[0]
        if not prices.empty:
            return result
        logger.warning("yfinance returned empty data, falling back to AkShare")
    except Exception as e:
        logger.warning("yfinance failed (%s), falling back to AkShare", e)

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

    Tries yfinance first (global) using Yahoo Finance index symbols,
    falls back to AkShare stock_zh_index_daily (China/HK IP required).

    Yahoo Finance index symbols:
      000300.SH (沪深300) → 000300.SS
      000905.SH (中证500) → 000905.SS
      000016.SH (上证50)  → 000016.SS
    """
    import yfinance as yf

    yf_sym = _ts_to_yf(index_code)
    try:
        raw = yf.download(yf_sym, start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
        if not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.rename(index_code)
    except Exception as e:
        logger.warning("yfinance index fetch failed for %s: %s — trying AkShare", index_code, e)

    # AkShare fallback
    sym    = index_code.split(".")[0]
    prefix = "sh" if index_code.endswith(".SH") else "sz"
    ak_sym = f"{prefix}{sym}"

    try:
        df = ak.stock_zh_index_daily(symbol=ak_sym)
    except Exception as e:
        logger.error("stock_zh_index_daily failed for %s: %s", ak_sym, e)
        return pd.Series(dtype=float, name=index_code)

    if "date" not in df.columns or "close" not in df.columns:
        logger.error("Unexpected index_daily columns: %s", df.columns.tolist())
        return pd.Series(dtype=float, name=index_code)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    close = pd.to_numeric(df["close"], errors="coerce").rename(index_code)
    mask  = (close.index >= pd.to_datetime(start_date)) & (close.index <= pd.to_datetime(end_date))
    return close[mask]
