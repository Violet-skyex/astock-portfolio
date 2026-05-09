"""
Rolling walk-forward backtesting engine for A-share portfolio.

Architecture
------------
- Uses ONLY price-based technical factors at each rebalancing step,
  because historical fundamental/fund-flow data is not stored locally.
  (Fundamental factors are used for the current recommendation, not backtest.)
- Current index constituents are used for the full window (accepts survivorship bias).
- Transaction cost: TRANSACTION_COST applied to portfolio turnover at each rebalance.

Two public entry points:
  run_stock_backtest(...)  — for individual stocks
  run_etf_backtest(...)    — for ETFs

Both return a BacktestResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .config import TRANSACTION_COST, TRADING_DAYS_YEAR, RISK_FREE_RATE
from .factors.technical import (
    momentum_1m, momentum_3m, momentum_6m,
    volatility_20d, avg_turnover_20d, volume_ratio,
)
from .factors.utils import zscore
from .optimizer import build_portfolio, _cap_and_renorm

RebalFreq = Literal["daily", "weekly", "biweekly", "monthly"]

FREQ_DAYS: dict[str, int] = {
    "daily":    1,
    "weekly":   5,
    "biweekly": 10,
    "monthly":  21,
}


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    nav:           pd.Series              # cumulative NAV, starts at 1.0
    port_returns:  pd.Series              # daily portfolio returns (after costs)
    benchmark_nav: pd.Series              # benchmark cumulative NAV
    holdings:      pd.DataFrame           # sparse weight matrix over time
    turnover:      pd.Series              # per-rebalance turnover fraction
    metrics:       dict = field(default_factory=dict)
    factor_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    # factor_history: [rebal_date × ticker] composite scores at each rebalancing


# ── Technical-only factor table for backtest ──────────────────────────────────

def _technical_factor_table(
    prices:   pd.DataFrame,
    volume:   pd.DataFrame,
    turnover: pd.DataFrame,
    returns:  pd.DataFrame,
    w_mom_3m: float = 0.40,
    w_mom_6m: float = 0.30,
    w_mom_1m: float = 0.10,
    w_vol:    float = 0.10,
    w_liq:    float = 0.10,
) -> pd.DataFrame:
    """
    Compute cross-sectional technical factor table on a price/volume window.
    All tickers with < 40 days of price history in the window are dropped.
    """
    min_obs = min(40, int(len(prices) * 0.6))
    valid   = prices.columns[prices.notna().sum() >= min_obs]
    p = prices[valid]
    r = returns[valid]
    v = volume[valid] if not volume.empty else pd.DataFrame(index=prices.index, columns=valid)
    t = turnover[valid] if not turnover.empty else pd.DataFrame(index=prices.index, columns=valid)

    if len(p) < 21:
        return pd.DataFrame()

    mom1  = momentum_1m(p)
    mom3  = momentum_3m(p)  if len(p) >= 63  else pd.Series(dtype=float)
    mom6  = momentum_6m(p)  if len(p) >= 126 else pd.Series(dtype=float)
    vol20 = volatility_20d(r)
    liq20 = avg_turnover_20d(t) if not t.empty else pd.Series(dtype=float)

    # Normalise available factors
    z_mom1 = zscore(mom1.dropna())
    z_mom3 = zscore(mom3.dropna()) if not mom3.empty else pd.Series(0.0, index=valid)
    z_mom6 = zscore(mom6.dropna()) if not mom6.empty else pd.Series(0.0, index=valid)
    z_vol  = -zscore(vol20.dropna())   # inverted: lower vol is better
    z_liq  = zscore(liq20.dropna()) if not liq20.empty else pd.Series(0.0, index=valid)

    idx = pd.Index(valid)
    composite = (
        w_mom_1m * z_mom1.reindex(idx).fillna(0) +
        w_mom_3m * z_mom3.reindex(idx).fillna(0) +
        w_mom_6m * z_mom6.reindex(idx).fillna(0) +
        w_vol    * z_vol.reindex(idx).fillna(0)  +
        w_liq    * z_liq.reindex(idx).fillna(0)
    )

    df = pd.DataFrame({
        "Mom 1M":          mom1.reindex(idx),
        "Mom 3M":          mom3.reindex(idx) if not mom3.empty else float("nan"),
        "Mom 6M":          mom6.reindex(idx) if not mom6.empty else float("nan"),
        "Vol 20D":         vol20.reindex(idx),
        "Turnover Avg":    liq20.reindex(idx) if not liq20.empty else float("nan"),
        "Composite Score": composite,
    }, index=idx)

    return df.dropna(subset=["Composite Score"]).sort_values("Composite Score", ascending=False)


# ── Core walk-forward loop ────────────────────────────────────────────────────

def _walk_forward(
    prices:         pd.DataFrame,
    volume:         pd.DataFrame,
    turnover:       pd.DataFrame,
    returns:        pd.DataFrame,
    benchmark:      pd.Series,
    freq:           RebalFreq,
    lookback_days:  int,
    max_positions:  int,
    mode:           str,
    factor_kwargs:  dict,
    optimizer_kwargs: dict,
) -> BacktestResult:
    """Shared walk-forward loop for stocks and ETFs."""
    all_dates = returns.index.sort_values()
    if len(all_dates) < lookback_days + FREQ_DAYS[freq]:
        raise ValueError(
            f"Need at least {lookback_days + FREQ_DAYS[freq]} trading days of data, "
            f"got {len(all_dates)}."
        )

    rebal_interval = FREQ_DAYS[freq]
    backtest_dates = all_dates[lookback_days:]
    rebal_dates    = set(backtest_dates[::rebal_interval])

    current_weights: pd.Series = pd.Series(dtype=float)
    nav   = 1.0
    nav_vals:       list[tuple] = []
    turnover_vals:  list[tuple] = []
    holdings_rows:  list        = []
    factor_rows:    dict        = {}

    for date in backtest_dates:
        is_rebal = date in rebal_dates

        if is_rebal:
            loc = all_dates.get_loc(date)
            start = max(0, loc - lookback_days)
            hist_idx = all_dates[start:loc]

            p_win = prices.loc[hist_idx]
            r_win = returns.loc[hist_idx]
            v_win = volume.loc[hist_idx]
            t_win = turnover.loc[hist_idx]

            factor_df = _technical_factor_table(p_win, v_win, t_win, r_win, **factor_kwargs)

            if not factor_df.empty:
                try:
                    new_port, _ = build_portfolio(
                        factor_df, r_win,
                        mode=mode,
                        top_n=len(factor_df),
                        max_positions=max_positions,
                        **optimizer_kwargs,
                    )
                    new_weights = new_port["Weight"] if not new_port.empty else pd.Series(dtype=float)
                except Exception as exc:
                    import logging as _log
                    _log.getLogger(__name__).warning("build_portfolio failed at %s: %s", date, exc)
                    new_weights = current_weights.copy()
            else:
                new_weights = current_weights.copy()

            # Turnover and transaction cost
            old_w  = current_weights.reindex(new_weights.index).fillna(0)
            to_val = float((new_weights - old_w).abs().sum() / 2)
            nav   *= (1.0 - to_val * TRANSACTION_COST)
            turnover_vals.append((date, to_val))

            current_weights = new_weights
            if not factor_df.empty:
                factor_rows[date] = factor_df["Composite Score"]

        # Daily portfolio return
        if current_weights.empty:
            port_ret = 0.0
        else:
            avail    = current_weights.index.intersection(returns.columns)
            day_rets = returns.loc[date, avail].fillna(0)
            port_ret = float((day_rets * current_weights[avail]).sum())

        nav *= (1.0 + port_ret)
        nav_vals.append((date, nav))
        holdings_rows.append(current_weights.rename(date))

    nav_series    = pd.Series(dict(nav_vals), name="NAV")
    port_returns  = nav_series.pct_change().fillna(0)
    turnover_ser  = pd.Series(dict(turnover_vals), name="Turnover")
    holdings_df   = pd.DataFrame(holdings_rows).fillna(0)

    # Benchmark alignment
    bench_slice = benchmark.reindex(backtest_dates).ffill()
    bench_nav   = (bench_slice / bench_slice.iloc[0]).rename("Benchmark")

    factor_hist = pd.DataFrame(factor_rows).T if factor_rows else pd.DataFrame()

    metrics = _compute_metrics(port_returns, bench_nav, nav_series)

    return BacktestResult(
        nav=nav_series,
        port_returns=port_returns,
        benchmark_nav=bench_nav,
        holdings=holdings_df,
        turnover=turnover_ser,
        metrics=metrics,
        factor_history=factor_hist,
    )


# ── Public entry points ────────────────────────────────────────────────────────

def run_stock_backtest(
    prices:        pd.DataFrame,
    volume:        pd.DataFrame,
    turnover:      pd.DataFrame,
    returns:       pd.DataFrame,
    benchmark:     pd.Series,
    freq:          RebalFreq = "monthly",
    lookback_days: int = 126,
    max_positions: int = 10,
    mode:          str = "factor_score",
    # Technical factor weights (must sum to 1)
    w_mom_1m: float = 0.10,
    w_mom_3m: float = 0.40,
    w_mom_6m: float = 0.30,
    w_vol:    float = 0.10,
    w_liq:    float = 0.10,
) -> BacktestResult:
    """
    Run walk-forward stock backtest using technical factors only.
    Fundamental/fund-flow factors are excluded (no historical data available).
    """
    factor_kw    = dict(w_mom_1m=w_mom_1m, w_mom_3m=w_mom_3m, w_mom_6m=w_mom_6m,
                        w_vol=w_vol, w_liq=w_liq)
    optimizer_kw = {}
    return _walk_forward(prices, volume, turnover, returns, benchmark,
                         freq, lookback_days, max_positions, mode,
                         factor_kw, optimizer_kw)


def run_etf_backtest(
    prices:        pd.DataFrame,
    volume:        pd.DataFrame,
    turnover:      pd.DataFrame,
    returns:       pd.DataFrame,
    benchmark:     pd.Series,
    freq:          RebalFreq = "monthly",
    lookback_days: int = 126,
    max_positions: int = 5,
    mode:          str = "factor_score",
) -> BacktestResult:
    """
    Run walk-forward ETF backtest. Uses ETF-appropriate technical factors
    (heavier weight on 3M/6M momentum, lower on liquidity since ETFs are liquid).
    """
    factor_kw    = dict(w_mom_1m=0.05, w_mom_3m=0.45, w_mom_6m=0.35,
                        w_vol=0.10, w_liq=0.05)
    optimizer_kw = {}
    return _walk_forward(prices, volume, turnover, returns, benchmark,
                         freq, lookback_days, max_positions, mode,
                         factor_kw, optimizer_kw)


def combine_backtests(
    stock_bt: BacktestResult,
    etf_bt:   BacktestResult,
    stock_weight: float = 0.7,
    etf_weight:   float = 0.3,
) -> BacktestResult:
    """
    Combine stock and ETF backtest results into a single blended portfolio.
    Aligns on common dates; fills missing with 0 return.
    """
    idx = stock_bt.port_returns.index.union(etf_bt.port_returns.index).sort_values()

    s_ret = stock_bt.port_returns.reindex(idx).fillna(0)
    e_ret = etf_bt.port_returns.reindex(idx).fillna(0)
    combined_ret = stock_weight * s_ret + etf_weight * e_ret

    combined_nav = (1 + combined_ret).cumprod().rename("NAV")

    # Use stock benchmark (same benchmark for both)
    bench_nav = stock_bt.benchmark_nav.reindex(idx).ffill()

    metrics = _compute_metrics(combined_ret, bench_nav, combined_nav)

    return BacktestResult(
        nav=combined_nav,
        port_returns=combined_ret,
        benchmark_nav=bench_nav,
        holdings=pd.DataFrame(),   # combined holdings not tracked separately
        turnover=pd.Series(dtype=float),
        metrics=metrics,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(
    port_returns: pd.Series,
    bench_nav:    pd.Series,
    nav:          pd.Series,
) -> dict:
    ann_ret  = port_returns.mean() * TRADING_DAYS_YEAR
    ann_vol  = port_returns.std()  * np.sqrt(TRADING_DAYS_YEAR)
    sharpe   = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    max_dd   = float(((nav / nav.cummax()) - 1).min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    bench_ret  = bench_nav.pct_change().fillna(0).reindex(port_returns.index).fillna(0)
    bench_ann  = bench_ret.mean() * TRADING_DAYS_YEAR

    # Alpha / Beta via OLS
    cov_mat   = np.cov(port_returns.values, bench_ret.values)
    bench_var = cov_mat[1, 1]
    beta      = float(cov_mat[0, 1] / bench_var) if bench_var > 0 else 1.0
    alpha     = float(ann_ret - RISK_FREE_RATE - beta * (bench_ann - RISK_FREE_RATE))

    # Win rate
    win_rate = float((port_returns > 0).sum() / max(len(port_returns), 1))

    return {
        "年化收益":   ann_ret,
        "年化波动":   ann_vol,
        "夏普比率":   sharpe,
        "最大回撤":   max_dd,
        "卡玛比率":   calmar,
        "Alpha":     alpha,
        "Beta":      beta,
        "胜率":      win_rate,
        "基准年化":   bench_ann,
        "超额收益":   ann_ret - bench_ann,
    }


def rolling_metrics(
    port_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio and max drawdown over a rolling window.
    Used for the rolling metrics chart in the backtest tab.
    """
    ann = TRADING_DAYS_YEAR
    roll_sharpe = (
        port_returns.rolling(window).mean() * ann /
        (port_returns.rolling(window).std() * np.sqrt(ann) + 1e-9)
    )
    cum = (1 + port_returns).cumprod()
    roll_dd = cum / cum.rolling(window).max() - 1

    return pd.DataFrame({
        "Rolling Sharpe":   roll_sharpe,
        "Rolling Drawdown": roll_dd,
    })
