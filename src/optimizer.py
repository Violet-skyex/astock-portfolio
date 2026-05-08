"""
Portfolio weight optimizer — three modes.

  factor_score  (default): weights proportional to composite score,
                            minimum 5% floor, remainder to cash.
  equal_weight           : 1/N for selected positions.
  risk_parity            : each position contributes equal portfolio risk (vol).

Cash allocation:
  In factor_score mode, positions whose score-implied weight falls below
  MIN_WEIGHT are dropped. The unallocated fraction stays as cash.
  In equal/risk-parity modes, N is determined by how many stocks pass
  the score threshold (composite_score > 0 by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from .config import (
    TRADING_DAYS_YEAR, RISK_FREE_RATE,
    DEFAULT_MIN_WEIGHT, DEFAULT_MAX_WEIGHT,
)

WeightMode = str  # "factor_score" | "equal_weight" | "risk_parity"


# ── Factor-score weighted ──────────────────────────────────────────────────────

def factor_score_weights(
    factor_df: pd.DataFrame,
    top_n: int,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, float]:
    """
    Assign weights proportional to composite score for the top_n candidates.

    Steps:
      1. Take top_n by composite score (scores must be positive for meaningful weights)
      2. Shift scores so minimum = 0, then normalize
      3. Drop positions below min_weight floor (their allocation becomes cash)
      4. Renormalize remaining positions; cap at max_weight iteratively

    Returns: (weights_series, cash_fraction)
      cash_fraction = 1 - weights_series.sum()
    """
    candidates = factor_df.head(top_n).copy()

    # Shift so all scores are non-negative before proportional allocation
    scores = candidates["Composite Score"]
    scores = scores - scores.min()
    total = scores.sum()
    if total == 0:
        raw_weights = pd.Series(1.0 / len(candidates), index=candidates.index)
    else:
        raw_weights = scores / total

    # Drop below floor → becomes cash
    keep_mask = raw_weights >= min_weight
    kept = raw_weights[keep_mask]
    if kept.empty:
        # All below floor — hold cash entirely
        return pd.Series(dtype=float), 1.0

    # Renormalize kept positions
    kept = kept / kept.sum()

    # Cap at max_weight
    kept = _cap_and_renorm(kept.values, kept.index, max_weight)

    cash = 1.0 - kept.sum()
    return kept, max(cash, 0.0)


def equal_weights(
    tickers: list[str],
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, float]:
    """1/N equal weight across selected tickers. No cash."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float), 1.0
    w = min(1.0 / n, max_weight)
    weights = pd.Series(w, index=tickers)
    weights /= weights.sum()
    return weights, 0.0


def risk_parity_weights(
    returns: pd.DataFrame,
    tickers: list[str],
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, float]:
    """
    Equal risk contribution weights (risk parity).
    Solved via CVXPY: minimise sum of squared differences in risk contributions.
    Falls back to equal weight if optimisation fails.
    """
    ret_sub = returns[tickers].dropna()
    if len(ret_sub) < 20 or len(tickers) < 2:
        return equal_weights(tickers, max_weight)

    sigma = ret_sub.cov().values * TRADING_DAYS_YEAR
    n = len(tickers)

    w = cp.Variable(n, nonneg=True)
    risk_contrib = cp.multiply(w, sigma @ w)
    avg_contrib  = cp.sum(risk_contrib) / n

    objective = cp.Minimize(cp.sum_squares(risk_contrib - avg_contrib))
    constraints = [cp.sum(w) == 1, w <= max_weight]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return equal_weights(tickers, max_weight)

    weights_arr = np.maximum(w.value, 0)
    weights_arr /= weights_arr.sum()
    weights = pd.Series(weights_arr, index=tickers)
    return weights, 0.0


def build_portfolio(
    factor_df: pd.DataFrame,
    returns: pd.DataFrame,
    mode: WeightMode = "factor_score",
    top_n: int = 20,
    max_positions: int = 10,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    score_threshold: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    """
    Main portfolio construction entry point.

    1. Pre-select top_n candidates by composite score.
    2. Apply score_threshold filter (negative-composite stocks excluded).
    3. Run chosen weight mode.
    4. Enforce max_positions.

    Returns: (portfolio_df, cash_fraction)
      portfolio_df has columns: Weight + all factor columns
    """
    candidates = factor_df.head(top_n)
    candidates = candidates[candidates["Composite Score"] > score_threshold]
    if candidates.empty:
        return pd.DataFrame(), 1.0

    # Limit to max_positions
    candidates = candidates.head(max_positions)
    tickers = candidates.index.tolist()

    if mode == "factor_score":
        weights, cash = factor_score_weights(candidates, len(tickers), min_weight, max_weight)
    elif mode == "equal_weight":
        weights, cash = equal_weights(tickers, max_weight)
    elif mode == "risk_parity":
        avail = [t for t in tickers if t in returns.columns]
        weights, cash = risk_parity_weights(returns, avail, max_weight)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = candidates.reindex(weights.index).copy()
    result.insert(0, "Weight", weights.values)
    result = result.sort_values("Weight", ascending=False)
    return result, cash


def portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
) -> dict:
    """Compute ex-post portfolio statistics over the returns window."""
    ret_sub = returns[weights.index]
    w = weights.values
    port_ret = ret_sub @ w
    ann_ret = port_ret.mean() * TRADING_DAYS_YEAR
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    max_dd  = _max_drawdown(port_ret)
    return {
        "Annualised Return":     ann_ret,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio":          sharpe,
        "Max Drawdown":          max_dd,
    }


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return ((cumulative - peak) / peak).min()


def _cap_and_renorm(
    weights: np.ndarray,
    index: pd.Index,
    cap: float,
    max_iter: int = 50,
) -> pd.Series:
    w = weights.copy()
    for _ in range(max_iter):
        excess = np.maximum(w - cap, 0)
        if excess.sum() < 1e-10:
            break
        w = np.minimum(w, cap)
        uncapped = w < cap
        if uncapped.sum() == 0:
            break
        w[uncapped] += excess.sum() / uncapped.sum()
    total = w.sum()
    return pd.Series(w / total if total > 0 else w, index=index)
