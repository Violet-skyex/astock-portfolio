"""
Factor composite scoring.

Pipeline:
  1. Compute each raw factor series (from factors/)
  2. Cross-sectionally z-score each factor
  3. Shrink toward zero for low-data stocks (optional)
  4. Weighted sum → composite score
  5. Sort descending; stocks below min_weight threshold can become cash

Sentiment is added as a multiplicative overlay on top of the composite score
(not included in the z-score normalisation pool) so it doesn't distort
the cross-sectional ranking of the base model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import technical as T
from .factors import fundamental as F
from .factors import fund_flow as FF


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score. Returns zeros if std = 0."""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_stock_factor_table(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    turnover: pd.DataFrame,
    returns: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    fund_flow_df: pd.DataFrame,
    # Factor weights — normalised internally
    w_momentum:  float = 0.20,
    w_value:     float = 0.20,
    w_quality:   float = 0.15,
    w_growth:    float = 0.15,
    w_fund_flow: float = 0.15,
    w_liquidity: float = 0.15,
    # Momentum sub-weights (sum to 1)
    w_mom_1m: float = 0.15,
    w_mom_3m: float = 0.35,
    w_mom_6m: float = 0.35,
    w_mom_12m: float = 0.15,
    reverse_1m: bool = False,
) -> pd.DataFrame:
    """
    Build full cross-sectional factor table for stocks.
    Returns DataFrame sorted descending by composite_score.
    Columns: individual factors + composite_score
    """
    # ── Technical factors ──────────────────────────────────────────────────────
    mom_1m  = T.momentum_1m(prices, reverse=reverse_1m)
    mom_3m  = T.momentum_3m(prices)
    mom_6m  = T.momentum_6m(prices)
    mom_12m = T.momentum_12m(prices)
    vol_20d = T.volatility_20d(returns)
    turnover_avg = T.avg_turnover_20d(turnover)
    vol_ratio = T.volume_ratio(volume)

    # Combined momentum z-score
    all_idx = prices.columns
    z_mom = (
        w_mom_1m  * zscore(mom_1m.reindex(all_idx).fillna(0))  +
        w_mom_3m  * zscore(mom_3m.reindex(all_idx).fillna(0))  +
        w_mom_6m  * zscore(mom_6m.reindex(all_idx).fillna(0))  +
        w_mom_12m * zscore(mom_12m.reindex(all_idx).fillna(0))
    )

    z_vol       = -zscore(vol_20d.reindex(all_idx).fillna(vol_20d.mean()))  # inverted
    z_liquidity =  zscore(turnover_avg.reindex(all_idx).fillna(0))

    # ── Fundamental factors ────────────────────────────────────────────────────
    z_value   = F.value_composite(fundamental_df).reindex(all_idx).fillna(0)
    z_quality = F.quality_composite(fundamental_df).reindex(all_idx).fillna(0)
    z_growth  = F.growth_composite(fundamental_df).reindex(all_idx).fillna(0)

    # ── Fund flow factors ──────────────────────────────────────────────────────
    z_fund_flow = FF.fund_flow_composite(fund_flow_df).reindex(all_idx).fillna(0)

    # ── Composite ─────────────────────────────────────────────────────────────
    weights = np.array([w_momentum, w_value, w_quality, w_growth, w_fund_flow, w_liquidity])
    weights = weights / weights.sum()

    composite = (
        weights[0] * z_mom        +
        weights[1] * z_value      +
        weights[2] * z_quality    +
        weights[3] * z_growth     +
        weights[4] * z_fund_flow  +
        weights[5] * z_liquidity
    )

    df = pd.DataFrame({
        "Momentum 1M":   mom_1m,
        "Momentum 3M":   mom_3m,
        "Momentum 6M":   mom_6m,
        "Volatility 20D": vol_20d,
        "Turnover Avg":  turnover_avg,
        "Vol Ratio":     vol_ratio,
        "Z Momentum":    z_mom,
        "Z Value":       z_value,
        "Z Quality":     z_quality,
        "Z Growth":      z_growth,
        "Z Fund Flow":   z_fund_flow,
        "Z Liquidity":   z_liquidity,
        "Composite Score": composite,
    }, index=all_idx)

    return df.dropna(subset=["Composite Score"]).sort_values("Composite Score", ascending=False)


def apply_sentiment_overlay(
    factor_df: pd.DataFrame,
    sentiment_series: pd.Series,
    sentiment_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Adjust composite score with sentiment as a multiplicative overlay.

    new_score = base_score * (1 + sentiment_weight * z_sentiment)

    This keeps sentiment forward-looking and separable — the base score
    (used for backtesting) is unchanged; sentiment only shifts the final ranking.
    """
    df = factor_df.copy()
    z_sent = zscore(sentiment_series.reindex(df.index).fillna(0))
    df["Sentiment Score"] = sentiment_series.reindex(df.index)
    df["Z Sentiment"]     = z_sent
    df["Composite Score"] = df["Composite Score"] * (1 + sentiment_weight * z_sent)
    return df.sort_values("Composite Score", ascending=False)


def build_etf_factor_table(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    amount: pd.DataFrame,
    w_momentum:   float = 0.40,
    w_vol_trend:  float = 0.30,
    w_vol_risk:   float = 0.30,
) -> pd.DataFrame:
    """
    Simplified factor table for ETFs (no fundamental data available).
    Factors: 3M/6M momentum, volume trend, volatility risk.
    """
    mom_3m = T.momentum_3m(prices)
    mom_6m = T.momentum_6m(prices)
    vol_20d = T.volatility_20d(returns)
    vol_ratio = T.volume_ratio(volume)

    all_idx = prices.columns
    z_mom       = 0.5 * zscore(mom_3m.reindex(all_idx).fillna(0)) + \
                  0.5 * zscore(mom_6m.reindex(all_idx).fillna(0))
    z_vol_trend =  zscore(vol_ratio.reindex(all_idx).fillna(1))
    z_vol_risk  = -zscore(vol_20d.reindex(all_idx).fillna(vol_20d.mean()))

    weights = np.array([w_momentum, w_vol_trend, w_vol_risk])
    weights /= weights.sum()

    composite = weights[0] * z_mom + weights[1] * z_vol_trend + weights[2] * z_vol_risk

    df = pd.DataFrame({
        "Momentum 3M":   mom_3m,
        "Momentum 6M":   mom_6m,
        "Volatility 20D": vol_20d,
        "Vol Ratio":     vol_ratio,
        "Composite Score": composite,
    }, index=all_idx)

    return df.dropna(subset=["Composite Score"]).sort_values("Composite Score", ascending=False)
