"""
A股智能组合推荐 — Streamlit App
A-Share Smart Portfolio — bilingual (ZH/EN)

Separate ETF and stock recommendations, factor-driven scoring,
optional news sentiment overlay, rolling backtest with cost simulation.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.i18n import get_strings
from src.config import (
    BENCHMARK_INDICES, DEFAULT_MIN_WEIGHT, DEFAULT_MAX_WEIGHT,
    ETF_MIN_AUM_CNY_M, ETF_MIN_VOL_CNY_M, TRADING_DAYS_YEAR,
)
from src.universe import get_stock_universe, fetch_etf_universe, filter_etf_universe
from src.data.prices import fetch_prices_auto, compute_returns, fetch_index_daily
from src.data.fundamentals import build_fundamental_table
from src.data.fund_flow import build_fund_flow_table
from src.scorer import build_stock_factor_table, build_etf_factor_table, apply_sentiment_overlay
from src.optimizer import build_portfolio, portfolio_metrics
from src.backtest import (
    run_stock_backtest, run_etf_backtest, combine_backtests,
    rolling_metrics, BacktestResult,
)


# ── Streamlit-cached data loaders (1-hour TTL for prices, 6h for fundamentals) ─

@st.cache_data(ttl=3600, show_spinner=False)
def _load_stock_universe():
    return get_stock_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_etf_universe():
    return fetch_etf_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(ts_codes: tuple, start: str, end: str, is_etf: bool):
    return fetch_prices_auto(list(ts_codes), start, end, is_etf=is_etf)


@st.cache_data(ttl=21600, show_spinner=False)
def _load_fundamentals(ts_codes: tuple):
    return build_fundamental_table(list(ts_codes))


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fund_flow(ts_codes: tuple):
    return build_fund_flow_table(list(ts_codes))


@st.cache_data(ttl=3600, show_spinner=False)
def _load_benchmark(code: str, start: str, end: str):
    return fetch_index_daily(code, start, end)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A股组合推荐",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

t = get_strings(st.session_state.lang)

# ── Language toggle ────────────────────────────────────────────────────────────
_, lang_col = st.columns([12, 1])
with lang_col:
    if st.button(t["lang_toggle"], key="lang_btn"):
        st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
        st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title(t["sidebar_title"])

# Asset allocation
st.sidebar.subheader(t["alloc_section"])
etf_ratio   = st.sidebar.slider(t["etf_ratio_label"],   0, 100, 30, 5) / 100
stock_ratio = 1.0 - etf_ratio
st.sidebar.caption(f"{t['stock_ratio_label']}: {stock_ratio:.0%}")

max_etfs    = st.sidebar.slider(t["max_etfs_label"],   1, 5,  3)
max_stocks  = st.sidebar.slider(t["max_stocks_label"], 1, 10, 7)

st.sidebar.markdown("---")

# Rebalancing frequency & backtest
st.sidebar.subheader(t["freq_label"])
freq_map = {
    t["freq_daily"]:     "daily",
    t["freq_weekly"]:    "weekly",
    t["freq_biweekly"]:  "biweekly",
    t["freq_monthly"]:   "monthly",
}
freq_label  = st.sidebar.radio(t["freq_label"], list(freq_map.keys()), index=3,
                                label_visibility="collapsed")
freq        = freq_map[freq_label]
backtest_yrs = st.sidebar.slider(t["backtest_years"], 1, 5, 3)
benchmark_name = st.sidebar.selectbox(t["benchmark_label"], list(BENCHMARK_INDICES.keys()))
benchmark_code = BENCHMARK_INDICES[benchmark_name]

st.sidebar.markdown("---")

# Factor weights
st.sidebar.subheader(t["factor_section"])
w_momentum  = st.sidebar.slider(t["w_momentum"],  0.0, 1.0, 0.20, 0.05)
w_value     = st.sidebar.slider(t["w_value"],     0.0, 1.0, 0.20, 0.05)
w_quality   = st.sidebar.slider(t["w_quality"],   0.0, 1.0, 0.15, 0.05)
w_growth    = st.sidebar.slider(t["w_growth"],    0.0, 1.0, 0.15, 0.05)
w_fund_flow = st.sidebar.slider(t["w_fund_flow"], 0.0, 1.0, 0.15, 0.05)
w_liquidity = st.sidebar.slider(t["w_liquidity"], 0.0, 1.0, 0.15, 0.05)

st.sidebar.markdown("---")

# Portfolio construction mode
st.sidebar.subheader(t["optim_section"])
mode_map = {
    t["mode_factor"]: "factor_score",
    t["mode_equal"]:  "equal_weight",
    t["mode_rp"]:     "risk_parity",
}
mode_label = st.sidebar.radio(t["optim_section"], list(mode_map.keys()), index=0,
                               label_visibility="collapsed")
mode = mode_map[mode_label]

st.sidebar.markdown("---")

# ETF filters
st.sidebar.subheader(t["etf_filter"])
min_aum = st.sidebar.slider(t["etf_min_aum"], 0.0, 2000.0, float(ETF_MIN_AUM_CNY_M), 50.0)
min_vol = st.sidebar.slider(t["etf_min_vol"], 0.0,  500.0, float(ETF_MIN_VOL_CNY_M / 10), 5.0)

st.sidebar.markdown("---")

# Sentiment
st.sidebar.subheader(t["sentiment_section"])
use_sentiment    = st.sidebar.checkbox(t["use_sentiment"], value=False)
w_sentiment      = 0.0
if use_sentiment:
    w_sentiment  = st.sidebar.slider(t["sentiment_weight"], 0.0, 0.30, 0.15, 0.05)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(t["app_title"])
st.caption(
    f"{t['app_caption']} · {datetime.date.today().strftime('%Y-%m-%d')} · "
    f"{freq_label} · {benchmark_name}"
)

run_btn = st.button(t["run_btn"], type="primary")
if not run_btn:
    st.info("← 在左侧设置参数后点击「▶ 运行分析」" if st.session_state.lang == "zh"
            else "← Configure parameters in the sidebar then click ▶ Run Analysis")
    st.stop()

# ── Data loading ───────────────────────────────────────────────────────────────
st.markdown("---")
_zh = st.session_state.lang == "zh"
progress = st.progress(0, text="加载股票池..." if _zh else "Loading universe...")

import datetime as _dt
end_date   = _dt.date.today().strftime("%Y-%m-%d")
start_date = (_dt.date.today() - _dt.timedelta(days=backtest_yrs * 365 + 60)).strftime("%Y-%m-%d")

# 1. Universe
try:
    stock_universe = _load_stock_universe()
    stock_codes    = tuple(stock_universe["ts_code"].tolist())
except Exception as e:
    st.error(f"{'股票池加载失败' if _zh else 'Failed to load stock universe'}: {e}")
    st.stop()

progress.progress(10, text="加载ETF池..." if _zh else "Loading ETF universe...")

try:
    etf_df      = _load_etf_universe()
    etf_codes   = tuple(filter_etf_universe(etf_df, min_amount_bn=min_vol / 100))
except Exception as e:
    st.error(f"{'ETF池加载失败' if _zh else 'Failed to load ETF universe'}: {e}")
    st.stop()

progress.progress(15, text=f"下载 {len(stock_codes)} 只股票行情..." if _zh
                  else f"Fetching prices for {len(stock_codes)} stocks...")

# 2. Prices — stocks
try:
    stock_prices, stock_volume, stock_amount, stock_turnover = _load_prices(
        stock_codes, start_date, end_date, is_etf=False
    )
    stock_returns = compute_returns(stock_prices)
except Exception as e:
    st.error(f"{'股票行情下载失败' if _zh else 'Stock price fetch failed'}: {e}")
    st.stop()

progress.progress(55, text=f"下载 {len(etf_codes)} 只ETF行情..." if _zh
                  else f"Fetching prices for {len(etf_codes)} ETFs...")

# 3. Prices — ETFs
try:
    etf_prices, etf_volume, etf_amount, etf_turnover = _load_prices(
        etf_codes, start_date, end_date, is_etf=True
    )
    etf_returns = compute_returns(etf_prices)
except Exception as e:
    st.error(f"{'ETF行情下载失败' if _zh else 'ETF price fetch failed'}: {e}")
    st.stop()

progress.progress(65, text="加载财务数据..." if _zh else "Loading fundamental data...")

# 4. Fundamentals
try:
    fundamental_df = _load_fundamentals(stock_codes)
except Exception as e:
    st.warning(f"{'财务数据加载失败，将跳过基本面因子' if _zh else 'Fundamentals failed — skipping'}: {e}")
    fundamental_df = pd.DataFrame(index=list(stock_codes))

progress.progress(75, text="加载资金流数据..." if _zh else "Loading fund flow data...")

# 5. Fund flow
try:
    fund_flow_df = _load_fund_flow(stock_codes)
except Exception as e:
    st.warning(f"{'资金流数据加载失败，将跳过资金流因子' if _zh else 'Fund flow failed — skipping'}: {e}")
    fund_flow_df = pd.DataFrame(index=list(stock_codes))

progress.progress(85, text="加载基准指数..." if _zh else "Loading benchmark...")

# 6. Benchmark
try:
    benchmark_series = _load_benchmark(benchmark_code, start_date, end_date)
except Exception as e:
    st.warning(f"{'基准指数加载失败' if _zh else 'Benchmark failed'}: {e}")
    benchmark_series = pd.Series(dtype=float)

progress.progress(100, text="数据加载完成" if _zh else "Data loaded")
progress.empty()

# ── Factor scoring & portfolio construction ────────────────────────────────────
with st.spinner("因子打分中..." if _zh else "Scoring factors..."):
    stock_factor_df = build_stock_factor_table(
        prices=stock_prices, volume=stock_volume,
        turnover=stock_turnover, returns=stock_returns,
        fundamental_df=fundamental_df, fund_flow_df=fund_flow_df,
        w_momentum=w_momentum, w_value=w_value, w_quality=w_quality,
        w_growth=w_growth, w_fund_flow=w_fund_flow, w_liquidity=w_liquidity,
    )
    etf_factor_df = build_etf_factor_table(
        prices=etf_prices, volume=etf_volume,
        returns=etf_returns, amount=etf_amount,
    )

# Sentiment overlay (if enabled) — runs on top candidates only
if use_sentiment and w_sentiment > 0:
    from src.sentiment import run_sentiment
    top_candidates = stock_factor_df.head(max_stocks * 3).index.tolist()
    with st.spinner("情感分析中..." if _zh else "Running sentiment..."):
        sent_df = run_sentiment(top_candidates, max_workers=3)
    stock_factor_df = apply_sentiment_overlay(
        stock_factor_df, sent_df["sentiment_score"], w_sentiment
    )

# Build ETF and stock portfolios separately, then combine
etf_port_df, etf_cash = build_portfolio(
    etf_factor_df, etf_returns, mode=mode,
    top_n=len(etf_factor_df), max_positions=max_etfs,
)
stock_port_df, stock_cash = build_portfolio(
    stock_factor_df, stock_returns, mode=mode,
    top_n=min(len(stock_factor_df), max_stocks * 3), max_positions=max_stocks,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_rec, tab_bt, tab_fac, tab_sent = st.tabs([
    t["tab_recommend"],
    t["tab_backtest"],
    t["tab_factors"],
    t["tab_sentiment"],
])

# ── Recommendation tab ─────────────────────────────────────────────────────────
with tab_rec:
    col_etf, col_stock = st.columns(2)

    with col_etf:
        st.subheader(f"ETF 推荐 ({etf_ratio:.0%})" if _zh else f"ETF Picks ({etf_ratio:.0%})")
        if etf_port_df.empty:
            st.info("无满足条件的ETF" if _zh else "No qualifying ETFs.")
        else:
            show_cols = [c for c in ["Weight", "Momentum 3M", "Momentum 6M",
                                      "Volatility 20D", "Composite Score"] if c in etf_port_df.columns]
            st.dataframe(
                etf_port_df[show_cols].style
                    .format({c: "{:.1%}" for c in ["Weight", "Momentum 3M", "Momentum 6M"]
                              if c in show_cols})
                    .format({"Volatility 20D": "{:.1%}", "Composite Score": "{:.2f}"}
                             if "Composite Score" in show_cols else {})
                    .background_gradient(subset=["Weight"], cmap="Blues"),
                use_container_width=True,
            )
            if etf_cash > 0.01:
                st.caption(f"{'现金': <6} {etf_cash:.1%}" if _zh else f"Cash: {etf_cash:.1%}")

    with col_stock:
        st.subheader(f"个股推荐 ({stock_ratio:.0%})" if _zh else f"Stock Picks ({stock_ratio:.0%})")
        if stock_port_df.empty:
            st.info("无满足条件的个股" if _zh else "No qualifying stocks.")
        else:
            show_cols = [c for c in ["Weight", "Z Momentum", "Z Value", "Z Quality",
                                      "Z Growth", "Z Fund Flow", "Composite Score",
                                      "Sentiment Score"]
                          if c in stock_port_df.columns]
            fmt = {c: "{:.1%}" for c in ["Weight"] if c in show_cols}
            fmt.update({c: "{:.2f}" for c in show_cols if c not in fmt})
            st.dataframe(
                stock_port_df[show_cols].style
                    .format(fmt)
                    .background_gradient(subset=["Weight"], cmap="Blues"),
                use_container_width=True,
            )
            if stock_cash > 0.01:
                st.caption(f"{'现金': <6} {stock_cash:.1%}" if _zh else f"Cash: {stock_cash:.1%}")

    # Combined allocation pie
    st.subheader("资产配置" if _zh else "Allocation")
    pie_names, pie_vals = [], []
    for tk, row in etf_port_df.iterrows() if not etf_port_df.empty else []:
        pie_names.append(f"ETF {tk}")
        pie_vals.append(row["Weight"] * etf_ratio)
    for tk, row in stock_port_df.iterrows() if not stock_port_df.empty else []:
        pie_names.append(tk)
        pie_vals.append(row["Weight"] * stock_ratio)
    total_invested = sum(pie_vals)
    if total_invested < 0.999:
        pie_names.append("现金" if _zh else "Cash")
        pie_vals.append(1.0 - total_invested)

    fig_pie = px.pie(
        names=pie_names, values=pie_vals, hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Backtest tab ───────────────────────────────────────────────────────────────
with tab_bt:
    st.subheader(
        f"{t['bt_nav_label']} — {backtest_yrs}年回测 · {freq_label}调仓"
        if _zh else
        f"{t['bt_nav_label']} — {backtest_yrs}Y · {freq_label}"
    )
    st.caption(
        "注：回测仅使用技术因子（动量/波动/换手），不含基本面/资金流（无历史数据）。当前推荐使用全因子。"
        if _zh else
        "Note: backtest uses technical factors only (momentum/vol/turnover). "
        "Fundamental & fund-flow factors are forward-looking only."
    )

    if benchmark_series.empty or stock_prices.empty:
        st.warning("数据不足，无法运行回测。" if _zh else "Insufficient data for backtest.")
    else:
        with st.spinner("回测计算中..." if _zh else "Running backtest..."):
            try:
                stock_bt = run_stock_backtest(
                    prices=stock_prices, volume=stock_volume,
                    turnover=stock_turnover, returns=stock_returns,
                    benchmark=benchmark_series,
                    freq=freq, lookback_days=126,
                    max_positions=max_stocks, mode=mode,
                )
                etf_bt = run_etf_backtest(
                    prices=etf_prices, volume=etf_volume,
                    turnover=etf_turnover, returns=etf_returns,
                    benchmark=benchmark_series,
                    freq=freq, lookback_days=126,
                    max_positions=max_etfs, mode=mode,
                )
                combined_bt = combine_backtests(stock_bt, etf_bt, stock_weight=stock_ratio, etf_weight=etf_ratio)
                bt_ok = True
            except Exception as e:
                st.error(f"回测失败: {e}" if _zh else f"Backtest failed: {e}")
                bt_ok = False

        if bt_ok:
            # ── KPI row ────────────────────────────────────────────────────────
            m = combined_bt.metrics
            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            k1.metric(t["kpi_annual_ret"],
                      f"{m['年化收益']:.1%}",
                      f"{m['超额收益']:+.1%} vs {benchmark_name}")
            k2.metric(t["kpi_sharpe"],   f"{m['夏普比率']:.2f}")
            k3.metric(t["kpi_max_dd"],   f"{m['最大回撤']:.1%}")
            k4.metric(t["kpi_alpha"],    f"{m['Alpha']:.2%}")
            k5.metric(t["kpi_beta"],     f"{m['Beta']:.2f}")
            k6.metric("卡玛比率" if _zh else "Calmar", f"{m['卡玛比率']:.2f}")
            k7.metric("胜率"   if _zh else "Win Rate", f"{m['胜率']:.1%}")

            st.markdown("---")

            # ── NAV curve ─────────────────────────────────────────────────────
            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(
                x=combined_bt.nav.index, y=combined_bt.nav.values,
                name="综合组合" if _zh else "Combined Portfolio",
                line=dict(color="#2563EB", width=2.5),
            ))
            fig_nav.add_trace(go.Scatter(
                x=stock_bt.nav.index, y=stock_bt.nav.values,
                name="个股子组合" if _zh else "Stock Sub-portfolio",
                line=dict(color="#7B61FF", width=1.5, dash="dot"),
            ))
            fig_nav.add_trace(go.Scatter(
                x=etf_bt.nav.index, y=etf_bt.nav.values,
                name="ETF子组合" if _zh else "ETF Sub-portfolio",
                line=dict(color="#10B981", width=1.5, dash="dot"),
            ))
            fig_nav.add_trace(go.Scatter(
                x=combined_bt.benchmark_nav.index,
                y=combined_bt.benchmark_nav.values,
                name=benchmark_name,
                line=dict(color="#F59E0B", width=1.5, dash="dash"),
            ))
            fig_nav.update_layout(
                xaxis_title="日期" if _zh else "Date",
                yaxis_title="净值" if _zh else "NAV",
                legend=dict(orientation="h", y=1.05),
                margin=dict(t=20, b=30), height=420,
            )
            st.plotly_chart(fig_nav, use_container_width=True)

            # ── Rolling Sharpe & Drawdown ──────────────────────────────────────
            roll = rolling_metrics(combined_bt.port_returns, window=63)
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**滚动夏普（63日）**" if _zh else "**Rolling Sharpe (63D)**")
                fig_rs = go.Figure(go.Scatter(
                    x=roll.index, y=roll["Rolling Sharpe"],
                    fill="tozeroy", line=dict(color="#2563EB", width=1.5),
                ))
                fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_rs.update_layout(height=250, margin=dict(t=10, b=20))
                st.plotly_chart(fig_rs, use_container_width=True)

            with col_r2:
                st.markdown("**滚动回撤（63日）**" if _zh else "**Rolling Drawdown (63D)**")
                fig_rd = go.Figure(go.Scatter(
                    x=roll.index, y=roll["Rolling Drawdown"],
                    fill="tozeroy", line=dict(color="#EF4444", width=1.5),
                ))
                fig_rd.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_rd.update_layout(height=250, margin=dict(t=10, b=20))
                st.plotly_chart(fig_rd, use_container_width=True)

            # ── Turnover & cost ────────────────────────────────────────────────
            st.markdown("---")
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                st.markdown("**调仓换手率**" if _zh else "**Rebalance Turnover**")
                if not stock_bt.turnover.empty:
                    fig_to = go.Figure(go.Bar(
                        x=stock_bt.turnover.index, y=stock_bt.turnover.values,
                        marker_color="#94A3B8",
                    ))
                    fig_to.update_layout(height=200, margin=dict(t=10, b=20))
                    st.plotly_chart(fig_to, use_container_width=True)

            with col_t2:
                st.markdown("**分项指标**" if _zh else "**Sub-portfolio Metrics**")
                sub_m = pd.DataFrame({
                    "指标" if _zh else "Metric": ["年化收益", "夏普", "最大回撤", "超额"] if _zh
                                                else ["Ann.Ret", "Sharpe", "MaxDD", "Alpha"],
                    "个股" if _zh else "Stocks": [
                        f"{stock_bt.metrics['年化收益']:.1%}",
                        f"{stock_bt.metrics['夏普比率']:.2f}",
                        f"{stock_bt.metrics['最大回撤']:.1%}",
                        f"{stock_bt.metrics['Alpha']:.2%}",
                    ],
                    "ETF": [
                        f"{etf_bt.metrics['年化收益']:.1%}",
                        f"{etf_bt.metrics['夏普比率']:.2f}",
                        f"{etf_bt.metrics['最大回撤']:.1%}",
                        f"{etf_bt.metrics['Alpha']:.2%}",
                    ],
                })
                st.dataframe(sub_m, use_container_width=True, hide_index=True)

            # ── Historical holdings heatmap ────────────────────────────────────
            if not stock_bt.holdings.empty:
                st.markdown("---")
                st.markdown("**持仓历史热力图（个股，Top 20 by 出现次数）**"
                            if _zh else "**Holdings History Heatmap — Stocks, Top 20 by frequency**")
                hold_top = stock_bt.holdings
                freq_count = (hold_top > 0).sum().sort_values(ascending=False)
                top_tickers = freq_count.head(20).index
                hold_disp = hold_top[top_tickers].T

                fig_hm = px.imshow(
                    hold_disp.values,
                    x=[str(d.date()) for d in hold_disp.columns],
                    y=hold_disp.index.tolist(),
                    color_continuous_scale="Blues",
                    aspect="auto",
                )
                fig_hm.update_layout(height=max(300, len(top_tickers) * 18),
                                      margin=dict(t=10, b=10))
                st.plotly_chart(fig_hm, use_container_width=True)

# ── Factor scores tab ──────────────────────────────────────────────────────────
with tab_fac:
    st.subheader(t["tab_factors"])
    fac_col1, fac_col2 = st.columns([2, 1])

    with fac_col1:
        st.markdown("**个股 Top 30**" if _zh else "**Stock Top 30**")
        z_cols = [c for c in stock_factor_df.columns if c.startswith("Z ")]
        disp_cols = z_cols + ["Composite Score"]
        if not stock_factor_df.empty and disp_cols:
            st.dataframe(
                stock_factor_df[disp_cols].head(30).style
                    .format("{:.2f}")
                    .background_gradient(subset=["Composite Score"], cmap="RdYlGn"),
                use_container_width=True, height=600,
            )

    with fac_col2:
        st.markdown("**因子权重**" if _zh else "**Factor Weights**")
        total_w = w_momentum + w_value + w_quality + w_growth + w_fund_flow + w_liquidity
        fw_df = pd.DataFrame({
            "因子" if _zh else "Factor": [
                t["w_momentum"], t["w_value"], t["w_quality"],
                t["w_growth"], t["w_fund_flow"], t["w_liquidity"],
            ],
            "归一化权重" if _zh else "Weight": [
                w_momentum / total_w, w_value / total_w, w_quality / total_w,
                w_growth / total_w,  w_fund_flow / total_w, w_liquidity / total_w,
            ],
        })
        st.dataframe(
            fw_df.style.format({"归一化权重" if _zh else "Weight": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

        # Bar chart of factor weights
        fig_fw = px.bar(fw_df, x="归一化权重" if _zh else "Weight",
                        y="因子" if _zh else "Factor", orientation="h",
                        color="归一化权重" if _zh else "Weight",
                        color_continuous_scale="Blues")
        fig_fw.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_fw, use_container_width=True)

# ── Sentiment tab ──────────────────────────────────────────────────────────────
with tab_sent:
    if not use_sentiment:
        st.info(
            "请在左侧勾选「启用新闻情感」后重新运行。" if _zh else
            "Enable sentiment in the sidebar and re-run."
        )
    elif "sent_df" not in dir():
        st.warning("情感数据未加载（可能缺少百度API密钥）。" if _zh else
                   "Sentiment data not loaded — check Baidu API credentials in .env.")
    else:
        st.subheader(t["tab_sentiment"])
        st.caption(
            "情感得分来自东方财富新闻标题，通过百度NLP分析。仅用于当前推荐，不影响回测。"
            if _zh else
            "Scores derived from East Money news headlines via Baidu NLP. "
            "Forward-looking only — not used in backtest."
        )

        # ── Score bar chart ────────────────────────────────────────────────────
        if not sent_df.empty:
            sent_display = sent_df[sent_df["news_count"] > 0].copy()
            sent_display = sent_display.sort_values("sentiment_score", ascending=True)
            sent_display["color"] = sent_display["sentiment_score"].apply(
                lambda x: "pos" if x > 0.05 else ("neg" if x < -0.05 else "neu")
            )
            color_map = {"pos": "#10B981", "neg": "#EF4444", "neu": "#94A3B8"}

            fig_sent = go.Figure()
            for grp, color in color_map.items():
                sub = sent_display[sent_display["color"] == grp]
                if sub.empty:
                    continue
                fig_sent.add_trace(go.Bar(
                    x=sub["sentiment_score"],
                    y=sub.index,
                    orientation="h",
                    marker_color=color,
                    name={"pos": "正面", "neg": "负面", "neu": "中性"}.get(grp, grp)
                    if _zh else grp,
                ))
            fig_sent.add_vline(x=0, line_dash="dash", opacity=0.4)
            fig_sent.update_layout(
                barmode="overlay",
                height=max(300, len(sent_display) * 22),
                margin=dict(t=20, b=20),
                legend=dict(orientation="h", y=1.05),
                xaxis_title="情感得分" if _zh else "Sentiment Score",
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        # ── Scatter: composite score vs sentiment ──────────────────────────────
        st.markdown("---")
        st.markdown("**因子得分 vs 情感得分**" if _zh else "**Factor Score vs Sentiment Score**")
        if not sent_df.empty and not stock_factor_df.empty:
            scat = stock_factor_df[["Composite Score"]].join(
                sent_df[["sentiment_score", "news_count", "positive_ratio"]], how="inner"
            ).reset_index()
            scat.columns = ["Ticker", "Composite", "Sentiment", "News Count", "Pos Ratio"]
            scat["In Portfolio"] = scat["Ticker"].isin(
                stock_port_df.index if not stock_port_df.empty else []
            )
            fig_sc = px.scatter(
                scat, x="Composite", y="Sentiment", text="Ticker",
                size="News Count", size_max=16,
                color="In Portfolio",
                color_discrete_map={True: "#2563EB", False: "#CBD5E1"},
                hover_data=["Pos Ratio"],
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.add_hline(y=0, line_dash="dash", opacity=0.3)
            fig_sc.add_vline(x=0, line_dash="dash", opacity=0.3)
            fig_sc.update_layout(height=480, margin=dict(t=30, b=20))
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Per-stock sentiment detail ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("**推荐股票情感明细**" if _zh else "**Sentiment Detail — Recommended Stocks**")
        if not stock_port_df.empty and not sent_df.empty:
            for ticker in stock_port_df.index:
                if ticker not in sent_df.index:
                    continue
                row = sent_df.loc[ticker]
                score = row["sentiment_score"]
                ico   = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
                header = (
                    f"**{ticker}** {ico} 情感: {score:+.3f} | "
                    f"新闻: {int(row['news_count'])}条 | "
                    f"正面比: {row['positive_ratio']:.0%}"
                )
                with st.expander(header, expanded=False):
                    st.caption("（新闻标题明细功能预留，需在 sentiment.py 中存储原始标题）"
                               if _zh else
                               "(Headline detail reserved — store raw headlines in sentiment.py)")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"{t['footer']} · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
