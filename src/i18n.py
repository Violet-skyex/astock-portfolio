"""UI strings — Chinese and English."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "lang_toggle":        "EN",
        "app_title":          "A股智能组合推荐",
        "app_caption":        "沪深300 · 中证500 · 科创50 · A股ETF",

        # Sidebar sections
        "sidebar_title":      "参数设置",
        "alloc_section":      "资产配置",
        "etf_ratio_label":    "ETF占比",
        "stock_ratio_label":  "个股占比",
        "max_etfs_label":     "ETF最多持有数",
        "max_stocks_label":   "个股最多持有数",
        "freq_label":         "调仓频率",
        "freq_daily":         "日频",
        "freq_weekly":        "周频",
        "freq_biweekly":      "半月频",
        "freq_monthly":       "月频",
        "benchmark_label":    "基准指数",
        "backtest_years":     "回测年数",

        "factor_section":     "因子权重",
        "w_momentum":         "动量",
        "w_value":            "估值",
        "w_quality":          "质量",
        "w_growth":           "成长",
        "w_fund_flow":        "资金流",
        "w_liquidity":        "流动性",

        "optim_section":      "组合构建方式",
        "mode_factor":        "因子加权（默认）",
        "mode_equal":         "等权",
        "mode_rp":            "风险平价",

        "etf_filter":         "ETF筛选条件",
        "etf_min_aum":        "最小AUM（亿元）",
        "etf_min_vol":        "最小日均成交额（千万元）",

        "sentiment_section":  "情感分析",
        "use_sentiment":      "启用新闻情感（实时）",
        "sentiment_weight":   "情感权重",

        "run_btn":            "▶ 运行分析",

        # Tabs
        "tab_recommend":      "推荐组合",
        "tab_backtest":       "历史回测",
        "tab_factors":        "因子详情",
        "tab_sentiment":      "情感分析",

        # Portfolio table
        "col_weight":         "权重",
        "col_composite":      "综合得分",
        "col_momentum":       "动量",
        "col_value":          "估值",
        "col_quality":        "质量",
        "col_growth":         "成长",
        "col_fund_flow":      "资金流",
        "col_sentiment":      "情感得分",
        "cash_label":         "现金",

        # Backtest
        "bt_nav_label":       "净值曲线",
        "bt_vs_bench":        "vs 基准",
        "kpi_annual_ret":     "年化收益",
        "kpi_sharpe":         "夏普比率",
        "kpi_max_dd":         "最大回撤",
        "kpi_alpha":          "Alpha",
        "kpi_beta":           "Beta",

        "footer":             "仅供参考，不构成投资建议",
    },
    "en": {
        "lang_toggle":        "中文",
        "app_title":          "A-Share Smart Portfolio",
        "app_caption":        "HS300 · CSI500 · STAR50 · A-Share ETFs",

        "sidebar_title":      "Parameters",
        "alloc_section":      "Asset Allocation",
        "etf_ratio_label":    "ETF Weight",
        "stock_ratio_label":  "Stock Weight",
        "max_etfs_label":     "Max ETFs",
        "max_stocks_label":   "Max Stocks",
        "freq_label":         "Rebalance Frequency",
        "freq_daily":         "Daily",
        "freq_weekly":        "Weekly",
        "freq_biweekly":      "Bi-Weekly",
        "freq_monthly":       "Monthly",
        "benchmark_label":    "Benchmark",
        "backtest_years":     "Backtest Years",

        "factor_section":     "Factor Weights",
        "w_momentum":         "Momentum",
        "w_value":            "Value",
        "w_quality":          "Quality",
        "w_growth":           "Growth",
        "w_fund_flow":        "Fund Flow",
        "w_liquidity":        "Liquidity",

        "optim_section":      "Portfolio Construction",
        "mode_factor":        "Factor Score (Default)",
        "mode_equal":         "Equal Weight",
        "mode_rp":            "Risk Parity",

        "etf_filter":         "ETF Filters",
        "etf_min_aum":        "Min AUM (100M CNY)",
        "etf_min_vol":        "Min Avg Daily Vol (10M CNY)",

        "sentiment_section":  "Sentiment",
        "use_sentiment":      "Enable News Sentiment (live)",
        "sentiment_weight":   "Sentiment Weight",

        "run_btn":            "▶ Run Analysis",

        "tab_recommend":      "Recommendations",
        "tab_backtest":       "Backtest",
        "tab_factors":        "Factor Scores",
        "tab_sentiment":      "Sentiment",

        "col_weight":         "Weight",
        "col_composite":      "Composite Score",
        "col_momentum":       "Momentum",
        "col_value":          "Value",
        "col_quality":        "Quality",
        "col_growth":         "Growth",
        "col_fund_flow":      "Fund Flow",
        "col_sentiment":      "Sentiment",
        "cash_label":         "Cash",

        "bt_nav_label":       "NAV Curve",
        "bt_vs_bench":        "vs Benchmark",
        "kpi_annual_ret":     "Annual Return",
        "kpi_sharpe":         "Sharpe Ratio",
        "kpi_max_dd":         "Max Drawdown",
        "kpi_alpha":          "Alpha",
        "kpi_beta":           "Beta",

        "footer":             "For reference only. Not investment advice.",
    },
}


def get_strings(lang: str = "zh") -> dict[str, str]:
    return STRINGS.get(lang, STRINGS["zh"])
