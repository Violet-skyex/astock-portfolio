"""
Chinese financial news sentiment scoring via Baidu NLP API.

Flow:
  1. Scrape recent news headlines for each stock from East Money (东方财富).
  2. Batch-call Baidu NLP sentiment analysis endpoint.
  3. Aggregate per-stock: weighted mean score in [-1, +1].

This module is FORWARD-LOOKING ONLY — results are not stored historically
and are not used in backtesting. Sentiment is an overlay on the current
recommendation's composite score.

Score convention: +1 = strongly positive, -1 = strongly negative, 0 = neutral.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

from .config import BAIDU_API_KEY, BAIDU_SECRET_KEY

logger = logging.getLogger(__name__)

_BAIDU_TOKEN: str = ""
_TOKEN_EXPIRES: float = 0.0

# ── Baidu OAuth token ──────────────────────────────────────────────────────────

def _get_baidu_token() -> str:
    """Fetch (or return cached) Baidu API access token."""
    global _BAIDU_TOKEN, _TOKEN_EXPIRES
    if time.time() < _TOKEN_EXPIRES - 60 and _BAIDU_TOKEN:
        return _BAIDU_TOKEN

    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type":    "client_credentials",
        "client_id":     BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY,
    }
    resp = requests.post(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _BAIDU_TOKEN   = data["access_token"]
    _TOKEN_EXPIRES = time.time() + data.get("expires_in", 2592000)
    return _BAIDU_TOKEN


# ── News scraping ──────────────────────────────────────────────────────────────

def scrape_stock_news(ts_code: str, max_articles: int = 10) -> list[str]:
    """
    Scrape recent news headlines for a stock from East Money.
    Returns list of headline strings.

    TODO: identify the correct East Money API endpoint for individual stock news.
    Candidate: https://np-listapi.eastmoney.com/comm/web/getListInfo?...
    """
    symbol = ts_code.split(".")[0]
    url = (
        "https://np-listapi.eastmoney.com/comm/web/getListInfo"
        f"?client=web&type=1&mTypeAndCode=0%2C{symbol}"
        "&pageSize=20&pageIndex=1&callback=&_=0"
    )
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
        articles = data.get("data", {}).get("list", [])
        return [a.get("title", "") for a in articles[:max_articles] if a.get("title")]
    except Exception as e:
        logger.warning("News scrape failed for %s: %s", ts_code, e)
        return []


# ── Baidu NLP sentiment ────────────────────────────────────────────────────────

def score_texts_baidu(texts: list[str]) -> list[float]:
    """
    Score a list of texts using Baidu NLP sentiment API.
    Returns list of scores in [-1, +1].
    Baidu returns: sentiment 0=negative, 1=neutral, 2=positive + confidence.

    We map: positive → +confidence, negative → -confidence, neutral → 0.
    """
    if not texts or not BAIDU_API_KEY:
        return [0.0] * len(texts)

    token  = _get_baidu_token()
    url    = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={token}"
    scores = []

    for text in texts:
        try:
            resp = requests.post(
                url,
                json={"text": text[:512]},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            result = resp.json()
            items  = result.get("items", [{}])
            if not items:
                scores.append(0.0)
                continue
            item       = items[0]
            sentiment  = item.get("sentiment", 1)    # 0=neg, 1=neu, 2=pos
            confidence = item.get("confidence", 0.5)
            if sentiment == 2:
                scores.append(confidence)
            elif sentiment == 0:
                scores.append(-confidence)
            else:
                scores.append(0.0)
        except Exception as e:
            logger.warning("Baidu NLP failed: %s", e)
            scores.append(0.0)

    return scores


# ── Per-stock aggregation ──────────────────────────────────────────────────────

def score_stock(ts_code: str, max_articles: int = 10) -> dict:
    """
    Scrape + score a single stock. Returns dict with keys:
      ts_code, sentiment_score, news_count, positive_ratio
    """
    headlines = scrape_stock_news(ts_code, max_articles)
    if not headlines:
        return {"ts_code": ts_code, "sentiment_score": 0.0,
                "news_count": 0, "positive_ratio": 0.0}

    scores = score_texts_baidu(headlines)
    positive_ratio = sum(1 for s in scores if s > 0.05) / len(scores)
    return {
        "ts_code":        ts_code,
        "sentiment_score": float(pd.Series(scores).mean()),
        "news_count":      len(headlines),
        "positive_ratio":  positive_ratio,
    }


def run_sentiment(
    ts_codes: list[str],
    max_workers: int = 5,
    max_articles: int = 10,
) -> pd.DataFrame:
    """
    Score a list of stocks in parallel.
    Returns DataFrame indexed by ts_code with sentiment metrics.
    """
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(score_stock, code, max_articles): code for code in ts_codes}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                code = futures[fut]
                logger.warning("Sentiment failed for %s: %s", code, e)
                rows.append({"ts_code": code, "sentiment_score": 0.0,
                             "news_count": 0, "positive_ratio": 0.0})

    df = pd.DataFrame(rows).set_index("ts_code")
    return df.reindex(ts_codes).fillna({"sentiment_score": 0.0,
                                         "news_count": 0,
                                         "positive_ratio": 0.0})
