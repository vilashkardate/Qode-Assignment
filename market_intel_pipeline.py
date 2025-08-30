
import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

#!/usr/bin/env python3
"""
market_intel_pipeline.py
---------------------------------
Real-time market intelligence pipeline for Indian stock market tweets (X/Twitter)
without using paid APIs.

Features
- Collection via snscrape (no paid API), with rate limiting & retries
- Cleansing & normalization (Unicode-safe; Indian languages)
- Deduplication
- Storage to Parquet (partitioned by date)
- Text-to-signal using TF-IDF + lightweight lexicon features
- Signal aggregation with bootstrap confidence intervals
- Memory-efficient plotting with strategic downsampling
- Concurrency for collection across multiple hashtags
- Production-grade logging, error handling, and CLI

DISCLAIMER
Scraping X/Twitter may be restricted by their Terms of Service (ToS). This code uses the
community tool "snscrape" that queries public content without login. Ensure your use
complies with local laws and the platform ToS. Use responsibly and rate-limit politely.

Usage
-----
# 1) Install dependencies (see requirements.txt)
# 2) Example (collect last 24h, 2000+ tweets total across hashtags):
python market_intel_pipeline.py collect --min-total 2000 --hours 24

# 3) Clean/process/store to Parquet
python market_intel_pipeline.py process --input-jsonl data/raw/*.jsonl --out data/curated

# 4) Build signals
python market_intel_pipeline.py analyze --parquet data/curated --out data/signals

# 5) Plot (memory-efficient)
python market_intel_pipeline.py plot --signals data/signals/signals.parquet --out plots

# Or do all in one go:
python market_intel_pipeline.py all --min-total 2000 --hours 24

Notes
-----
- Parquet is written under data/curated/ingest_date=YYYY-MM-DD/*.parquet by default.
- Signals output: data/signals/signals.parquet and CSV summary.
"""

import argparse
import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import logging.handlers
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

# Third-party
import pandas as pd
import numpy as np

# snscrape (pip install snscrape)
try:
    import snscrape.modules.twitter as sntwitter
except Exception as e:
    sntwitter = None

# Parquet
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
except Exception as e:
    # Pandas can write Parquet if pyarrow is installed; we check at write-time.
    pass

# ML/vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Plotting
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ------------------------------ Logging ------------------------------------

def setup_logging(log_dir: Path = Path("logs")) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(log_dir / "pipeline.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)


# ------------------------------ Utilities ----------------------------------

HASHTAGS_DEFAULT = ["#nifty50", "#sensex", "#intraday", "#banknifty"]

BULLISH_WORDS = {
    "bull", "bullish", "buy", "long", "rally", "breakout", "accumulate", "uptrend",
    "green", "support", "break above", "higher high", "ath", "bounce"
}
BEARISH_WORDS = {
    "bear", "bearish", "sell", "short", "dump", "breakdown", "distribution", "downtrend",
    "red", "resistance", "break below", "lower low", "crash", "falling"
}

MENTION_RE = re.compile(r"@([A-Za-z0-9_]{1,15})")
HASHTAG_RE = re.compile(r"#(\w+)")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    """
    Unicode normalize text (NFKC), strip control chars, standardize whitespace.
    Preserves Indian scripts; does not lowercase to keep capitalization features optional.
    """
    text = unicodedata.normalize("NFKC", text or "")
    # Replace unusual whitespace with single space
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    # Remove zero-width chars
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text


def extract_entities(text: str) -> Tuple[List[str], List[str]]:
    mentions = MENTION_RE.findall(text)
    hashtags = HASHTAG_RE.findall(text)
    return mentions, hashtags


def ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet writes. Install with: pip install pyarrow")


# ------------------------------ Data Classes -------------------------------

@dataclasses.dataclass
class TweetRecord:
    tweet_id: str
    url: str
    username: str
    displayname: str
    timestamp_utc: str  # ISO8601
    content: str
    like_count: int
    retweet_count: int
    reply_count: int
    quote_count: int
    hashtags: List[str]
    mentions: List[str]
    scraped_at_utc: str
    source_hashtag: str


# ------------------------------ Collection ---------------------------------

def _scrape_hashtag(
    hashtag: str,
    since: datetime,
    until: datetime,
    max_items: int,
    sleep_every: int = 200,
    sleep_seconds: float = 5.0,
    jitter: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Scrape tweets for a single hashtag using snscrape Search.
    Returns list of dicts.
    """
    if sntwitter is None:
        raise RuntimeError("snscrape is not installed. Install with: pip install snscrape")

    query = f"{hashtag} since:{since.strftime('%Y-%m-%d_%H:%M:%S_UTC')} until:{until.strftime('%Y-%m-%d_%H:%M:%S_UTC')}"
    logging.info(f"Scraping {hashtag} | window: {since} -> {until} | max_items={max_items}")
    items = []
    count = 0
    retries = 0
    max_retries = 5

    while True:
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                # tweet.date is timezone-aware
                if tweet.date < since or tweet.date >= until:
                    continue
                record = {
                    "tweet_id": str(tweet.id),
                    "url": tweet.url,
                    "username": tweet.user.username if tweet.user else "",
                    "displayname": tweet.user.displayname if tweet.user else "",
                    "timestamp_utc": tweet.date.astimezone(timezone.utc).isoformat(),
                    "content": tweet.rawContent or tweet.content or "",
                    "like_count": getattr(tweet, "likeCount", 0) or 0,
                    "retweet_count": getattr(tweet, "retweetCount", 0) or 0,
                    "reply_count": getattr(tweet, "replyCount", 0) or 0,
                    "quote_count": getattr(tweet, "quoteCount", 0) or 0,
                    "scraped_at_utc": utc_now().isoformat(),
                    "source_hashtag": hashtag,
                }
                # Extract entities
                content_norm = normalize_text(record["content"])
                mentions, tags = extract_entities(content_norm)
                record["mentions"] = mentions
                record["hashtags"] = tags

                items.append(record)
                count += 1

                if count % sleep_every == 0:
                    # Polite pause to reduce detection and be nice to servers
                    pause = sleep_seconds + random.uniform(-jitter, jitter)
                    logging.info(f"[{hashtag}] Collected={count}. Sleeping {pause:.2f}s...")
                    time.sleep(max(0.0, pause))

                if count >= max_items:
                    break
            break  # normal exit
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logging.exception(f"[{hashtag}] Failed after {retries} retries")
                break
            backoff = min(60, (2 ** retries) + random.uniform(0, 1.0))
            logging.warning(f"[{hashtag}] Error: {e}. Backing off {backoff:.1f}s (retry {retries}/{max_retries})")
            time.sleep(backoff)

    logging.info(f"Scraped {len(items)} items for {hashtag}")
    return items


def collect_tweets(
    hashtags: List[str],
    hours: int,
    min_total: int,
    per_tag_cap: int = 5000,
    out_dir: Path = Path("data/raw"),
) -> List[Path]:
    """
    Collect tweets across hashtags within the last `hours` hours, targeting at least `min_total`.
    Returns list of JSONL file paths created.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    until = utc_now()
    since = until - timedelta(hours=hours)

    per_tag_target = max(min_total // max(1, len(hashtags)), 1)
    per_tag_target = min(per_tag_target, per_tag_cap)

    logging.info(f"Collection window: {since.isoformat()} -> {until.isoformat()}")
    logging.info(f"Target per hashtag: {per_tag_target} (min_total={min_total})")

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Concurrency: scrape each hashtag in a separate thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(hashtags))) as ex:
        futs = {
            ex.submit(_scrape_hashtag, tag, since, until, per_tag_target): tag for tag in hashtags
        }
        for fut in concurrent.futures.as_completed(futs):
            tag = futs[fut]
            try:
                results[tag] = fut.result()
            except Exception as e:
                logging.exception(f"Failed to collect {tag}: {e}")
                results[tag] = []

    # Persist each hashtag to its own JSONL for traceability
    files = []
    ts = until.strftime("%Y%m%dT%H%M%SZ")
    for tag, rows in results.items():
        safe = tag.replace("#", "")
        fpath = out_dir / f"{safe}_{ts}.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                # Ensure normalized content before writing
                r["content"] = normalize_text(r["content"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files.append(fpath)
        logging.info(f"Wrote {len(rows)} rows -> {fpath}")

    total = sum(len(v) for v in results.values())
    if total < min_total:
        logging.warning(f"Collected {total} < requested min_total={min_total}. You may re-run or extend hours.")
    else:
        logging.info(f"Collected total {total} tweets. ✅")
    return files


# ------------------------------ Processing ---------------------------------

def _read_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping bad line in {p}")
    df = pd.DataFrame.from_records(records)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["tweet_id"])
    after = len(df)
    logging.info(f"Deduplicated: {before} -> {after} (removed {before - after})")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize content
    df["content"] = df["content"].astype(str).map(normalize_text)
    # Ensure lists for mentions/hashtags
    for col in ["mentions", "hashtags"]:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [str(x)]))
    # Parse time
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df["scraped_at_utc"] = pd.to_datetime(df["scraped_at_utc"], errors="coerce", utc=True)
    # Basic sanity
    for col in ["like_count", "retweet_count", "reply_count", "quote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # Partition date
    df["ingest_date"] = df["timestamp_utc"].dt.date.astype(str)
    return df


def write_parquet_partitioned(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    ensure_parquet_support()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for date, part in df.groupby("ingest_date"):
        part_dir = out_dir / f"ingest_date={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fpath = part_dir / "tweets.parquet"
        part.to_parquet(fpath, index=False)
        written.append(fpath)
        logging.info(f"Wrote {len(part)} rows -> {fpath}")
    return written


def process_raw(input_jsonl_glob: str, out_dir: Path) -> Path:
    paths = [Path(p) for p in sorted(map(str, Path().glob(input_jsonl_glob)))]
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_jsonl_glob}")
    df = _read_jsonl(paths)
    df = deduplicate(df)
    df = clean_df(df)
    write_parquet_partitioned(df, out_dir)
    return out_dir


# ------------------------------ Analysis -----------------------------------

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col: str):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.col].values


class LexiconFeaturizer(BaseEstimator, TransformerMixin):
    """Compute bullish/bearish keyword densities as lightweight numeric features."""
    def __init__(self, text_col: str = "content"):
        self.text_col = text_col

    def fit(self, X, y=None):
        return self

    def _score_text(self, txt: str) -> Tuple[float, float]:
            if not isinstance(txt, str):
                return 0.0, 0.0
            t = txt.lower()
            bull = sum(1 for w in BULLISH_WORDS if w in t)
            bear = sum(1 for w in BEARISH_WORDS if w in t)
            length = max(len(t.split()), 1)
            return bull / length, bear / length

    def transform(self, X):
        bulls, bears = [], []
        for txt in X[self.text_col].values:
            b, br = self._score_text(txt)
            bulls.append(b)
            bears.append(br)
        return np.vstack([bulls, bears]).T  # shape (n, 2)


def build_vectorizer(max_features: int = 20000, min_df: int = 2, ngram_max: int = 3):
    """
    Character n-gram TF-IDF (robust for multilingual/Indian scripts) + lexicon features.
    """
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, ngram_max),
        min_df=min_df,
        max_features=max_features,
        strip_accents=None,
        lowercase=False,  # keep original scripts/case
    )
    union = FeatureUnion([
        ("tfidf", tfidf),
        ("lexicon", LexiconFeaturizer("content")),
    ])
    return union


def compute_signals(parquet_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read all partitions
    paths = list(parquet_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files under {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.sort_values("timestamp_utc")
    # Vectorize
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df)  # sparse matrix + dense lexicon columns via union
    # Composite signal: combine engagement + lexicon
    # Example: signal = zscore(likes+retweets+replies+quotes) * (bullish-bearish)
    engagement = (df["like_count"] + df["retweet_count"] + df["reply_count"] + df["quote_count"]).values.reshape(-1, 1)
    scaler = StandardScaler(with_mean=False)  # engagement is sparse-friendly
    eng_z = scaler.fit_transform(engagement).A.ravel()
    # Lexicon signal (we can recompute quickly)
    lex = LexiconFeaturizer().transform(df)
    lex_delta = lex[:, 0] - lex[:, 1]
    # Composite
    signal = eng_z * (1.0 + 10.0 * lex_delta)  # amplify lexicon but keep bounded
    df["signal"] = signal

    # Aggregate per 15-min bucket and source_hashtag
    df["bucket"] = df["timestamp_utc"].dt.floor("15min")
    agg = df.groupby(["bucket", "source_hashtag"]).agg(
        n=("tweet_id", "count"),
        signal_mean=("signal", "mean"),
        signal_std=("signal", "std"),
    ).reset_index()
    agg["signal_std"] = agg["signal_std"].fillna(0.0)

    # Bootstrap CI for mean (approximate, B=200 for speed)
    def bootstrap_ci(group: pd.Series, B: int = 200, alpha: float = 0.05) -> Tuple[float, float]:
        if len(group) <= 1:
            return float(group.mean()), float(group.mean())
        means = []
        arr = group.values
        n = len(arr)
        rng = np.random.default_rng(42)
        for _ in range(B):
            sample = rng.choice(arr, size=n, replace=True)
            means.append(sample.mean())
        lo = np.quantile(means, alpha/2)
        hi = np.quantile(means, 1 - alpha/2)
        return float(lo), float(hi)

    cis = agg.groupby(["bucket", "source_hashtag"])["signal_mean"].apply(lambda s: bootstrap_ci(s))
    ci_df = cis.apply(pd.Series)
    ci_df.columns = ["ci_lo", "ci_hi"]
    agg = agg.merge(ci_df, left_on=["bucket", "source_hashtag"], right_index=True, how="left")

    # Write outputs
    signals_parquet = out_dir / "signals.parquet"
    agg.to_parquet(signals_parquet, index=False)
    summary_csv = out_dir / "signals_summary.csv"
    agg.sort_values(["bucket", "source_hashtag"]).to_csv(summary_csv, index=False)
    logging.info(f"Wrote signals: {signals_parquet} and {summary_csv}")
    return signals_parquet


# ------------------------------ Plotting -----------------------------------

def downsample_time_series(df: pd.DataFrame, target_points: int = 1000) -> pd.DataFrame:
    if len(df) <= target_points:
        return df
    # Take evenly spaced samples to avoid heavy memory
    idx = np.linspace(0, len(df) - 1, target_points).astype(int)
    return df.iloc[idx]


def plot_signals(signals_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(signals_path)
    paths = []
    for tag, sub in df.groupby("source_hashtag"):
        sub = sub.sort_values("bucket")
        sub_ds = downsample_time_series(sub, target_points=800)
        plt.figure(figsize=(10, 5))
        plt.plot(sub_ds["bucket"], sub_ds["signal_mean"], label=f"{tag} mean")
        # CI band
        plt.fill_between(sub_ds["bucket"], sub_ds["ci_lo"], sub_ds["ci_hi"], alpha=0.2, label=f"{tag} CI")
        plt.title(f"Composite Signal Over Time – {tag}")
        plt.xlabel("Time (15-min buckets)")
        plt.ylabel("Signal (unitless)")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"signal_{tag.replace('#','')}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)
        logging.info(f"Wrote plot -> {out_path}")
    return paths


# ------------------------------ CLI ----------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-time market intelligence pipeline (X/Twitter)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="Collect tweets via snscrape")
    pc.add_argument("--hashtags", nargs="+", default=HASHTAGS_DEFAULT, help="Hashtags to collect")
    pc.add_argument("--hours", type=int, default=24, help="Window size in hours (default: 24)")
    pc.add_argument("--min-total", type=int, default=2000, help="Min tweets total across hashtags")
    pc.add_argument("--per-tag-cap", type=int, default=8000, help="Ceiling per hashtag")
    pc.add_argument("--out", type=str, default="data/raw", help="Output directory for JSONL")

    pp = sub.add_parser("process", help="Process raw JSONL into Parquet (partitioned)")
    pp.add_argument("--input-jsonl", type=str, required=True, help="Glob for input JSONL, e.g., 'data/raw/*.jsonl'")
    pp.add_argument("--out", type=str, default="data/curated", help="Output directory for Parquet")

    pa = sub.add_parser("analyze", help="Compute composite signals from Parquet")
    pa.add_argument("--parquet", type=str, default="data/curated", help="Parquet root directory")
    pa.add_argument("--out", type=str, default="data/signals", help="Output directory for signals")

    pl = sub.add_parser("plot", help="Plot signals memory-efficiently")
    pl.add_argument("--signals", type=str, default="data/signals/signals.parquet", help="Signals Parquet path")
    pl.add_argument("--out", type=str, default="plots", help="Output directory for plots")

    pall = sub.add_parser("all", help="Run end-to-end: collect -> process -> analyze -> plot")
    pall.add_argument("--hashtags", nargs="+", default=HASHTAGS_DEFAULT, help="Hashtags to collect")
    pall.add_argument("--hours", type=int, default=24, help="Window size in hours")
    pall.add_argument("--min-total", type=int, default=2000, help="Min tweets total across hashtags")
    pall.add_argument("--per-tag-cap", type=int, default=8000, help="Ceiling per hashtag")
    pall.add_argument("--raw-out", type=str, default="data/raw", help="Raw JSONL output dir")
    pall.add_argument("--parquet-out", type=str, default="data/curated", help="Parquet output dir")
    pall.add_argument("--signals-out", type=str, default="data/signals", help="Signals output dir")
    pall.add_argument("--plots-out", type=str, default="plots", help="Plots output dir")

    return p.parse_args(argv)


def main(argv=None):
    setup_logging()
    args = parse_args(argv)
    if args.cmd == "collect":
        files = collect_tweets(
            hashtags=args.hashtags,
            hours=args.hours,
            min_total=args.min_total,
            per_tag_cap=args.per_tag_cap,
            out_dir=Path(args.out),
        )
        logging.info(f"Collect complete. Files: {files}")

    elif args.cmd == "process":
        out = process_raw(args.input_jsonl, Path(args.out))
        logging.info(f"Process complete. Parquet at: {out}")

    elif args.cmd == "analyze":
        out = compute_signals(Path(args.parquet), Path(args.out))
        logging.info(f"Analyze complete. Signals at: {out}")

    elif args.cmd == "plot":
        paths = plot_signals(Path(args.signals), Path(args.out))
        logging.info(f"Plot complete. Files: {paths}")

    elif args.cmd == "all":
        files = collect_tweets(
            hashtags=args.hashtags,
            hours=args.hours,
            min_total=args.min_total,
            per_tag_cap=args.per_tag_cap,
            out_dir=Path(args.raw_out),
        )
        process_raw(f"{args.raw_out}/*.jsonl", Path(args.parquet_out))
        compute_signals(Path(args.parquet_out), Path(args.signals_out))
        plot_signals(Path(args.signals_out) / "signals.parquet", Path(args.plots_out))
        logging.info("All stages complete. ✅")
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
