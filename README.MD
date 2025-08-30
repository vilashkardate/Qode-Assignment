# Real-time Market Intelligence (Indian Markets, X/Twitter)

This project collects, processes, stores, and analyzes tweets related to Indian stock markets (e.g., `#nifty50`, `#sensex`, `#intraday`, `#banknifty`) without using paid APIs.

## ⚠️ Important
Scraping X/Twitter may be restricted by their Terms of Service. This project uses `snscrape`, which can fetch public tweets without authentication. Ensure your usage complies with the platform's ToS and local laws. Use polite rate limiting.

## Features
- **Collection**: `snscrape`-based, concurrent across hashtags, backoff + retry, JSONL outputs.
- **Processing**: Unicode-safe normalization; deduplication; Parquet storage partitioned by date.
- **Analysis**: Char n-gram TF-IDF + lexicon features; composite trading signal with bootstrap CIs.
- **Visualization**: Memory-efficient downsampling, static PNG plots (no GUI required).
- **Scalability**: Designed to scale ~10x with concurrency, partitioned storage, sparse vectors.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Collect 24h tweets (target 2000 total across hashtags)
python market_intel_pipeline.py collect --min-total 2000 --hours 24

# Process -> Parquet
python market_intel_pipeline.py process --input-jsonl "data/raw/*.jsonl" --out data/curated

# Analyze -> Signals
python market_intel_pipeline.py analyze --parquet data/curated --out data/signals

# Plot
python market_intel_pipeline.py plot --signals data/signals/signals.parquet --out plots