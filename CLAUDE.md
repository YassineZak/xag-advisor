# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

XAG/EUR silver investment advisor — a single-page Streamlit app that fetches live market data and produces a monthly buy/wait signal for the user's Revolut silver portfolio.

Deployed on Streamlit Community Cloud, accessible from iPhone via browser.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# The app auto-reloads on file save — no restart needed during development
```

## Architecture

Single file: `app.py`. No backend, no database, no auth.

**Data flow:**
1. `get_data()` — downloads 400 days of `XAGUSD=X`, `EURUSD=X`, `XAUUSD=X` via yfinance, computes `XAG_EUR = XAG_USD / EUR_USD` and the gold/silver ratio. Cached 1 hour via `@st.cache_data(ttl=3600)`.
2. `rsi(series, period)` / `bollinger(series, period)` — pure pandas indicator functions, no external TA library.
3. `compute_score(df)` — scores 5 indicators (RSI, Bollinger band position, SMA20/50 vs price, gold/silver ratio, 1-month performance) into a 0–100 buy signal. Returns `(score, reasons, rsi_val)` where `reasons` is a list of `(icon, text)` tuples.
4. UI renders: 3 metric cards → buy/wait signal with score bar → price+Bollinger+SMA chart (top) + RSI chart (bottom, shared x-axis) → gold/silver ratio chart.

**Score thresholds:** ≥60 = buy, 40–59 = neutral, <40 = wait. Base score starts at 10.

**Charts:** Plotly dark theme throughout. Main chart uses `make_subplots` with `shared_xaxes=True` (rows 0.68/0.32 height split). All charts use `use_container_width=True` for mobile responsiveness.

## Data sources

All via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance, free, no API key):
- `XAGUSD=X` — silver spot price in USD
- `EURUSD=X` — EUR/USD exchange rate
- `XAUUSD=X` — gold spot price in USD (used for ratio only)

## Deployment

GitHub repo: https://github.com/YassineZak/xag-advisor  
Streamlit Cloud: deployed from `main` branch, entry point `app.py`.  
Any push to `main` triggers automatic redeployment.
