# Bitcoin Tab — Design Spec
**Date:** 2026-04-10  
**Project:** xag-advisor  
**Status:** Approved

---

## Overview

Add a Bitcoin (BTC) portfolio tracking tab to the existing XAG Advisor Streamlit app. The new tab provides real-time BTC balance from Binance, live EUR valuation, P&L tracking, and a buy signal scored 0–100 using technical indicators and the Fear & Greed Index.

---

## Architecture

### File Structure

```
app.py                          ← refactored: entry point only, renders st.tabs()
xag_tab.py                      ← extracted from current app.py (XAG logic unchanged)
btc_tab.py                      ← new: all Bitcoin logic
portfolio.json                  ← add btc_avg_price field
requirements.txt                ← add python-binance
.streamlit/
  secrets.toml                  ← BINANCE_API_KEY, BINANCE_API_SECRET (never committed)
```

### app.py (after refactor)

```python
tab1, tab2 = st.tabs(["🥈 Argent (XAG)", "₿ Bitcoin (BTC)"])
with tab1:
    xag_tab.render()
with tab2:
    btc_tab.render()
```

`app.py` contains no business logic — only tab orchestration.

---

## Data Sources

| Data | Source | API Key Required |
|------|--------|-----------------|
| BTC price history (6 months) | yfinance `BTC-USD` | No |
| BTC/EUR price (live) | yfinance `BTC-EUR` | No |
| BTC balance (live) | Binance API (read-only) | Yes |
| Fear & Greed Index (current + 30d history) | alternative.me free API | No |

### Binance API Security
- API key stored in `.streamlit/secrets.toml` — never committed to git
- Key created with **read-only permissions** on Binance (no trading, no withdrawal)
- On Streamlit Cloud: configured via project Settings → Secrets
- `.streamlit/secrets.toml` added to `.gitignore`

### portfolio.json changes
Add one field: `btc_avg_price` (user's average BTC purchase price in EUR).  
BTC quantity is fetched live from Binance — no need to store it.

```json
{
  "quantity": 16.7594,
  "avg_price": 0.0,
  "last_updated": "2026-04-09",
  "btc_avg_price": 0.0
}
```

---

## BTC Tab — UI Layout

### Bloc 1 — Real-Time Metrics (4 columns)
- Current BTC price (EUR)
- BTC balance (from Binance)
- Portfolio value in EUR
- P&L since average purchase price (% and EUR)

### Bloc 2 — Buy Signal (centered, prominent)
- Score 0–100 with color gradient (red → orange → green)
- Label: Strong Buy / Buy / Neutral / Wait / Avoid
- Breakdown of all 4 indicators with individual contributions

### Bloc 3 — Fear & Greed Index
- Current value + label (Extreme Fear / Fear / Neutral / Greed / Extreme Greed)
- 30-day historical chart

### Bloc 4 — Technical Charts
- BTC price + Bollinger Bands (20-period) + SMA20/SMA50 (6 months)
- RSI 14-day below, shared x-axis

### Bloc 5 — Portfolio Evolution
- EUR value over time since average purchase date
- Same dark-mode Plotly style as XAG tab

---

## Buy Signal Scoring

Score: **0 to 100** (higher = stronger buy signal)

| Indicator | Weight | Buy Condition | Points |
|-----------|--------|---------------|--------|
| RSI 14-day | 25% | <30 oversold | 0–25 |
| Bollinger position | 20% | Price below lower band | 0–20 |
| Fear & Greed Index | 35% | <25 Extreme Fear | 0–35 |
| Post-halving cycle | 20% | Months 6–18 after halving (Oct 2024 – Oct 2025) | 0–20 |

**Thresholds:**
- **75–100** → "Strong Buy" (green)
- **60–74** → "Buy" (light green)
- **40–59** → "Neutral / Accumulate" (yellow)
- **25–39** → "Wait" (orange)
- **0–24** → "Avoid / Overbought" (red)

### Rationale
Fear & Greed carries the highest weight (35%) because BTC's best historical entry points (March 2020, June 2022, November 2022) all occurred during Extreme Fear (<20). Technical indicators provide timing precision; the halving cycle provides macro context.

### Halving Cycle Logic
Last halving: April 19, 2024. Historical bull market peak: months 12–18 post-halving (April–October 2025). After month 18, reduce cycle score gradually. Score decays linearly to 0 by month 30 (October 2026).

---

## Caching Strategy

Consistent with existing XAG tab:
- BTC price history: `@st.cache_data(ttl=3600)` (1 hour)
- Binance balance: `@st.cache_data(ttl=300)` (5 minutes)
- Fear & Greed: `@st.cache_data(ttl=3600)` (1 hour)

---

## Error Handling

- If Binance API key not configured: display a warning and show 0 balance (app still usable)
- If yfinance fails: display error message, skip charts
- If Fear & Greed API fails: exclude from score, show "data unavailable"

---

## Dependencies

Add to `requirements.txt`:
```
python-binance>=1.0.17
```

---

## Out of Scope

- Price alerts / notifications
- Buy/sell order execution via Binance API
- Multi-exchange support
- Tax reporting
