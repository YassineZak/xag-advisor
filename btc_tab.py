import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date


# ── Données historiques BTC ───────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_btc_data() -> pd.DataFrame:
    """
    Récupère 400 jours d'historique BTC-USD via yfinance.
    Calcule RSI, Bollinger Bands, SMA20, SMA50.
    Retourne un DataFrame avec colonnes : Close, RSI, SMA20, SMA50, BB_upper, BB_lower, BB_mid
    """
    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(period="400d")

    if hist.empty or len(hist) < 30:
        raise ValueError("Données BTC insuffisantes — réessaie dans quelques minutes.")

    df = pd.DataFrame()
    df["Close"] = hist["Close"].copy()
    df.index = df.index.tz_localize(None)

    # RSI 14 jours
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # Bollinger Bands 20 périodes
    df["BB_mid"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std
    df["BB_lower"] = df["BB_mid"] - 2 * std

    # Moyennes mobiles
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    return df


# ── Prix BTC live ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_btc_live_price() -> tuple:
    """
    Retourne (prix_eur, timestamp) du dernier prix BTC/EUR.
    Cache 5 minutes.
    """
    try:
        info = yf.Ticker("BTC-EUR").fast_info
        price = info.last_price
        if price and price > 0:
            return float(price), datetime.now()
    except Exception:
        pass
    return None, None




# ── Univers & signaux crypto ──────────────────────────────────────────────────

# Mapping Yahoo Finance ticker → slug Bitpanda (format: /en/prices/name-symbol)
BITPANDA_SLUGS = {
    "BTC-USD": "bitcoin-btc", "ETH-USD": "ethereum-eth", "BNB-USD": "bnb-bnb",
    "SOL-USD": "solana-sol", "XRP-USD": "xrp-xrp", "ADA-USD": "cardano-ada",
    "AVAX-USD": "avalanche-avax", "DOT-USD": "polkadot-dot", "LINK-USD": "chainlink-link",
    "LTC-USD": "litecoin-ltc", "ATOM-USD": "cosmos-atom", "NEAR-USD": "near-protocol-near",
    "UNI-USD": "uniswap-uni", "ALGO-USD": "algorand-algo", "FIL-USD": "filecoin-fil",
    "DOGE-USD": "doge-doge", "SHIB-USD": "shiba-inu-shib", "PEPE-USD": "pepe-pepe",
    "FLOKI-USD": "floki-floki", "BONK-USD": "bonk-bonk", "WIF-USD": "dogwifhat-wif",
    "RENDER-USD": "render-rndr", "INJ-USD": "injective-inj", "SEI-USD": "sei-sei",
    "SUI-USD": "sui-sui", "APT-USD": "aptos-apt", "TIA-USD": "celestia-tia",
    "SAND-USD": "the-sandbox-sand", "JASMY-USD": "jasmycoin-jasmy", "FET-USD": "fetch-ai-fet",
}


CRYPTO_UNIVERSE = {
    "longterm": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD",
        "ATOM-USD", "NEAR-USD", "UNI-USD", "ALGO-USD", "FIL-USD",
    ],
    "shortterm": [
        "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD",
        "WIF-USD", "RENDER-USD", "INJ-USD", "SEI-USD", "SUI-USD",
        "APT-USD", "TIA-USD", "SAND-USD", "JASMY-USD", "FET-USD",
    ],
}


@st.cache_data(ttl=3600)
def get_crypto_signals(category: str) -> list:
    """
    Télécharge 90j d'historique pour les cryptos de la catégorie (batch yfinance),
    calcule un score 0-100 par crypto, retourne les 5 meilleures triées par score.
    Score : RSI 14j (30 pts) + Bollinger (25 pts) + Perf 1 mois (25 pts) + vs SMA50 (20 pts).
    """
    tickers = CRYPTO_UNIVERSE[category]
    try:
        raw = yf.download(tickers, period="90d", auto_adjust=True, progress=False)
        close_df = raw["Close"]
    except Exception:
        return []

    results = []
    for ticker in tickers:
        try:
            close = close_df[ticker].dropna()
            if len(close) < 30:
                continue

            symbol = ticker.replace("-USD", "")
            current_price = float(close.iloc[-1])
            price_24h = float(close.iloc[-2]) if len(close) >= 2 else current_price
            price_1m = float(close.iloc[-21]) if len(close) >= 21 else current_price

            var_24h = (current_price - price_24h) / price_24h * 100
            perf_1m = (current_price - price_1m) / price_1m * 100

            # RSI 14j
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])

            # Bollinger 20j
            bb_mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            bb_upper = float((bb_mid + 2 * std).iloc[-1])
            bb_lower = float((bb_mid - 2 * std).iloc[-1])
            bb_range = bb_upper - bb_lower
            bb_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

            # SMA50
            sma50 = float(close.rolling(50).mean().iloc[-1])

            # ── Score ────────────────────────────────────────────────────────
            score = 30  # base neutre

            # RSI (30 pts)
            if rsi < 30:
                score += 30
            elif rsi < 45:
                score += 15
            elif rsi > 70:
                score -= 20
            elif rsi > 55:
                score -= 8

            # Bollinger (25 pts)
            if bb_pos < 0.2:
                score += 25
            elif bb_pos < 0.4:
                score += 12
            elif bb_pos > 0.8:
                score -= 15

            # Perf 1 mois (25 pts)
            if perf_1m < -20:
                score += 25
            elif perf_1m < -10:
                score += 15
            elif perf_1m < 0:
                score += 5
            elif perf_1m > 30:
                score -= 15
            elif perf_1m > 15:
                score -= 8

            # vs SMA50 (20 pts)
            if current_price < sma50 * 0.85:
                score += 20
            elif current_price < sma50:
                score += 10
            elif current_price > sma50 * 1.2:
                score -= 10

            final_score = max(0, min(100, score))

            if final_score >= 75:
                label = "ACHAT FORT"
            elif final_score >= 60:
                label = "ACHETER"
            elif final_score >= 40:
                label = "NEUTRE"
            elif final_score >= 25:
                label = "ATTENDRE"
            else:
                label = "ÉVITER"

            results.append({
                "symbol": symbol,
                "price": current_price,
                "var_24h": var_24h,
                "perf_1m": perf_1m,
                "rsi": rsi,
                "score": final_score,
                "label": label,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


def _render_crypto_signals(signals: list, color_accent: str, category: str) -> None:
    """Affiche 5 cartes de signaux crypto avec lien d'achat vers Bitpanda."""
    if not signals:
        st.info("Données indisponibles — réessaie dans quelques instants.")
        return

    COLOR_MAP = {
        "ACHAT FORT": "#22c55e",
        "ACHETER":    "#86efac",
        "NEUTRE":     "#facc15",
        "ATTENDRE":   "#f97316",
        "ÉVITER":     "#ef4444",
    }

    cols = st.columns(len(signals))
    for col, s in zip(cols, signals):
        c = COLOR_MAP.get(s["label"], "#94a3b8")
        var_arrow = "▲" if s["var_24h"] >= 0 else "▼"
        var_color = "#22c55e" if s["var_24h"] >= 0 else "#ef4444"
        with col:
            st.markdown(f"""
<div style="background:#1e1e2e; border-radius:10px; padding:12px; text-align:center; border-top: 3px solid {color_accent};">
  <div style="font-size:1.3rem; font-weight:bold; color:#f1f5f9;">{s['symbol']}</div>
  <div style="font-size:0.85rem; color:#94a3b8; margin:2px 0;">${s['price']:,.4g}</div>
  <div style="font-size:0.8rem; color:{var_color};">{var_arrow} {abs(s['var_24h']):.1f}% 24h</div>
  <div style="margin:8px 0; font-size:1.5rem; font-weight:bold; color:{c};">{s['score']}</div>
  <div style="font-size:0.75rem; font-weight:600; color:{c};">{s['label']}</div>
  <div style="font-size:0.7rem; color:#64748b; margin-top:4px;">RSI {s['rsi']:.0f} · 1m {s['perf_1m']:+.0f}%</div>
</div>
            """, unsafe_allow_html=True)
            slug = BITPANDA_SLUGS.get(s["symbol"], s["symbol"].replace("-USD", "").lower())
            st.link_button(
                "🛒 Acheter sur Bitpanda",
                url=f"https://www.bitpanda.com/en/prices/{slug}",
                use_container_width=True,
            )


# ── Portfolio Bitpanda ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_bitpanda_portfolio() -> dict:
    """
    Récupère tous les soldes depuis Bitpanda : crypto ET fiat.
    Retourne {"crypto": {symbol: balance}, "fiat": {symbol: balance}}.
    Cache 5 minutes.
    """
    result: dict = {"crypto": {}, "fiat": {}}
    api_key = st.secrets.get("BITPANDA_API_KEY", "")
    if not api_key:
        return result

    headers = {"X-API-KEY": api_key}

    try:
        resp = requests.get("https://api.bitpanda.com/v1/wallets", headers=headers, timeout=10)
        resp.raise_for_status()
        for w in resp.json().get("data", []):
            attrs = w.get("attributes", {})
            symbol = attrs.get("cryptocoin_symbol", "")
            balance = float(attrs.get("balance", 0))
            if symbol and balance > 0:
                result["crypto"][symbol] = result["crypto"].get(symbol, 0) + balance
    except Exception:
        pass

    try:
        resp = requests.get("https://api.bitpanda.com/v1/fiatwallets", headers=headers, timeout=10)
        resp.raise_for_status()
        for w in resp.json().get("data", []):
            attrs = w.get("attributes", {})
            symbol = attrs.get("fiat_symbol", "")
            balance = float(attrs.get("balance", 0))
            if symbol and balance > 0:
                result["fiat"][symbol] = result["fiat"].get(symbol, 0) + balance
    except Exception:
        pass

    return result


@st.cache_data(ttl=300)
def get_bitpanda_values() -> dict:
    """
    Retourne holdings Bitpanda valorisés en EUR + total.
    {"holdings": {symbol: {balance, value_eur, type}}, "total_eur": float}
    """
    data = get_bitpanda_portfolio()
    holdings: dict = {}
    total_eur = 0.0

    for symbol, balance in data["fiat"].items():
        holdings[symbol] = {"balance": balance, "value_eur": balance, "type": "fiat"}
        total_eur += balance

    for symbol, balance in data["crypto"].items():
        price = 0.0
        try:
            p = yf.Ticker(f"{symbol}-EUR").fast_info.last_price
            if p and p > 0:
                price = float(p)
        except Exception:
            pass
        value = balance * price
        holdings[symbol] = {"balance": balance, "price_eur": price, "value_eur": value, "type": "crypto"}
        total_eur += value

    return {"holdings": holdings, "total_eur": total_eur}


# ── Historique des transactions Bitpanda ──────────────────────────────────────

@st.cache_data(ttl=86400)
def _historical_price_eur(symbol: str, date_only: str) -> float:
    """Prix EUR d'une crypto à une date donnée (YYYY-MM-DD). Cache 24h."""
    if not symbol or not date_only:
        return 0.0
    try:
        target = pd.to_datetime(date_only).normalize()
        hist = yf.Ticker(f"{symbol}-EUR").history(
            start=(target - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
            end=(target + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=600)
def get_bitpanda_wallet_txs() -> list:
    """
    Récupère les dépôts/retraits on-chain crypto via /v1/wallets/transactions.
    Ces mouvements (transfert depuis Revolut, retrait vers Ledger, etc.) ne sont
    PAS dans /v1/trades — il faut les compter séparément pour un P&L correct.

    EUR value : champ amount_eur Bitpanda si dispo, sinon prix historique yfinance.
    Retourne [{date, symbol, direction ('in'|'out'), amount_crypto, amount_eur}].
    Cache 10 min.
    """
    api_key = st.secrets.get("BITPANDA_API_KEY", "")
    if not api_key:
        return []

    headers = {"X-API-KEY": api_key}
    txs: list = []
    page = 1
    page_size = 500

    while True:
        try:
            resp = requests.get(
                "https://api.bitpanda.com/v1/wallets/transactions",
                headers=headers,
                params={"page": page, "page_size": page_size},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break

        rows = payload.get("data", [])
        if not rows:
            break

        for t in rows:
            a = t.get("attributes", {})
            direction_raw = a.get("in_or_out", "")
            tx_type = a.get("type", "")

            if tx_type not in ("deposit", "withdrawal"):
                continue
            if direction_raw not in ("incoming", "outgoing"):
                continue

            symbol = a.get("cryptocoin_symbol", "")
            if not symbol:
                continue

            try:
                amount_crypto = float(a.get("amount", 0) or 0)
            except (TypeError, ValueError):
                continue
            if amount_crypto <= 0:
                continue

            date_str = a.get("time", {}).get("date_iso8601", "")

            amount_eur = 0.0
            try:
                ae = a.get("amount_eur")
                if ae:
                    amount_eur = float(ae)
            except (TypeError, ValueError):
                pass
            if amount_eur <= 0:
                price = _historical_price_eur(symbol, date_str[:10] if date_str else "")
                amount_eur = amount_crypto * price

            txs.append({
                "date": date_str,
                "symbol": symbol,
                "direction": "in" if direction_raw == "incoming" else "out",
                "amount_crypto": amount_crypto,
                "amount_eur": amount_eur,
            })

        if len(rows) < page_size:
            break
        page += 1
        if page > 50:
            break

    txs.sort(key=lambda x: x["date"], reverse=True)
    return txs


@st.cache_data(ttl=600)
def get_bitpanda_trades() -> list:
    """
    Récupère toutes les transactions crypto Bitpanda via /v1/trades (paginé).
    Retourne une liste de dicts triés du plus récent au plus ancien :
        {date, symbol, type, amount_crypto, amount_eur, price_eur, is_swap, is_savings}
    Cache 10 minutes.
    """
    api_key = st.secrets.get("BITPANDA_API_KEY", "")
    if not api_key:
        return []

    headers = {"X-API-KEY": api_key}
    trades: list = []
    page = 1
    page_size = 500

    while True:
        try:
            resp = requests.get(
                "https://api.bitpanda.com/v1/trades",
                headers=headers,
                params={"page": page, "page_size": page_size},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break

        rows = payload.get("data", [])
        if not rows:
            break

        for t in rows:
            a = t.get("attributes", {})
            if a.get("status") != "finished":
                continue
            try:
                amount_fiat = float(a.get("amount_fiat", 0) or 0)
                fiat_to_eur = float(a.get("fiat_to_eur_rate", 1) or 1)
                price = float(a.get("price", 0) or 0)
                trades.append({
                    "date": a.get("time", {}).get("date_iso8601", ""),
                    "symbol": a.get("cryptocoin_symbol", ""),
                    "type": a.get("type", ""),  # "buy" ou "sell"
                    "amount_crypto": float(a.get("amount_cryptocoin", 0) or 0),
                    "amount_eur": amount_fiat * fiat_to_eur,
                    "price_eur": price * fiat_to_eur,
                    "is_swap": bool(a.get("is_swap", False)),
                    "is_savings": bool(a.get("is_savings", False)),
                })
            except (TypeError, ValueError):
                continue

        if len(rows) < page_size:
            break
        page += 1
        if page > 50:  # garde-fou
            break

    trades.sort(key=lambda x: x["date"], reverse=True)
    return trades


def compute_crypto_pnl(
    trades: list,
    wallet_txs: list,
    current_prices_eur: dict,
    balances: dict,
) -> dict:
    """
    Calcule le P&L par crypto et global à partir des trades + transferts on-chain.

    Méthode cash-flow simple :
        Acheté  = trades buy  + dépôts entrants  (valorisés EUR au moment du transfert)
        Revendu = trades sell + retraits sortants (valorisés EUR au moment du transfert)
        P&L     = valeur_actuelle + revendu − acheté

    Les transferts entrants comptent comme un achat à la valeur EUR du jour : ils
    représentent de la crypto acquise hors Bitpanda (Revolut, Ledger, etc.).
    Les retraits sortants comptent comme une vente virtuelle au prix du marché.
    """
    per_asset: dict = {}

    def _bucket(sym: str) -> dict:
        return per_asset.setdefault(sym, {
            "bought": 0.0, "sold": 0.0,
            "deposited_eur": 0.0, "withdrawn_eur": 0.0,
            "deposited_qty": 0.0, "withdrawn_qty": 0.0,
        })

    for t in trades:
        sym = t["symbol"]
        if not sym:
            continue
        a = _bucket(sym)
        if t["type"] == "buy":
            a["bought"] += t["amount_eur"]
        elif t["type"] == "sell":
            a["sold"] += t["amount_eur"]

    for w in wallet_txs:
        sym = w["symbol"]
        if not sym:
            continue
        a = _bucket(sym)
        if w["direction"] == "in":
            a["bought"] += w["amount_eur"]
            a["deposited_eur"] += w["amount_eur"]
            a["deposited_qty"] += w["amount_crypto"]
        elif w["direction"] == "out":
            a["sold"] += w["amount_eur"]
            a["withdrawn_eur"] += w["amount_eur"]
            a["withdrawn_qty"] += w["amount_crypto"]

    total_bought = total_sold = total_current = 0.0
    for sym in set(per_asset.keys()) | set(balances.keys()):
        a = _bucket(sym)
        balance = float(balances.get(sym, 0.0))
        price = float(current_prices_eur.get(sym, 0.0))
        net_invested = a["bought"] - a["sold"]
        a["balance"] = balance
        a["current_value"] = balance * price
        a["net_invested"] = net_invested
        a["pnl"] = a["current_value"] + a["sold"] - a["bought"]
        denom = net_invested if net_invested > 0 else a["bought"]
        a["pnl_pct"] = (a["pnl"] / denom * 100) if denom > 0 else 0.0
        total_bought += a["bought"]
        total_sold += a["sold"]
        total_current += a["current_value"]

    total_net = total_bought - total_sold
    total_pnl = total_current + total_sold - total_bought
    total_denom = total_net if total_net > 0 else total_bought
    return {
        "per_asset": per_asset,
        "total": {
            "bought": total_bought,
            "sold": total_sold,
            "net_invested": total_net,
            "current_value": total_current,
            "pnl": total_pnl,
            "pnl_pct": (total_pnl / total_denom * 100) if total_denom > 0 else 0.0,
        },
    }


# ── Historique quotidien du portefeuille ──────────────────────────────────────

@st.cache_data(ttl=3600)
def _fetch_crypto_history_eur(symbols: tuple, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Prix EUR quotidiens (clôture) en batch pour une liste de cryptos via yfinance.
    Retourne un DataFrame index=date naïve, colonnes=symboles. Cache 1h.
    """
    if not symbols:
        return pd.DataFrame()

    tickers = [f"{s}-EUR" for s in symbols]
    try:
        if len(tickers) == 1:
            raw = yf.Ticker(tickers[0]).history(start=start_date, end=end_date)
            if raw.empty:
                return pd.DataFrame()
            df = pd.DataFrame({symbols[0]: raw["Close"]})
        else:
            raw = yf.download(
                tickers, start=start_date, end=end_date,
                auto_adjust=True, progress=False,
            )
            df = raw["Close"].copy()
            df.columns = [c.replace("-EUR", "") for c in df.columns]
    except Exception:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df.index = df.index.normalize()
    return df


def compute_portfolio_history(trades: list, wallet_txs: list) -> pd.DataFrame:
    """
    Reconstruit la valeur quotidienne du portefeuille crypto + investi net cumulé,
    depuis la première transaction jusqu'à aujourd'hui.

    Pour chaque jour :
        - Solde par crypto = somme cumulée des entrées (buy + deposit) − sorties
        - Valeur = solde × prix EUR du jour (clôture, ffill week-ends)
        - Investi net cumulé = somme cumulée des EUR engagés (buy + deposit − sell − withdraw)

    Retourne DataFrame [value_eur, net_invested_eur, pnl_eur] indexé quotidien.
    """
    qty_events: list = []   # (date, sym, qty_delta)
    cash_events: list = []  # (date, eur_delta)

    def _to_naive_day(s: str):
        return pd.to_datetime(s, utc=True).tz_localize(None).normalize()

    for t in trades:
        if not t.get("date") or not t.get("symbol"):
            continue
        date = _to_naive_day(t["date"])
        if t["type"] == "buy":
            qty_events.append((date, t["symbol"], t["amount_crypto"]))
            cash_events.append((date, t["amount_eur"]))
        elif t["type"] == "sell":
            qty_events.append((date, t["symbol"], -t["amount_crypto"]))
            cash_events.append((date, -t["amount_eur"]))

    for w in wallet_txs:
        if not w.get("date") or not w.get("symbol"):
            continue
        date = _to_naive_day(w["date"])
        sign = 1 if w["direction"] == "in" else -1
        qty_events.append((date, w["symbol"], sign * w["amount_crypto"]))
        cash_events.append((date, sign * w["amount_eur"]))

    if not qty_events:
        return pd.DataFrame()

    first_date = min(e[0] for e in qty_events)
    today = pd.Timestamp.now().normalize()
    dates = pd.date_range(first_date, today, freq="D")

    symbols = sorted({e[1] for e in qty_events})

    # Holdings cumulés par crypto par jour
    holdings = pd.DataFrame(0.0, index=dates, columns=symbols)
    for date, sym, delta in qty_events:
        if date in holdings.index:
            holdings.loc[date:, sym] = holdings.loc[date:, sym] + delta

    # Prix historiques (batch + cached)
    prices_raw = _fetch_crypto_history_eur(
        tuple(symbols),
        first_date.strftime("%Y-%m-%d"),
        (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    prices = pd.DataFrame(0.0, index=dates, columns=symbols)
    if not prices_raw.empty:
        aligned = prices_raw.reindex(dates).ffill()
        for s in symbols:
            if s in aligned.columns:
                prices[s] = aligned[s].fillna(0.0)

    values = holdings * prices
    total_value = values.sum(axis=1)

    # Investi net cumulé
    cash_series = pd.Series(0.0, index=dates)
    for date, delta in cash_events:
        if date in cash_series.index:
            cash_series.loc[date:] = cash_series.loc[date:] + delta

    result = pd.DataFrame({
        "value_eur": total_value,
        "net_invested_eur": cash_series,
    })
    result["pnl_eur"] = result["value_eur"] - result["net_invested_eur"]
    return result


# ── Fear & Greed Index ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_fear_greed() -> tuple:
    """
    Récupère le Fear & Greed Index depuis alternative.me (gratuit, pas de clé API).
    Retourne (valeur_actuelle, label_actuel, dataframe_30j).
    dataframe_30j a colonnes : value, label (index = date)
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=30",
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        records = []
        for entry in data:
            records.append({
                "date": pd.Timestamp(int(entry["timestamp"]), unit="s"),
                "value": int(entry["value"]),
                "label": entry["value_classification"],
            })

        df_fg = pd.DataFrame(records).set_index("date").sort_index()
        current_value = int(df_fg["value"].iloc[-1])
        current_label = df_fg["label"].iloc[-1]
        return current_value, current_label, df_fg

    except Exception:
        return None, None, pd.DataFrame()


# ── Score d'achat BTC ─────────────────────────────────────────────────────────

HALVING_DATE = date(2024, 4, 19)   # dernier halving Bitcoin


def _halving_cycle_score() -> tuple:
    """
    Calcule les points liés au cycle post-halving (0–20 pts).
    Mois 0–6   : 10 pts (début cycle, incertitude)
    Mois 6–18  : 20 pts (bull market historique)
    Mois 18–30 : décroissance linéaire 20→0 pts
    Mois >30   : 0 pts
    """
    today = date.today()
    months_since = (today.year - HALVING_DATE.year) * 12 + (today.month - HALVING_DATE.month)

    if months_since < 6:
        return 10, f"Cycle précoce post-halving (mois {months_since}) — accumulation"
    elif months_since <= 18:
        return 20, f"Phase bull historique post-halving (mois {months_since}/18) — favorable"
    elif months_since <= 30:
        pts = int(20 * (30 - months_since) / 12)
        return pts, f"Fin de cycle post-halving (mois {months_since}) — signal décroissant ({pts} pts)"
    else:
        return 0, f"Cycle post-halving terminé (mois {months_since}) — neutre"


def compute_btc_score(
    df: pd.DataFrame,
    fear_greed,
) -> tuple:
    """
    Score d'achat BTC 0–100 basé sur 4 indicateurs pondérés.

    Pondérations :
      RSI 14j           : 25 pts max
      Bollinger position : 20 pts max
      Fear & Greed Index : 35 pts max
      Cycle post-halving : 20 pts max

    Retourne (score, label, reasons) où reasons est une liste de (icon, texte).
    """
    score = 30  # base neutre (conditions totalement neutres + cycle normal → ~40 pts)
    reasons = []

    rsi_val  = df["RSI"].iloc[-1]
    close    = df["Close"].iloc[-1]
    bb_lower = df["BB_lower"].iloc[-1]
    bb_upper = df["BB_upper"].iloc[-1]

    # ── 1. RSI (25 pts max) ──────────────────────────────────────────────────
    if rsi_val < 30:
        score += 25
        reasons.append(("✅", f"RSI très bas ({rsi_val:.1f}) — zone de survente forte"))
    elif rsi_val < 45:
        score += 12
        reasons.append(("✅", f"RSI bas ({rsi_val:.1f}) — légère survente"))
    elif rsi_val > 70:
        score -= 20
        reasons.append(("🔴", f"RSI élevé ({rsi_val:.1f}) — zone de surachat"))
    elif rsi_val > 55:
        score -= 8
        reasons.append(("⚠️", f"RSI légèrement élevé ({rsi_val:.1f})"))
    else:
        reasons.append(("➡️", f"RSI neutre ({rsi_val:.1f})"))

    # ── 2. Bollinger position (20 pts max) ───────────────────────────────────
    bb_range = bb_upper - bb_lower
    bb_pos   = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    if bb_pos < 0.2:
        score += 20
        reasons.append(("✅", f"Prix sous bande basse Bollinger ({bb_pos:.0%}) — signal d'achat fort"))
    elif bb_pos < 0.4:
        score += 10
        reasons.append(("✅", f"Prix proche bande basse Bollinger ({bb_pos:.0%})"))
    elif bb_pos > 0.8:
        score -= 12
        reasons.append(("🔴", f"Prix proche bande haute Bollinger ({bb_pos:.0%}) — attention"))
    else:
        reasons.append(("➡️", f"Prix dans bande médiane Bollinger ({bb_pos:.0%})"))

    # ── 3. Fear & Greed Index (35 pts max) ───────────────────────────────────
    if fear_greed is None:
        reasons.append(("➡️", "Fear & Greed Index indisponible"))
    elif fear_greed <= 25:
        score += 35
        reasons.append(("✅", f"Extreme Fear ({fear_greed}/100) — historiquement les meilleurs points d'entrée BTC"))
    elif fear_greed <= 40:
        score += 20
        reasons.append(("✅", f"Fear ({fear_greed}/100) — sentiment négatif, opportunité potentielle"))
    elif fear_greed <= 60:
        reasons.append(("➡️", f"Neutre ({fear_greed}/100) — sentiment équilibré"))
    elif fear_greed <= 75:
        score -= 15
        reasons.append(("⚠️", f"Greed ({fear_greed}/100) — marché euphorique, prudence"))
    else:
        score -= 25
        reasons.append(("🔴", f"Extreme Greed ({fear_greed}/100) — risque élevé de correction"))

    # ── 4. Cycle post-halving (20 pts max) ───────────────────────────────────
    cycle_pts, cycle_msg = _halving_cycle_score()
    score += cycle_pts
    icon = "✅" if cycle_pts >= 15 else ("➡️" if cycle_pts >= 5 else "🔴")
    reasons.append((icon, cycle_msg))

    final_score = max(0, min(100, score))

    # ── Label ────────────────────────────────────────────────────────────────
    if final_score >= 75:
        label = "ACHAT FORT"
    elif final_score >= 60:
        label = "ACHETER"
    elif final_score >= 40:
        label = "NEUTRE"
    elif final_score >= 25:
        label = "ATTENDRE"
    else:
        label = "ÉVITER"

    return final_score, label, reasons


# ── Interface UI ──────────────────────────────────────────────────────────────

def render():
    """Point d'entrée du tab Portfolio Crypto — appelé par app.py."""

    # ── Titre & bouton refresh ────────────────────────────────────────────────
    col_title, col_refresh = st.columns([5, 1])
    col_title.title("📊 Portfolio Bitpanda")
    if col_refresh.button("🔄 Rafraîchir", key="btc_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Chargement des données ────────────────────────────────────────────────
    portfolio_data = get_bitpanda_portfolio()
    crypto_balances = portfolio_data["crypto"]
    fiat_balances = portfolio_data["fiat"]

    if not st.secrets.get("BITPANDA_API_KEY", ""):
        st.warning("⚠️ Clé API Bitpanda non configurée — ajoute BITPANDA_API_KEY dans les secrets Streamlit.")

    # ── Bloc 1 : Vue d'ensemble portfolio ────────────────────────────────────
    st.subheader("💼 Mes avoirs Bitpanda")

    total_eur = sum(fiat_balances.values())  # fiat déjà en EUR

    # Prix live des crypto pour conversion en EUR
    crypto_prices_eur: dict = {}
    for symbol in crypto_balances:
        try:
            price = yf.Ticker(f"{symbol}-EUR").fast_info.last_price
            if price and price > 0:
                crypto_prices_eur[symbol] = float(price)
        except Exception:
            pass

    total_eur += sum(
        crypto_balances[s] * crypto_prices_eur.get(s, 0)
        for s in crypto_balances
    )

    # Affichage fiat
    all_holdings = {}
    for symbol, balance in fiat_balances.items():
        all_holdings[symbol] = {"balance": balance, "value_eur": balance, "type": "fiat"}
    for symbol, balance in crypto_balances.items():
        price = crypto_prices_eur.get(symbol, 0)
        all_holdings[symbol] = {"balance": balance, "value_eur": balance * price, "type": "crypto"}

    if all_holdings:
        # Grille en lignes de 4 colonnes : "Total" puis une carte par avoir.
        # (st.columns crée une rangée fixe → on découpe en rangées pour gérer
        #  n'importe quel nombre d'avoirs sans IndexError.)
        PER_ROW = 4
        items = [("__total__", {"type": "total"})] + list(all_holdings.items())
        for start in range(0, len(items), PER_ROW):
            row = items[start:start + PER_ROW]
            cols = st.columns(len(row))
            for col, (symbol, info) in zip(cols, row):
                if info["type"] == "total":
                    col.metric("Total portfolio", f"{total_eur:,.2f} €")
                elif info["type"] == "fiat":
                    col.metric(f"{symbol}", f"{info['balance']:,.2f} €")
                else:
                    price_str = f"≈ {info['value_eur']:,.2f} €" if info["value_eur"] > 0 else ""
                    col.metric(f"{symbol}", f"{info['balance']:.6f}",
                               delta=price_str if price_str else None)
    else:
        st.info("Aucun avoir détecté sur Bitpanda pour l'instant.")

    st.divider()

    # ── Bloc P&L crypto depuis le début ──────────────────────────────────────
    st.subheader("💰 P&L Crypto depuis le début")
    st.caption("Calculé à partir de l'historique complet de tes transactions Bitpanda (achats, ventes, swaps).")

    with st.spinner("Récupération de l'historique des transactions..."):
        trades = get_bitpanda_trades()
        wallet_txs = get_bitpanda_wallet_txs()

    if not trades and not wallet_txs:
        if st.secrets.get("BITPANDA_API_KEY", ""):
            st.info(
                "Aucune transaction récupérée. Vérifie que ta clé API Bitpanda a les scopes "
                "**READ_TRADES** et **READ_TRANSACTIONS** activés."
            )
        # sinon le warning sur la clé est déjà affiché plus haut
    else:
        pnl = compute_crypto_pnl(trades, wallet_txs, crypto_prices_eur, crypto_balances)
        t_pnl = pnl["total"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Total acheté",
            f"{t_pnl['bought']:,.2f} €",
            help="Somme de tous tes achats crypto (inclut les montants ensuite revendus ou swapés).",
        )
        c2.metric(
            "Total revendu",
            f"{t_pnl['sold']:,.2f} €" if t_pnl["sold"] > 0 else "0,00 €",
            help="Somme de toutes tes ventes crypto (inclut les swaps sortants).",
        )
        c3.metric(
            "Investi net",
            f"{t_pnl['net_invested']:,.2f} €",
            help="Acheté − Revendu : ce que tu as réellement engagé en crypto.",
        )
        c4.metric("Valeur actuelle", f"{t_pnl['current_value']:,.2f} €")
        c5.metric(
            "P&L global",
            f"{t_pnl['pnl']:+,.2f} €",
            delta=f"{t_pnl['pnl_pct']:+.1f}%" if (t_pnl["net_invested"] > 0 or t_pnl["bought"] > 0) else None,
            help="Valeur actuelle + Revendu − Acheté. Le % est calculé sur l'investi net.",
        )

        # Détail par crypto
        with st.expander("📊 Détail par crypto", expanded=True):
            rows = []
            for sym, a in sorted(
                pnl["per_asset"].items(),
                key=lambda x: -(x[1]["current_value"] + x[1]["sold"]),
            ):
                if a["bought"] == 0 and a["sold"] == 0 and a.get("balance", 0) == 0:
                    continue
                rows.append({
                    "Crypto": sym,
                    "Investi (€)": f"{a['bought']:,.2f}",
                    "Revendu (€)": f"{a['sold']:,.2f}" if a["sold"] > 0 else "—",
                    "Solde": f"{a['balance']:.6f}" if a.get("balance", 0) > 0 else "—",
                    "Valeur actuelle (€)": f"{a['current_value']:,.2f}" if a["current_value"] > 0 else "—",
                    "P&L (€)": f"{a['pnl']:+,.2f}",
                    "P&L (%)": f"{a['pnl_pct']:+.1f}%" if a["bought"] > 0 else "—",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Aucune position à afficher.")

        # Historique des transactions (trades + transferts on-chain combinés)
        display_rows = []
        for tr in trades:
            type_lbl = ("🔄 Swap " if tr["is_swap"] else ("🐷 Épargne " if tr["is_savings"] else ""))
            type_lbl += ("📥 Achat" if tr["type"] == "buy" else "📤 Vente")
            display_rows.append({
                "Date": tr["date"],
                "Type": type_lbl,
                "Crypto": tr["symbol"],
                "amount_crypto": tr["amount_crypto"],
                "amount_eur": tr["amount_eur"],
                "price_eur": tr["price_eur"],
            })
        for w in wallet_txs:
            type_lbl = "📨 Dépôt externe" if w["direction"] == "in" else "📩 Retrait externe"
            unit_price = (w["amount_eur"] / w["amount_crypto"]) if w["amount_crypto"] > 0 else 0.0
            display_rows.append({
                "Date": w["date"],
                "Type": type_lbl,
                "Crypto": w["symbol"],
                "amount_crypto": w["amount_crypto"],
                "amount_eur": w["amount_eur"],
                "price_eur": unit_price,
            })
        display_rows.sort(key=lambda x: x["Date"], reverse=True)

        total_txs = len(display_rows)
        n_transfers = len(wallet_txs)
        title_extra = f" · dont {n_transfers} transfert{'s' if n_transfers > 1 else ''} on-chain" if n_transfers else ""
        with st.expander(f"📜 Historique des transactions ({total_txs}{title_extra})", expanded=False):
            df_tx = pd.DataFrame(display_rows)
            df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            df_tx["Quantité"] = df_tx["amount_crypto"].apply(lambda x: f"{x:.6f}")
            df_tx["Montant (€)"] = df_tx["amount_eur"].apply(
                lambda x: f"{x:,.2f}" if x > 0 else "—"
            )
            df_tx["Prix (€)"] = df_tx["price_eur"].apply(
                lambda x: ("—" if x <= 0 else (f"{x:,.4f}" if x < 1 else f"{x:,.2f}"))
            )
            df_tx = df_tx[["Date", "Type", "Crypto", "Quantité", "Montant (€)", "Prix (€)"]]
            st.dataframe(df_tx, use_container_width=True, hide_index=True, height=400)

        # ── Évolution quotidienne du portefeuille ─────────────────────────────
        with st.spinner("Reconstruction de l'historique quotidien..."):
            history = compute_portfolio_history(trades, wallet_txs)

        if not history.empty and len(history) > 1:
            st.divider()
            st.subheader("📈 Évolution quotidienne du portefeuille")
            st.caption(
                "Reconstruction rétroactive : pour chaque jour, solde détenu × prix EUR de clôture. "
                "La ligne pointillée montre ce que tu as engagé (acheté + déposé − revendu − retiré)."
            )

            last_value = float(history["value_eur"].iloc[-1])
            last_net = float(history["net_invested_eur"].iloc[-1])
            last_pnl = float(history["pnl_eur"].iloc[-1])
            ath = float(history["value_eur"].max())
            ath_date = history["value_eur"].idxmax().strftime("%d/%m/%Y")
            atl = float(history["value_eur"][history["value_eur"] > 0].min()) if (history["value_eur"] > 0).any() else 0.0

            # Variation 24h / 7j / 30j (€ et %)
            def _var(days: int):
                if len(history) <= days:
                    return "—", None
                past = float(history["value_eur"].iloc[-(days + 1)])
                if past <= 0:
                    return "—", None
                eur = last_value - past
                pct = eur / past * 100
                return f"{eur:+,.2f} €", f"{pct:+.1f}%"

            v_24h, p_24h = _var(1)
            v_7j,  p_7j  = _var(7)
            v_30j, p_30j = _var(30)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ATH portefeuille", f"{ath:,.2f} €", help=f"Atteint le {ath_date}")
            c2.metric("Variation 24h", v_24h, delta=p_24h)
            c3.metric("Variation 7j",  v_7j,  delta=p_7j)
            c4.metric("Variation 30j", v_30j, delta=p_30j)

    # ── Bloc 2 : Opportunités long terme ─────────────────────────────────────
    st.subheader("🏦 Top 5 — Cryptos sûres (long terme)")
    st.caption("Grandes capitalisations — investissement stable. Classées par score d'achat technique.")
    with st.spinner("Calcul des signaux long terme..."):
        signals_lt = get_crypto_signals("longterm")
    _render_crypto_signals(signals_lt, color_accent="#60a5fa", category="longterm")

    st.divider()

    # ── Bloc 3 : Opportunités court terme ────────────────────────────────────
    st.subheader("🚀 Top 5 — Cryptos volatiles (court terme)")
    st.caption("Petites capitalisations — fort potentiel, risque élevé. Investissement < 5 €.")
    with st.spinner("Calcul des signaux court terme..."):
        signals_st = get_crypto_signals("shortterm")
    _render_crypto_signals(signals_st, color_accent="#a78bfa", category="shortterm")

    st.divider()

    # ── Chargement données BTC ────────────────────────────────────────────────
    with st.spinner("Chargement analyse BTC..."):
        try:
            df = get_btc_data()
        except Exception as e:
            st.error(f"Erreur chargement données BTC : {e}")
            return

    btc_price_eur, price_ts = get_btc_live_price()
    if btc_price_eur and btc_price_eur > 0:
        current_price = btc_price_eur
        price_label = f"Live ~{price_ts.strftime('%H:%M:%S')}"
    else:
        try:
            eur_usd = yf.Ticker("EURUSD=X").fast_info.last_price or 1.08
        except Exception:
            eur_usd = 1.08
        current_price = float(df["Close"].iloc[-1]) / eur_usd
        price_label = "Estimé (EUR)"

    fg_value, fg_label, df_fg = get_fear_greed()
    score, signal_label, reasons = compute_btc_score(df, fear_greed=fg_value)

    btc_balance = crypto_balances.get("BTC", 0.0)

    # Prix moyen d'achat BTC depuis portfolio.json (saisi manuellement)
    btc_avg_price = 0.0
    try:
        from github import Github
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo("YassineZak/xag-advisor")
        f = repo.get_contents("portfolio.json")
        portfolio_file = json.loads(f.decoded_content)
        btc_avg_price = float(portfolio_file.get("btc_avg_price", 0.0))
    except Exception:
        pass

    portfolio_value = btc_balance * current_price
    pnl_eur = (current_price - btc_avg_price) * btc_balance if btc_avg_price > 0 else 0.0
    pnl_pct = ((current_price - btc_avg_price) / btc_avg_price * 100) if btc_avg_price > 0 else 0.0

    # ── Bloc 2 : Métriques BTC ───────────────────────────────────────────────
    st.subheader(f"₿ Bitcoin · {price_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix BTC", f"{current_price:,.0f} €")
    c2.metric("Solde BTC", f"{btc_balance:.6f} BTC" if btc_balance > 0 else "0 BTC")
    c3.metric("Valeur BTC", f"{portfolio_value:,.0f} €" if btc_balance > 0 else "—")
    pnl_display = f"{pnl_eur:+,.0f} € ({pnl_pct:+.1f}%)" if btc_avg_price > 0 else "—"
    c4.metric("P&L BTC", pnl_display, delta=f"{pnl_pct:+.1f}%" if btc_avg_price > 0 else None)

    st.divider()

    # ── Bloc 2 : Signal d'achat ───────────────────────────────────────────────
    COLOR_MAP = {
        "ACHAT FORT": "#22c55e",
        "ACHETER":    "#86efac",
        "NEUTRE":     "#facc15",
        "ATTENDRE":   "#f97316",
        "ÉVITER":     "#ef4444",
    }
    signal_color = COLOR_MAP.get(signal_label, "#94a3b8")

    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:#1e1e2e; border-radius:12px; margin-bottom:1rem;">
        <div style="font-size:3rem; font-weight:bold; color:{signal_color};">{score}</div>
        <div style="font-size:1.4rem; color:{signal_color}; font-weight:600;">{signal_label}</div>
        <div style="color:#94a3b8; font-size:0.85rem;">Score sur 100</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Détail des indicateurs", expanded=True):
        for icon, text in reasons:
            st.markdown(f"{icon} {text}")

    st.divider()

    # ── Bloc 3 : Fear & Greed Index ───────────────────────────────────────────
    st.subheader("😨 Fear & Greed Index")

    if fg_value is not None:
        fg_color = (
            "#22c55e" if fg_value <= 25 else
            "#86efac" if fg_value <= 40 else
            "#facc15" if fg_value <= 60 else
            "#f97316" if fg_value <= 75 else
            "#ef4444"
        )
        st.markdown(f"""
        <div style="text-align:center; padding:15px; background:#1e1e2e; border-radius:10px;">
            <div style="font-size:2.5rem; font-weight:bold; color:{fg_color};">{fg_value}</div>
            <div style="color:{fg_color}; font-weight:600;">{fg_label}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Fear & Greed Index indisponible — API alternative.me inaccessible.")

    st.divider()

    # ── Mise à jour prix moyen BTC ────────────────────────────────────────────
    st.subheader("⚙️ Mettre à jour mon prix moyen BTC")
    st.caption("Ton solde BTC est récupéré automatiquement depuis Bitpanda. Saisis uniquement ton prix moyen d'achat.")

    with st.form("btc_portfolio_form"):
        new_avg_btc = st.number_input(
            "Prix moyen d'achat BTC (€)", min_value=0.0, value=float(btc_avg_price),
            step=100.0, format="%.2f",
            help="Prix moyen auquel tu as acheté ton BTC (en EUR)"
        )
        submitted = st.form_submit_button("Sauvegarder", use_container_width=True)
        if submitted:
            try:
                from github import Github, GithubException
                g = Github(st.secrets["GITHUB_TOKEN"])
                repo = g.get_repo("YassineZak/xag-advisor")
                f = repo.get_contents("portfolio.json")
                portfolio_data = json.loads(f.decoded_content)
                portfolio_data["btc_avg_price"] = new_avg_btc
                import datetime as dt
                portfolio_data["last_updated"] = dt.date.today().strftime("%Y-%m-%d")
                new_content = json.dumps(portfolio_data, indent=2)
                repo.update_file("portfolio.json", "Update BTC avg price", new_content, f.sha)
                st.success("Prix moyen BTC mis à jour !")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Erreur de sauvegarde : {e}")

    # ── Pied de page ──────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Données : Yahoo Finance (~15 min de délai) · Bitpanda API (5 min, fiat + crypto) · "
        "Fear & Greed : alternative.me (1h) · "
        f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} · "
        "⚠️ Outil d'analyse uniquement — pas un conseil financier"
    )
