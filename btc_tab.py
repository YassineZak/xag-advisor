import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# ── Soldes Bitpanda ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_bitpanda_balances() -> dict:
    """
    Récupère tous les soldes crypto depuis l'API Bitpanda.
    Retourne un dict {symbol: balance} pour les wallets avec solde > 0.
    Cache 5 minutes.
    """
    try:
        api_key = st.secrets.get("BITPANDA_API_KEY", "")
        if not api_key:
            return {}
        resp = requests.get(
            "https://api.bitpanda.com/v1/wallets",
            headers={"X-API-KEY": api_key},
            timeout=10
        )
        resp.raise_for_status()
        wallets = resp.json().get("data", [])
        balances = {}
        for w in wallets:
            attrs = w.get("attributes", {})
            symbol = attrs.get("cryptocoin_symbol", "")
            balance = float(attrs.get("balance", 0))
            if balance > 0:
                balances[symbol] = balances.get(symbol, 0) + balance
        return balances
    except Exception:
        return {}


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
    """Point d'entrée du tab Bitcoin — appelé par app.py."""

    # ── Titre & bouton refresh ────────────────────────────────────────────────
    col_title, col_refresh = st.columns([5, 1])
    col_title.title("₿ Bitcoin (BTC)")
    if col_refresh.button("🔄 Rafraîchir", key="btc_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Chargement des données ────────────────────────────────────────────────
    with st.spinner("Chargement des données BTC..."):
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
        # Fallback : convertir USD→EUR approximativement
        try:
            eur_usd = yf.Ticker("EURUSD=X").fast_info.last_price or 1.08
        except Exception:
            eur_usd = 1.08
        current_price = float(df["Close"].iloc[-1]) / eur_usd
        price_label = "Estimé (EUR)"

    fg_value, fg_label, df_fg = get_fear_greed()
    score, signal_label, reasons = compute_btc_score(df, fear_greed=fg_value)

    # Solde BTC depuis Bitpanda (live, pas de Pi ni GitHub Actions)
    bitpanda_balances = get_bitpanda_balances()
    btc_balance = bitpanda_balances.get("BTC", 0.0)

    # Prix moyen d'achat depuis portfolio.json (saisi manuellement)
    btc_avg_price = 0.0
    try:
        from github import Github
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo("YassineZak/xag-advisor")
        f = repo.get_contents("portfolio.json")
        portfolio = json.loads(f.decoded_content)
        btc_avg_price = float(portfolio.get("btc_avg_price", 0.0))
    except Exception:
        pass

    portfolio_value = btc_balance * current_price
    pnl_eur = (current_price - btc_avg_price) * btc_balance if btc_avg_price > 0 else 0.0
    pnl_pct = ((current_price - btc_avg_price) / btc_avg_price * 100) if btc_avg_price > 0 else 0.0

    # ── Bloc 1 : Métriques temps réel ────────────────────────────────────────
    st.subheader(f"Prix actuel · {price_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix BTC", f"{current_price:,.0f} €")
    c2.metric("Solde BTC", f"{btc_balance:.6f} BTC" if btc_balance > 0 else "Non configuré")
    c3.metric("Valeur portefeuille", f"{portfolio_value:,.0f} €" if btc_balance > 0 else "—")
    pnl_display = f"{pnl_eur:+,.0f} € ({pnl_pct:+.1f}%)" if btc_avg_price > 0 else "—"
    c4.metric("P&L", pnl_display, delta=f"{pnl_pct:+.1f}%" if btc_avg_price > 0 else None)

    if btc_balance == 0.0:
        if not st.secrets.get("BITPANDA_API_KEY", ""):
            st.warning("⚠️ Clé API Bitpanda non configurée — ajoute BITPANDA_API_KEY dans les secrets Streamlit.")
        else:
            st.info("ℹ️ Solde BTC à 0 sur Bitpanda, ou wallet non trouvé.")

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
        col_fg, col_fg_chart = st.columns([1, 3])
        col_fg.markdown(f"""
        <div style="text-align:center; padding:15px; background:#1e1e2e; border-radius:10px;">
            <div style="font-size:2.5rem; font-weight:bold; color:{fg_color};">{fg_value}</div>
            <div style="color:{fg_color}; font-weight:600;">{fg_label}</div>
        </div>
        """, unsafe_allow_html=True)

        if not df_fg.empty:
            fig_fg = go.Figure()
            fig_fg.add_trace(go.Scatter(
                x=df_fg.index, y=df_fg["value"],
                mode="lines+markers", line=dict(color="#facc15", width=2),
                fill="tozeroy", fillcolor="rgba(250,204,21,0.1)",
                name="Fear & Greed"
            ))
            fig_fg.add_hline(y=25, line_dash="dash", line_color="#22c55e", annotation_text="Extreme Fear")
            fig_fg.add_hline(y=75, line_dash="dash", line_color="#ef4444", annotation_text="Extreme Greed")
            fig_fg.update_layout(
                template="plotly_dark", height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False, yaxis=dict(range=[0, 100]),
            )
            col_fg_chart.plotly_chart(fig_fg, use_container_width=True)
    else:
        st.warning("Fear & Greed Index indisponible — API alternative.me inaccessible.")

    st.divider()

    # ── Bloc 4 : Charts techniques ────────────────────────────────────────────
    st.subheader("📈 Analyse technique BTC — 6 mois")
    df_6m = df.tail(180).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03,
        subplot_titles=("Prix BTC/EUR + Bollinger + Moyennes mobiles", "RSI (14 jours)"),
    )

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_upper"], name="BB Upper",
        line=dict(color="rgba(148,163,184,0.4)", dash="dot", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_lower"], name="BB Lower",
        line=dict(color="rgba(148,163,184,0.4)", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(148,163,184,0.06)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_mid"], name="BB Mid",
        line=dict(color="rgba(148,163,184,0.3)", width=1), showlegend=False), row=1, col=1)
    # SMAs
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["SMA20"], name="SMA20",
        line=dict(color="#60a5fa", dash="dot", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["SMA50"], name="SMA50",
        line=dict(color="#f59e0b", dash="dot", width=1.5)), row=1, col=1)
    # Prix
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["Close"], name="BTC/USD",
        line=dict(color="#f97316", width=2.5)), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["RSI"], name="RSI 14",
        line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=500,
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Bloc 5 : Évolution du portefeuille ────────────────────────────────────
    if btc_balance > 0 and btc_avg_price > 0:
        st.subheader("💼 Évolution du portefeuille BTC")
        try:
            eur_usd_hist = yf.Ticker("EURUSD=X").history(period="400d")["Close"]
            eur_usd_hist.index = eur_usd_hist.index.tz_localize(None)
            df_port = df[["Close"]].join(eur_usd_hist.rename("EUR_USD"), how="left")
            df_port["EUR_USD"] = df_port["EUR_USD"].ffill().fillna(1.08)
            df_port["Close_EUR"] = df_port["Close"] / df_port["EUR_USD"]
        except Exception:
            df_port = df.copy()
            df_port["Close_EUR"] = df_port["Close"] / 1.08

        df_port["value_eur"] = df_port["Close_EUR"] * btc_balance
        purchase_value = btc_avg_price * btc_balance

        fig_port = go.Figure()
        fig_port.add_hline(
            y=purchase_value,
            line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Prix moyen {btc_avg_price:,.0f} €",
            annotation_position="bottom right",
        )
        fig_port.add_trace(go.Scatter(
            x=df_port.index, y=df_port["value_eur"],
            mode="lines", line=dict(color="#f97316", width=2),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
            name="Valeur portefeuille BTC (€)"
        ))
        fig_port.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_port, use_container_width=True)

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
        "Données : Yahoo Finance (~15 min de délai) · Bitpanda API (5 min) · "
        "Fear & Greed : alternative.me (1h) · "
        f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} · "
        "⚠️ Outil d'analyse uniquement — pas un conseil financier"
    )
