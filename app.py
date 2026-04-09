import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="XAG Advisor",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Données ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_data():
    end = datetime.today()
    start = end - timedelta(days=400)

    raw_xag = yf.download("XAGUSD=X", start=start, end=end, progress=False)
    raw_eur = yf.download("EURUSD=X",  start=start, end=end, progress=False)
    raw_xau = yf.download("XAUUSD=X", start=start, end=end, progress=False)

    df = pd.DataFrame({
        "XAG_USD": raw_xag["Close"].squeeze(),
        "EUR_USD": raw_eur["Close"].squeeze(),
        "XAU_USD": raw_xau["Close"].squeeze(),
    }).dropna()

    df["XAG_EUR"] = df["XAG_USD"] / df["EUR_USD"]
    df["XAU_EUR"] = df["XAU_USD"] / df["EUR_USD"]
    df["ratio"]   = df["XAU_EUR"] / df["XAG_EUR"]
    return df


# ── Indicateurs ──────────────────────────────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + 2 * std, sma - 2 * std


# ── Score d'achat ─────────────────────────────────────────────────────────────

def compute_score(df):
    price   = df["XAG_EUR"].iloc[-1]
    rsi_val = rsi(df["XAG_EUR"]).iloc[-1]
    sma20_v, upper_v, lower_v = bollinger(df["XAG_EUR"])
    sma20   = sma20_v.iloc[-1]
    upper   = upper_v.iloc[-1]
    lower   = lower_v.iloc[-1]
    sma50   = df["XAG_EUR"].rolling(50).mean().iloc[-1]
    ratio   = df["ratio"].iloc[-1]
    ratio_m = df["ratio"].mean()
    perf_1m = (price - df["XAG_EUR"].iloc[-22]) / df["XAG_EUR"].iloc[-22] * 100

    score   = 10  # base neutre
    reasons = []

    # RSI
    if rsi_val < 30:
        score += 30
        reasons.append(("✅", f"RSI très bas ({rsi_val:.1f}) — zone de survente forte"))
    elif rsi_val < 45:
        score += 15
        reasons.append(("✅", f"RSI bas ({rsi_val:.1f}) — légère survente"))
    elif rsi_val > 70:
        score -= 20
        reasons.append(("🔴", f"RSI élevé ({rsi_val:.1f}) — zone de surachat"))
    else:
        reasons.append(("➡️", f"RSI neutre ({rsi_val:.1f})"))

    # Bollinger
    bb_pos = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    if bb_pos < 0.2:
        score += 25
        reasons.append(("✅", f"Prix proche de la bande basse Bollinger ({bb_pos:.0%}) — signal d'achat"))
    elif bb_pos > 0.8:
        score -= 15
        reasons.append(("🔴", f"Prix proche de la bande haute Bollinger ({bb_pos:.0%}) — attention"))
    else:
        reasons.append(("➡️", f"Prix dans la bande médiane Bollinger ({bb_pos:.0%})"))

    # Moyennes mobiles
    if price < sma50 and price < sma20:
        score += 15
        reasons.append(("✅", f"Prix sous SMA20 ({sma20:.2f}€) et SMA50 ({sma50:.2f}€) — bon point d'entrée potentiel"))
    elif price > sma50 and price > sma20:
        score -= 10
        reasons.append(("🔴", f"Prix au-dessus de SMA20 et SMA50 — prudence"))
    else:
        reasons.append(("➡️", f"Prix entre les moyennes mobiles"))

    # Ratio or/argent
    if ratio > ratio_m * 1.1:
        score += 20
        reasons.append(("✅", f"Ratio or/argent élevé ({ratio:.1f} vs moy. {ratio_m:.1f}) — argent bon marché vs l'or"))
    elif ratio < ratio_m * 0.9:
        score -= 10
        reasons.append(("🔴", f"Ratio or/argent bas ({ratio:.1f}) — argent cher vs l'or"))
    else:
        reasons.append(("➡️", f"Ratio or/argent normal ({ratio:.1f} | moy. {ratio_m:.1f})"))

    # Performance 1 mois
    if perf_1m < -5:
        score += 10
        reasons.append(("✅", f"Baisse mensuelle de {perf_1m:.1f}% — possible opportunité de racheter bas"))
    elif perf_1m > 10:
        score -= 5
        reasons.append(("⚠️", f"Forte hausse mensuelle +{perf_1m:.1f}% — momentum haussier mais prix élevé"))
    else:
        reasons.append(("➡️", f"Performance mensuelle neutre ({perf_1m:+.1f}%)"))

    score = max(0, min(100, score))
    return score, reasons, rsi_val


# ── Interface ─────────────────────────────────────────────────────────────────

st.title("🥈 XAG/EUR — Advisor Mensuel")

with st.spinner("Chargement des données marché..."):
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        st.stop()

price      = df["XAG_EUR"].iloc[-1]
prev_price = df["XAG_EUR"].iloc[-2]
change_pct = (price - prev_price) / prev_price * 100
ratio      = df["ratio"].iloc[-1]
rsi_now    = rsi(df["XAG_EUR"]).iloc[-1]

# ── Métriques ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Prix XAG/EUR",    f"{price:.2f} €",   f"{change_pct:+.2f}%")
col2.metric("Ratio Or/Argent", f"{ratio:.1f}",      help="Plus ce ratio est élevé, plus l'argent est bon marché vs l'or. Historiquement ~70-80.")
col3.metric("RSI 14j",         f"{rsi_now:.1f}",    help="<30 = survente (achat intéressant) | >70 = surachat (attendre)")

st.divider()

# ── Signal ────────────────────────────────────────────────────────────────────
score, reasons, _ = compute_score(df)

col_sig, col_why = st.columns([1, 2])

with col_sig:
    if score >= 60:
        st.success(f"### 🟢 ACHETER\nScore : **{score} / 100**")
    elif score >= 40:
        st.warning(f"### 🟡 NEUTRE\nScore : **{score} / 100**")
    else:
        st.error(f"### 🔴 ATTENDRE\nScore : **{score} / 100**")

    st.progress(score / 100)
    st.caption("Score basé sur 5 indicateurs techniques et fondamentaux")

with col_why:
    st.markdown("**Analyse détaillée :**")
    for icon, text in reasons:
        st.markdown(f"{icon} {text}")

st.divider()

# ── Graphique prix ────────────────────────────────────────────────────────────
st.subheader("📈 Prix XAG/EUR — 6 mois")

df_chart         = df.tail(180).copy()
sma20_s, ub, lb  = bollinger(df_chart["XAG_EUR"])
sma50_s          = df_chart["XAG_EUR"].rolling(50).mean()
rsi_s            = rsi(df_chart["XAG_EUR"])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.06,
    subplot_titles=("Prix XAG/EUR + Bollinger + Moyennes mobiles", "RSI (14 jours)"),
)

# Bandes de Bollinger (zone grisée)
fig.add_trace(go.Scatter(
    x=df_chart.index, y=ub, name="BB Sup",
    line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df_chart.index, y=lb, name="BB Inf",
    fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
    line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)

# Prix
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["XAG_EUR"],
    name="XAG/EUR", line=dict(color="#c0c0c0", width=2.5)), row=1, col=1)

# SMA
fig.add_trace(go.Scatter(
    x=df_chart.index, y=sma20_s,
    name="SMA 20", line=dict(color="#4da6ff", dash="dot", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df_chart.index, y=sma50_s,
    name="SMA 50", line=dict(color="#ff9f4a", dash="dot", width=1.5)), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_chart.index, y=rsi_s,
    name="RSI", line=dict(color="#b48ead", width=2)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", row=2, col=1)

fig.update_yaxes(title_text="€ / oz", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_layout(
    height=560,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ── Ratio Or/Argent ───────────────────────────────────────────────────────────
st.subheader("📊 Ratio Or / Argent — 6 mois")
st.caption("Un ratio élevé signifie que l'argent est historiquement bon marché par rapport à l'or → signal d'achat supplémentaire")

fig2 = go.Figure()
fig2.add_hrect(
    y0=df["ratio"].quantile(0.75), y1=df["ratio"].max() + 5,
    fillcolor="rgba(0,200,0,0.07)", annotation_text="Zone favorable argent",
    annotation_position="top left",
)
fig2.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["ratio"],
    name="Ratio XAU/XAG", line=dict(color="gold", width=2)))
fig2.add_hline(
    y=df["ratio"].mean(), line_dash="dash", line_color="white",
    annotation_text=f"Moy. 1 an ({df['ratio'].mean():.1f})",
    annotation_position="bottom right")
fig2.update_layout(
    height=280, template="plotly_dark",
    margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Pied de page ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Données : Yahoo Finance · Actualisé toutes les heures · "
    f"Dernière mise à jour : {datetime.today().strftime('%d/%m/%Y %H:%M')} · "
    f"⚠️ Outil d'aide à la décision uniquement — pas un conseil financier"
)
