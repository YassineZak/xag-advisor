"""
Onglet XAG — Silver WisdomTree (via Trade Republic).

Depuis que tout passe par Trade Republic, cet onglet ne suit plus le spot
XAG/EUR ni l'historique Revolut. Il :
  1. lit la position silver WisdomTree dans le dernier relevé Trade Republic
     importé (détection par ISIN JE00B1VS3333),
  2. suit le prix live du WisdomTree Physical Silver en EUR (proxy Yahoo
     `PHAG.MI`, Borsa Italiana, même sous-jacent que sur Trade Republic),
  3. affiche valeur + P&L (prix moyen d'achat éditable),
  4. donne un signal léger achat/attente basé sur le RSI.
"""

import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

import etf_pea_tab


# ── Stockage portefeuille (GitHub) ────────────────────────────────────────────

PORTFOLIO_FILE = "portfolio.json"
REPO_NAME      = "YassineZak/xag-advisor"

# WisdomTree Physical Silver — l'ETC silver détenu sur Trade Republic.
WISDOMTREE_SILVER_ISIN = "JE00B1VS3333"
WISDOMTREE_TICKER      = "PHAG.MI"   # cotation EUR (Borsa Italiana)
WISDOMTREE_NAME        = "WisdomTree Physical Silver"


def load_portfolio() -> dict:
    try:
        from github import Github, GithubException
        g    = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(REPO_NAME)
        f    = repo.get_contents(PORTFOLIO_FILE)
        return json.loads(f.decoded_content)
    except Exception:
        return {}


def save_silver_avg(avg_price: float) -> bool:
    """Enregistre le prix moyen d'achat du silver WisdomTree (champ dédié)."""
    try:
        from github import Github, GithubException
        g    = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(REPO_NAME)
        try:
            f    = repo.get_contents(PORTFOLIO_FILE)
            data = json.loads(f.decoded_content)
            sha  = f.sha
        except GithubException:
            f = None; data = {}; sha = None
        data["silver_avg_price"] = float(avg_price)
        data["last_updated"]     = datetime.today().strftime("%Y-%m-%d")
        content = json.dumps(data, indent=2, ensure_ascii=False)
        if sha:
            repo.update_file(PORTFOLIO_FILE, "Update silver avg price", content, sha)
        else:
            repo.create_file(PORTFOLIO_FILE, "Create portfolio", content)
        return True
    except Exception as e:
        st.error(f"Erreur de sauvegarde : {e}")
        return False


# ── Indicateur de base ────────────────────────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── Position silver depuis le relevé Trade Republic ───────────────────────────

def get_silver_holding() -> dict:
    """
    Cherche la ligne silver WisdomTree dans le dernier relevé Trade Republic
    importé. Détection prioritaire par ISIN, repli par nom.
    Retourne {found, qty, snapshot_price, snapshot_value, avg_price}.
    """
    out = {"found": False, "qty": 0.0, "snapshot_price": 0.0,
           "snapshot_value": 0.0, "avg_price": 0.0}
    try:
        portfolio = etf_pea_tab.load_tr_portfolio() or {}
        holdings  = portfolio.get("holdings", []) or []
        target    = WISDOMTREE_SILVER_ISIN.upper()

        match = None
        for h in holdings:
            if not isinstance(h, dict):
                continue
            isin = str(h.get("isin", "")).strip().upper()
            name = str(h.get("name", "")).lower()
            if isin == target:
                match = h
                break
            if match is None and ("silver" in name or "argent" in name):
                match = h  # repli : on garde le premier candidat par nom

        if match is not None:
            qty   = float(match.get("qty", 0) or 0)
            sval  = float(match.get("snapshot_value", 0) or 0)
            sprice = float(match.get("snapshot_price", 0) or 0)
            if sprice <= 0 and qty > 0 and sval > 0:
                sprice = sval / qty
            out.update(found=True, qty=qty, snapshot_price=sprice, snapshot_value=sval)
    except Exception:
        pass

    out["avg_price"] = float(load_portfolio().get("silver_avg_price", 0) or 0)
    return out


# ── Prix live & historique WisdomTree silver ──────────────────────────────────

@st.cache_data(ttl=300)
def get_silver_live() -> tuple:
    """Prix live du WisdomTree Physical Silver en EUR. Retourne (price, ts)."""
    try:
        price = yf.Ticker(WISDOMTREE_TICKER).fast_info.last_price
        if price and price > 0:
            return float(price), datetime.now()
    except Exception:
        pass
    try:
        hist = yf.Ticker(WISDOMTREE_TICKER).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]), datetime.now()
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=900)
def get_silver_history(days: int = 400) -> pd.Series:
    """Série de clôtures EUR du WisdomTree silver (index tz-naïf)."""
    try:
        hist = yf.Ticker(WISDOMTREE_TICKER).history(period=f"{days}d")
        if not hist.empty and "Close" in hist.columns:
            s = hist["Close"].copy()
            s.index = s.index.tz_localize(None)
            return s.dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def compute_silver_signal(hist: pd.Series, price: float) -> dict:
    """Signal léger achat/attente basé sur RSI + position vs moyennes mobiles."""
    out = {"rsi": None, "label": "—", "color": "#94a3b8", "reasons": [],
           "sma20": None, "sma50": None, "perf_1m": None}
    if hist is None or len(hist) < 20:
        out["reasons"].append(("ℹ️", "Historique insuffisant pour le signal."))
        return out

    rsi_val = float(rsi(hist).iloc[-1])
    sma20   = float(hist.rolling(20).mean().iloc[-1])
    sma50   = float(hist.rolling(50).mean().iloc[-1]) if len(hist) >= 50 else None
    ref     = price if (price and price > 0) else float(hist.iloc[-1])
    perf_1m = None
    if len(hist) > 21:
        past = float(hist.iloc[-22])
        if past > 0:
            perf_1m = (ref - past) / past * 100

    out.update(rsi=rsi_val, sma20=sma20, sma50=sma50, perf_1m=perf_1m)

    # RSI
    if rsi_val < 30:
        out["reasons"].append(("💚", f"RSI {rsi_val:.0f} — survente marquée, repli probablement excessif."))
    elif rsi_val < 40:
        out["reasons"].append(("🟢", f"RSI {rsi_val:.0f} — sous la zone neutre, point d'entrée correct."))
    elif rsi_val > 70:
        out["reasons"].append(("🔴", f"RSI {rsi_val:.0f} — surachat, mieux vaut attendre un repli."))
    elif rsi_val > 60:
        out["reasons"].append(("🟠", f"RSI {rsi_val:.0f} — au-dessus de la zone neutre."))
    else:
        out["reasons"].append(("🟡", f"RSI {rsi_val:.0f} — zone neutre."))

    # Position vs SMA20
    if ref < sma20:
        out["reasons"].append(("💚", f"Prix sous la moyenne 20 jours ({sma20:.2f} €) — en solde vs tendance récente."))
    else:
        out["reasons"].append(("🟠", f"Prix au-dessus de la moyenne 20 jours ({sma20:.2f} €)."))

    if perf_1m is not None:
        icon = "🟢" if perf_1m < 0 else "🟠"
        out["reasons"].append((icon, f"Performance 1 mois : {perf_1m:+.1f}%."))

    # Label de synthèse
    score = 0
    if rsi_val < 40: score += 1
    if rsi_val < 30: score += 1
    if ref < sma20:  score += 1
    if rsi_val > 70: score -= 2
    elif rsi_val > 60: score -= 1

    if score >= 2:
        out["label"], out["color"] = "BON POINT D'ENTRÉE", "#22c55e"
    elif score <= -1:
        out["label"], out["color"] = "ATTENDRE", "#f97316"
    else:
        out["label"], out["color"] = "NEUTRE", "#facc15"
    return out


# ── Rendu de l'onglet ─────────────────────────────────────────────────────────

def render():
    """Point d'entrée du tab XAG — appelé par app.py."""
    col_title, col_refresh = st.columns([5, 1])
    col_title.title("🥈 Silver WisdomTree")
    if col_refresh.button("🔄 Rafraîchir", use_container_width=True,
                          help="Force le rechargement du prix WisdomTree"):
        st.cache_data.clear()
        st.rerun()

    holding = get_silver_holding()
    price, ts = get_silver_live()
    hist = get_silver_history()

    if price is None and not hist.empty:
        price = float(hist.iloc[-1])

    if price is None:
        st.error("Prix WisdomTree indisponible (Yahoo Finance injoignable). Réessaie plus tard.")
        return

    price_label = f"Live ~{ts.strftime('%H:%M:%S')}" if ts else "Dernière clôture"

    # ── 1. Position & valeur ──────────────────────────────────────────────────
    qty       = holding["qty"]
    avg_price = holding["avg_price"]

    if not holding["found"]:
        st.info(
            "Aucune position silver détectée dans ton relevé Trade Republic. "
            "Importe un relevé dans l'onglet **ETF & Actions PEA** "
            f"(la ligne **{WISDOMTREE_NAME}**, ISIN `{WISDOMTREE_SILVER_ISIN}`)."
        )
    else:
        current_val = qty * price
        invested    = qty * avg_price if avg_price > 0 else 0.0
        pnl         = current_val - invested if invested > 0 else 0.0
        pnl_pct     = (pnl / invested * 100) if invested > 0 else 0.0

        st.subheader("💼 Ma position silver")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parts détenues", f"{qty:g}",
                  help="Nombre de parts WisdomTree Physical Silver (depuis ton relevé Trade Republic).")
        c2.metric(f"Prix WisdomTree · {price_label}", f"{price:.2f} €",
                  help=f"Cotation EUR du {WISDOMTREE_NAME} (proxy Yahoo {WISDOMTREE_TICKER}).")
        c3.metric("Valeur actuelle", f"{current_val:,.2f} €", help="Parts × prix WisdomTree actuel.")
        if avg_price > 0:
            c4.metric("P&L total", f"{pnl:+,.2f} €", f"{pnl_pct:+.2f}%",
                      help="Gain/perte vs ton prix moyen d'achat.")
        else:
            c4.metric("P&L total", "—", help="Saisis ton prix moyen d'achat plus bas pour l'activer.")

    st.divider()

    # ── 2. Signal léger ───────────────────────────────────────────────────────
    sig = compute_silver_signal(hist, price)

    col_sig, col_why = st.columns([1, 2])
    with col_sig:
        st.markdown(f"""
        <div style="text-align:center;padding:18px;background:#1e1e2e;border-radius:12px">
            <div style="font-size:1.5rem;font-weight:bold;color:{sig['color']}">{sig['label']}</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px">Signal indicatif (RSI)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_why:
        st.markdown("**Lecture du moment :**")
        for icon, text in sig["reasons"]:
            st.markdown(f"{icon} {text}")

    st.divider()

    # ── 3. Repères de prix ────────────────────────────────────────────────────
    st.subheader("📌 Repères de prix")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("RSI 14j", f"{sig['rsi']:.0f}" if sig["rsi"] is not None else "—")
    rc2.metric("Moyenne 20j", f"{sig['sma20']:.2f} €" if sig["sma20"] else "—")
    rc3.metric("Moyenne 50j", f"{sig['sma50']:.2f} €" if sig["sma50"] else "—")
    if not hist.empty:
        hi = float(hist.tail(252).max()); lo = float(hist.tail(252).min())
        rc4.metric("Plage 12 mois", f"{lo:.0f} – {hi:.0f} €")
    else:
        rc4.metric("Plage 12 mois", "—")

    st.divider()

    # ── 4. Prix moyen d'achat ─────────────────────────────────────────────────
    st.subheader("⚙️ Mon prix moyen d'achat")
    st.caption(
        "Ton solde de parts est lu automatiquement depuis ton relevé Trade Republic. "
        "Saisis seulement ton prix moyen d'achat (par part) pour activer le calcul du P&L."
    )
    with st.form("silver_avg_form"):
        new_avg = st.number_input(
            "Prix moyen d'achat (€ / part)", min_value=0.0, value=float(avg_price),
            step=0.01, format="%.2f",
        )
        if st.form_submit_button("Sauvegarder", use_container_width=True):
            if save_silver_avg(new_avg):
                st.success("Prix moyen enregistré !")
                st.cache_data.clear()
                st.rerun()

    st.caption(
        f"Source prix : Yahoo Finance {WISDOMTREE_TICKER} (~15 min de délai) · "
        f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} · "
        "⚠️ Outil d'analyse — pas un conseil financier."
    )
