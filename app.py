import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from github import Github, GithubException

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


# ── Authentification ──────────────────────────────────────────────────────────

def check_auth():
    if st.session_state.get("authenticated"):
        return
    st.title("🥈 XAG Advisor")
    with st.form("login_form"):
        pwd = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Connexion", use_container_width=True)
        if submitted:
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect")
    st.stop()

check_auth()


# ── Portfolio GitHub ──────────────────────────────────────────────────────────

PORTFOLIO_FILE = "portfolio.json"
REPO_NAME      = "YassineZak/xag-advisor"

def load_portfolio():
    try:
        g    = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(REPO_NAME)
        f    = repo.get_contents(PORTFOLIO_FILE)
        return json.loads(f.decoded_content)
    except GithubException:
        return {"quantity": 0.0, "avg_price": 0.0}

def save_portfolio(quantity: float, avg_price: float):
    g       = Github(st.secrets["GITHUB_TOKEN"])
    repo    = g.get_repo(REPO_NAME)
    content = json.dumps({"quantity": quantity, "avg_price": avg_price}, indent=2)
    try:
        f = repo.get_contents(PORTFOLIO_FILE)
        repo.update_file(PORTFOLIO_FILE, "Update portfolio", content, f.sha)
    except GithubException:
        repo.create_file(PORTFOLIO_FILE, "Create portfolio", content)


# ── Données marché ────────────────────────────────────────────────────────────
#
# On récupère 400 jours d'historique pour 7 actifs via yfinance (gratuit).
# Chaque actif a un ticker principal + un fallback en cas d'échec.
#
# Actifs récupérés :
#   XAG_USD  → argent spot en USD       (ticker : XAGUSD=X  / SI=F futures)
#   EUR_USD  → taux de change EUR/USD   (ticker : EURUSD=X  / EUR=X)
#   XAU_USD  → or spot en USD           (ticker : XAUUSD=X  / GC=F futures)
#   SPY      → ETF S&P 500 en USD       (ticker : SPY)
#   GLD      → ETF Or en USD            (ticker : GLD — utilisé pour RS vs XAG)
#   TLT      → ETF obligations 20+ ans  (ticker : TLT)
#   DXY      → indice Dollar US         (ticker : DX-Y.NYB  / DX=F)

@st.cache_data(ttl=3600)
def get_data():
    TICKERS = {
        "XAG_USD": ["XAGUSD=X", "SI=F"],
        "EUR_USD":  ["EURUSD=X", "EUR=X"],
        "XAU_USD":  ["XAUUSD=X", "GC=F"],
        "SPY":      ["SPY"],           # S&P 500 ETF (USD)
        "GLD":      ["GLD"],           # Gold ETF (USD) — pour force relative vs XAG
        "TLT":      ["TLT"],           # Obligations long terme (USD)
        "DXY":      ["DX-Y.NYB", "DX=F"],  # Dollar Index
    }

    def fetch_close(candidates):
        """Essaie chaque ticker de la liste jusqu'à obtenir des données valides."""
        for ticker in candidates:
            try:
                hist = yf.Ticker(ticker).history(period="400d")
                if not hist.empty and "Close" in hist.columns and len(hist) > 30:
                    s = hist["Close"].copy()
                    # Supprime le timezone pour éviter les conflits de merge
                    s.index = s.index.tz_localize(None)
                    return s
            except Exception:
                continue
        return pd.Series(dtype=float, name=candidates[0])

    # Téléchargement parallèle de tous les actifs
    raw = {key: fetch_close(tickers) for key, tickers in TICKERS.items()}

    # On signale dans les logs quels tickers ont échoué
    missing = [k for k, v in raw.items() if v.empty]
    if missing:
        st.warning(f"Données manquantes pour : {', '.join(missing)} — certains graphiques peuvent être incomplets.")

    df = pd.DataFrame(raw)

    # On supprime uniquement les colonnes critiques pour le calcul principal
    critical = ["XAG_USD", "EUR_USD", "XAU_USD"]
    df = df.dropna(subset=critical)

    if len(df) < 30:
        raise ValueError(f"Données insuffisantes ({len(df)} lignes) — réessaie dans quelques minutes.")

    # ── Conversions en EUR ──────────────────────────────────────────────────
    df["XAG_EUR"] = df["XAG_USD"] / df["EUR_USD"]   # prix argent en €/oz
    df["XAU_EUR"] = df["XAU_USD"] / df["EUR_USD"]   # prix or en €/oz

    # ── Ratios de force relative (tous en USD pour comparabilité) ───────────
    # Ratio or/argent : nombre d'oz d'argent pour acheter 1 oz d'or
    # → plus il est élevé, plus l'argent est bon marché vs l'or
    df["ratio_xau_xag"] = df["XAU_USD"] / df["XAG_USD"]

    return df


# ── Indicateurs techniques de base ───────────────────────────────────────────

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


# ── Calculs avancés : Force Relative & Sharpe ─────────────────────────────────

# Taux sans risque journalier utilisé pour le Sharpe.
# On utilise ~4.5% annualisé (taux court terme zone euro ~2025).
# → 0.045 / 252 jours de bourse = ~0.000179 par jour.
# Tu peux ajuster RF_DAILY pour refléter les taux actuels.
RF_DAILY = 0.045 / 252

def force_relative(xag: pd.Series, other: pd.Series) -> pd.Series:
    """
    Force Relative (RS) normalisée entre XAG et un autre actif.

    Méthode :
      1. On normalise chaque série à 100 à leur point de départ commun.
         normalized_XAG   = XAG   / XAG[0]   * 100
         normalized_other = other / other[0]  * 100
      2. RS = normalized_XAG / normalized_other

    Interprétation :
      RS > 1.0  → XAG surperforme l'actif de référence sur la période
      RS = 1.0  → performance identique
      RS < 1.0  → XAG sous-performe l'actif de référence

    Note : on aligne les deux séries sur l'index commun avant le calcul.
    """
    common = xag.index.intersection(other.index)
    x = xag.loc[common]
    o = other.loc[common]

    # Normalisation : chaque série rebase à 1.0 au premier point disponible
    x_norm = x / x.iloc[0]
    o_norm = o / o.iloc[0]

    return x_norm / o_norm


def sharpe_rolling(series: pd.Series, window: int) -> pd.Series:
    """
    Ratio de Sharpe glissant annualisé sur `window` jours de bourse.

    Formule :
      rendements   = pct_change() de la série de prix
      excess_ret   = rendements - RF_DAILY  (rendement au-dessus du taux sans risque)
      sharpe       = mean(excess_ret) / std(excess_ret) * sqrt(252)

    Le facteur sqrt(252) annualise le Sharpe (252 = nb jours de bourse/an).

    Interprétation :
      Sharpe > 1.0  → bon rendement ajusté au risque
      Sharpe > 2.0  → excellent
      Sharpe < 0    → perte en termes réels (sous le taux sans risque)

    Paramètres ajustables :
      window  → fenêtre glissante (30 = court terme, 90 = moyen terme)
      RF_DAILY → taux sans risque journalier (constante en haut du fichier)
    """
    returns     = series.pct_change()
    excess      = returns - RF_DAILY
    rolling_mean = excess.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()

    # On évite la division par zéro sur les périodes plates
    sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    return sharpe


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule tous les indicateurs avancés à partir du DataFrame principal.
    Retourne un dictionnaire de valeurs scalaires (dernière valeur de chaque série).
    """
    ind = {}

    # ── Force Relative (sur les 90 derniers jours) ──────────────────────────
    # On prend 90j pour capter la tendance moyen terme sans trop de bruit
    window_rs = 90
    xag = df["XAG_USD"].tail(window_rs)

    if "SPY" in df.columns and df["SPY"].notna().sum() > 30:
        rs_spy_series     = force_relative(xag, df["SPY"].tail(window_rs))
        ind["rs_spy"]     = rs_spy_series.iloc[-1]       # valeur actuelle
        ind["rs_spy_s"]   = rs_spy_series                # série complète pour graphique
    else:
        ind["rs_spy"] = None

    if "GLD" in df.columns and df["GLD"].notna().sum() > 30:
        rs_gld_series     = force_relative(xag, df["GLD"].tail(window_rs))
        ind["rs_gld"]     = rs_gld_series.iloc[-1]
        ind["rs_gld_s"]   = rs_gld_series
    else:
        ind["rs_gld"] = None

    if "TLT" in df.columns and df["TLT"].notna().sum() > 30:
        rs_tlt_series     = force_relative(xag, df["TLT"].tail(window_rs))
        ind["rs_tlt"]     = rs_tlt_series.iloc[-1]
        ind["rs_tlt_s"]   = rs_tlt_series
    else:
        ind["rs_tlt"] = None

    # ── Sharpe Ratio glissant (30j et 90j) pour XAG et SPY ─────────────────
    ind["sharpe_xag_30"] = sharpe_rolling(df["XAG_USD"], 30).iloc[-1]
    ind["sharpe_xag_90"] = sharpe_rolling(df["XAG_USD"], 90).iloc[-1]

    if "SPY" in df.columns and df["SPY"].notna().sum() > 30:
        ind["sharpe_spy_30"] = sharpe_rolling(df["SPY"], 30).iloc[-1]
        ind["sharpe_spy_90"] = sharpe_rolling(df["SPY"], 90).iloc[-1]
    else:
        ind["sharpe_spy_30"] = None
        ind["sharpe_spy_90"] = None

    # ── Momentum DXY sur 1 mois (22 jours de bourse) ────────────────────────
    # Un dollar en hausse pénalise les métaux (corrélation négative historique)
    if "DXY" in df.columns and df["DXY"].notna().sum() > 25:
        dxy = df["DXY"].dropna()
        ind["dxy_mom_1m"] = (dxy.iloc[-1] - dxy.iloc[-22]) / dxy.iloc[-22] * 100
    else:
        ind["dxy_mom_1m"] = None

    return ind


# ── Score d'achat enrichi ─────────────────────────────────────────────────────

def compute_score(df: pd.DataFrame, ind: dict):
    """
    Calcule un score d'achat 0–100 en croisant indicateurs techniques (RSI,
    Bollinger, moyennes mobiles, ratio XAU/XAG) et indicateurs avancés
    (Force Relative vs SPY, Sharpe comparé, momentum DXY).

    Pondérations (ajustables) :
      RSI           : ±20 pts
      Bollinger     : ±25 pts
      SMA 20/50     : ±15 pts
      Ratio XAU/XAG : ±20 pts
      Perf 1 mois   : ±10 pts
      Force Rel SPY : ±15 pts   ← nouveau
      Sharpe XAG/SPY: ±10 pts   ← nouveau
      DXY momentum  : ±15 pts   ← nouveau
    """
    price   = df["XAG_EUR"].iloc[-1]
    rsi_val = rsi(df["XAG_EUR"]).iloc[-1]
    sma20_v, upper_v, lower_v = bollinger(df["XAG_EUR"])
    sma20   = sma20_v.iloc[-1]
    upper   = upper_v.iloc[-1]
    lower   = lower_v.iloc[-1]
    sma50   = df["XAG_EUR"].rolling(50).mean().iloc[-1]
    ratio   = df["ratio_xau_xag"].iloc[-1]
    ratio_m = df["ratio_xau_xag"].mean()
    perf_1m = (price - df["XAG_EUR"].iloc[-22]) / df["XAG_EUR"].iloc[-22] * 100

    score   = 10   # base neutre
    reasons = []

    # ── 1. RSI (±20 pts) ────────────────────────────────────────────────────
    if rsi_val < 30:
        score += 20
        reasons.append(("✅", f"RSI très bas ({rsi_val:.1f}) — zone de survente forte"))
    elif rsi_val < 45:
        score += 10
        reasons.append(("✅", f"RSI bas ({rsi_val:.1f}) — légère survente"))
    elif rsi_val > 70:
        score -= 20
        reasons.append(("🔴", f"RSI élevé ({rsi_val:.1f}) — zone de surachat"))
    else:
        reasons.append(("➡️", f"RSI neutre ({rsi_val:.1f})"))

    # ── 2. Bollinger (±25 pts) ───────────────────────────────────────────────
    bb_pos = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    if bb_pos < 0.2:
        score += 25
        reasons.append(("✅", f"Prix proche bande basse Bollinger ({bb_pos:.0%}) — signal d'achat"))
    elif bb_pos > 0.8:
        score -= 15
        reasons.append(("🔴", f"Prix proche bande haute Bollinger ({bb_pos:.0%}) — attention"))
    else:
        reasons.append(("➡️", f"Prix dans bande médiane Bollinger ({bb_pos:.0%})"))

    # ── 3. Moyennes mobiles SMA20 / SMA50 (±15 pts) ─────────────────────────
    if price < sma50 and price < sma20:
        score += 15
        reasons.append(("✅", f"Prix sous SMA20 ({sma20:.2f}€) et SMA50 ({sma50:.2f}€) — bon point d'entrée"))
    elif price > sma50 and price > sma20:
        score -= 10
        reasons.append(("🔴", f"Prix au-dessus SMA20 et SMA50 — prudence"))
    else:
        reasons.append(("➡️", f"Prix entre les moyennes mobiles"))

    # ── 4. Ratio Or/Argent (±20 pts) ────────────────────────────────────────
    if ratio > ratio_m * 1.1:
        score += 20
        reasons.append(("✅", f"Ratio or/argent élevé ({ratio:.1f} vs moy. {ratio_m:.1f}) — argent bon marché"))
    elif ratio < ratio_m * 0.9:
        score -= 10
        reasons.append(("🔴", f"Ratio or/argent bas ({ratio:.1f}) — argent relativement cher vs l'or"))
    else:
        reasons.append(("➡️", f"Ratio or/argent normal ({ratio:.1f} | moy. {ratio_m:.1f})"))

    # ── 5. Performance 1 mois (±10 pts) ─────────────────────────────────────
    if perf_1m < -5:
        score += 10
        reasons.append(("✅", f"Baisse mensuelle {perf_1m:.1f}% — possible opportunité de racheter bas"))
    elif perf_1m > 10:
        score -= 5
        reasons.append(("⚠️", f"Forte hausse mensuelle +{perf_1m:.1f}% — momentum haussier mais prix élevé"))
    else:
        reasons.append(("➡️", f"Performance mensuelle neutre ({perf_1m:+.1f}%)"))

    # ── 6. Force Relative XAG vs SPY (±15 pts) ──────────────────────────────
    # RS > 1 sur 90j = XAG surperforme le S&P500 → signal positif
    # RS < 1 nettement = les actions font mieux, pénalité
    if ind.get("rs_spy") is not None:
        rs = ind["rs_spy"]
        if rs > 1.05:
            score += 15
            reasons.append(("✅", f"XAG surperforme SPY (RS={rs:.2f}) — momentum argent fort vs actions"))
        elif rs < 0.90:
            score -= 15
            reasons.append(("🔴", f"XAG sous-performe SPY (RS={rs:.2f}) — les actions dominent, prudence"))
        else:
            reasons.append(("➡️", f"Force Relative XAG/SPY neutre (RS={rs:.2f})"))

    # ── 7. Sharpe XAG vs SPY — 30 jours (±10 pts) ───────────────────────────
    # Si le Sharpe de XAG dépasse celui du SPY à 30j, l'argent offre
    # un meilleur rendement ajusté au risque à court terme → signal positif
    s_xag = ind.get("sharpe_xag_30")
    s_spy = ind.get("sharpe_spy_30")
    if s_xag is not None and s_spy is not None and not (np.isnan(s_xag) or np.isnan(s_spy)):
        if s_xag > s_spy + 0.3:
            score += 10
            reasons.append(("✅", f"Sharpe XAG 30j ({s_xag:.2f}) > Sharpe SPY ({s_spy:.2f}) — meilleur rendement/risque"))
        elif s_xag < s_spy - 0.3:
            score -= 10
            reasons.append(("🔴", f"Sharpe XAG 30j ({s_xag:.2f}) < Sharpe SPY ({s_spy:.2f}) — SPY plus efficace risque/rendement"))
        else:
            reasons.append(("➡️", f"Sharpe XAG 30j ({s_xag:.2f}) ≈ SPY ({s_spy:.2f}) — risque/rendement similaires"))

    # ── 8. Momentum DXY — Dollar Index (±15 pts) ────────────────────────────
    # Un dollar US fort est historiquement négatif pour les métaux précieux
    # (les matières premières sont cotées en USD → dollar fort = métal plus cher
    #  pour les acheteurs étrangers → demande plus faible → prix baisse)
    dxy_mom = ind.get("dxy_mom_1m")
    if dxy_mom is not None and not np.isnan(dxy_mom):
        if dxy_mom > 2.0:
            score -= 15
            reasons.append(("🔴", f"Dollar fort : DXY +{dxy_mom:.1f}% sur 1 mois — vent de face pour les métaux"))
        elif dxy_mom < -2.0:
            score += 10
            reasons.append(("✅", f"Dollar faible : DXY {dxy_mom:.1f}% sur 1 mois — favorable aux métaux précieux"))
        else:
            reasons.append(("➡️", f"Dollar stable (DXY {dxy_mom:+.1f}% sur 1 mois)"))

    final_score = max(0, min(100, score))

    # ── Signal qualitatif ────────────────────────────────────────────────────
    rs_spy = ind.get("rs_spy")
    if final_score >= 75:
        label = "ACHAT FORT"
    elif final_score >= 60:
        label = "ACHETER"
    elif final_score >= 40:
        label = "NEUTRE"
    elif final_score < 25 and rs_spy is not None and rs_spy < 0.90:
        label = "PRIVILÉGIER SPY"
    else:
        label = "ATTENDRE"

    return final_score, label, reasons, rsi_val


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
ratio      = df["ratio_xau_xag"].iloc[-1]
rsi_now    = rsi(df["XAG_EUR"]).iloc[-1]
ind        = compute_indicators(df)

# ── Statut des flux de données (debug discret) ────────────────────────────────
with st.expander("📡 Statut des données marché", expanded=False):
    status_cols = st.columns(5)
    eur_rate = df["EUR_USD"].iloc[-1]  # taux EUR/USD pour conversion

    # Actifs en USD → convertis en EUR | DXY = indice sans unité, affiché tel quel
    assets = {
        "XAG (Argent)": ("XAG_USD", "€"),
        "SPY (S&P500)":  ("SPY",     "€"),
        "GLD (Or ETF)":  ("GLD",     "€"),
        "TLT (Oblig.)":  ("TLT",     "€"),
        "DXY (Dollar)":  ("DXY",     "pts"),  # indice, pas de conversion
    }
    for col, (label, (key, unit)) in zip(status_cols, assets.items()):
        if key in df.columns and df[key].notna().sum() > 10:
            last_val = df[key].iloc[-1]
            display  = last_val / eur_rate if unit == "€" else last_val
            col.metric(label, f"{display:.2f} {unit}", "✅ OK")
        else:
            col.metric(label, "—", "❌ Manquant")

# ── Portefeuille ──────────────────────────────────────────────────────────────

portfolio = load_portfolio()
qty       = portfolio.get("quantity", 0.0)
avg_price = portfolio.get("avg_price", 0.0)

if qty > 0:
    current_val = qty * price
    invested    = qty * avg_price
    pnl         = current_val - invested
    pnl_pct     = (pnl / invested * 100) if invested > 0 else 0.0

    st.subheader("💼 Mon portefeuille XAG")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantité", f"{qty:.4f} oz", help="1 oz troy = 31,1 g")
    c2.metric("Valeur actuelle", f"{current_val:.2f} €")
    c3.metric("Prix moyen d'achat", f"{avg_price:.2f} €/oz")
    c4.metric("P&L", f"{pnl:+.2f} €", f"{pnl_pct:+.2f}%")
    st.divider()

# ── Métriques marché ──────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric("Prix XAG/EUR",    f"{price:.2f} €",  f"{change_pct:+.2f}%")
col2.metric("Ratio Or/Argent", f"{ratio:.1f}",     help="Plus ce ratio est élevé, plus l'argent est bon marché vs l'or.")
col3.metric("RSI 14j",         f"{rsi_now:.1f}",   help="<30 = survente (achat intéressant) | >70 = surachat (attendre)")

st.divider()

# ── Signal d'achat enrichi ────────────────────────────────────────────────────

score, label, reasons, _ = compute_score(df, ind)
col_sig, col_why = st.columns([1, 2])

with col_sig:
    if label == "ACHAT FORT":
        st.success(f"### 💚 {label}\nScore : **{score} / 100**")
    elif label == "ACHETER":
        st.success(f"### 🟢 {label}\nScore : **{score} / 100**")
    elif label == "NEUTRE":
        st.warning(f"### 🟡 {label}\nScore : **{score} / 100**")
    elif label == "PRIVILÉGIER SPY":
        st.error(f"### 📈 {label}\nScore : **{score} / 100**")
    else:
        st.error(f"### 🔴 {label}\nScore : **{score} / 100**")
    st.progress(score / 100)
    st.caption("Score basé sur 8 indicateurs techniques et fondamentaux")

with col_why:
    st.markdown("**Analyse détaillée :**")
    for icon, text in reasons:
        st.markdown(f"{icon} {text}")

# ── Métriques avancées : Sharpe & Force Relative ──────────────────────────────
st.divider()
st.subheader("📐 Indicateurs avancés")

mc1, mc2, mc3, mc4, mc5 = st.columns(5)

s30 = ind.get("sharpe_xag_30")
s90 = ind.get("sharpe_xag_90")
ss30 = ind.get("sharpe_spy_30")
rs_spy = ind.get("rs_spy")
rs_gld = ind.get("rs_gld")

mc1.metric(
    "Sharpe XAG 30j", f"{s30:.2f}" if s30 and not np.isnan(s30) else "—",
    help="Rendement ajusté au risque sur 30 jours. >1 = bon, >2 = excellent, <0 = perte réelle"
)
mc2.metric(
    "Sharpe XAG 90j", f"{s90:.2f}" if s90 and not np.isnan(s90) else "—",
    help="Rendement ajusté au risque sur 90 jours"
)
mc3.metric(
    "Sharpe SPY 30j", f"{ss30:.2f}" if ss30 and not np.isnan(ss30) else "—",
    help="Sharpe du S&P500 sur 30 jours — référence de comparaison"
)
mc4.metric(
    "RS XAG/SPY 90j", f"{rs_spy:.2f}" if rs_spy else "—",
    help="Force Relative vs S&P500 sur 90j. >1 = XAG surperforme, <1 = SPY surperforme"
)
mc5.metric(
    "RS XAG/Or 90j", f"{rs_gld:.2f}" if rs_gld else "—",
    help="Force Relative vs Or sur 90j. >1 = Argent surperforme l'or"
)

st.divider()

# ── Graphiques ────────────────────────────────────────────────────────────────

st.subheader("📈 Prix XAG/EUR — 6 mois")

df_chart        = df.tail(180).copy()
sma20_s, ub, lb = bollinger(df_chart["XAG_EUR"])
sma50_s         = df_chart["XAG_EUR"].rolling(50).mean()
rsi_s           = rsi(df_chart["XAG_EUR"])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.06,
    subplot_titles=("Prix XAG/EUR + Bollinger + Moyennes mobiles", "RSI (14 jours)"),
)

fig.add_trace(go.Scatter(x=df_chart.index, y=ub, name="BB Sup",
    line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df_chart.index, y=lb, name="BB Inf",
    fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
    line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["XAG_EUR"],
    name="XAG/EUR", line=dict(color="#c0c0c0", width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_chart.index, y=sma20_s,
    name="SMA 20", line=dict(color="#4da6ff", dash="dot", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_chart.index, y=sma50_s,
    name="SMA 50", line=dict(color="#ff9f4a", dash="dot", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_s,
    name="RSI", line=dict(color="#b48ead", width=2)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", row=2, col=1)

fig.update_yaxes(title_text="€ / oz", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_layout(height=560, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Ratio Or / Argent — 6 mois")
st.caption("Un ratio élevé signifie que l'argent est bon marché vs l'or → signal d'achat supplémentaire")

fig2 = go.Figure()
fig2.add_hrect(y0=df["ratio_xau_xag"].quantile(0.75), y1=df["ratio_xau_xag"].max() + 5,
    fillcolor="rgba(0,200,0,0.07)", annotation_text="Zone favorable argent",
    annotation_position="top left")
fig2.add_trace(go.Scatter(x=df_chart.index, y=df_chart["ratio_xau_xag"],
    name="Ratio XAU/XAG", line=dict(color="gold", width=2)))
fig2.add_hline(y=df["ratio_xau_xag"].mean(), line_dash="dash", line_color="white",
    annotation_text=f"Moy. 1 an ({df['ratio_xau_xag'].mean():.1f})",
    annotation_position="bottom right")
fig2.update_layout(height=280, template="plotly_dark",
    margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# ── Mise à jour du portefeuille ───────────────────────────────────────────────

st.divider()
st.subheader("⚙️ Mettre à jour mon portefeuille")

with st.form("portfolio_form"):
    col_a, col_b = st.columns(2)
    new_qty = col_a.number_input(
        "Quantité XAG (oz troy)", min_value=0.0, value=float(qty),
        step=0.001, format="%.4f",
        help="Consulte Revolut : Métaux → Argent → ta quantité en oz"
    )
    new_avg = col_b.number_input(
        "Prix moyen d'achat (€/oz)", min_value=0.0, value=float(avg_price),
        step=0.01, format="%.2f",
        help="Prix moyen auquel tu as acheté ton argent"
    )
    submitted = st.form_submit_button("Sauvegarder", use_container_width=True)
    if submitted:
        with st.spinner("Sauvegarde sur GitHub..."):
            save_portfolio(new_qty, new_avg)
        st.success("Portefeuille mis à jour !")
        st.rerun()

# ── Pied de page ──────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Données : Yahoo Finance · Actualisé toutes les heures · "
    f"Dernière mise à jour : {datetime.today().strftime('%d/%m/%Y %H:%M')} · "
    f"⚠️ Outil d'aide à la décision uniquement — pas un conseil financier"
)
