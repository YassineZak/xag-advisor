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
    content = json.dumps({
        "quantity":     quantity,
        "avg_price":    avg_price,
        "last_updated": datetime.today().strftime("%Y-%m-%d"),  # date de saisie
    }, indent=2)
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

@st.cache_data(ttl=900)   # données historiques : cache 15 min
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


# ── Prix spot quasi temps réel ───────────────────────────────────────────────
#
# Cache de 5 minutes — yfinance a environ 15 min de délai sur les données
# gratuites, mais c'est la meilleure approximation disponible sans API payante.

@st.cache_data(ttl=300)   # cache 5 minutes
def get_live_price():
    """
    Récupère le prix spot le plus récent pour XAG et EUR/USD.
    Utilise fast_info (plus rapide que history()) pour avoir la dernière cotation.
    Retourne (xag_eur, xag_usd, eur_usd, timestamp) ou None si échec.
    """
    try:
        xag_info = yf.Ticker("XAGUSD=X").fast_info
        eur_info = yf.Ticker("EURUSD=X").fast_info
        xag_usd  = xag_info.last_price
        eur_usd  = eur_info.last_price
        if xag_usd and eur_usd and xag_usd > 0 and eur_usd > 0:
            return xag_usd / eur_usd, xag_usd, eur_usd, datetime.now()
    except Exception:
        pass
    # Fallback : SI=F (futures argent)
    try:
        si_info  = yf.Ticker("SI=F").fast_info
        eur_info = yf.Ticker("EURUSD=X").fast_info
        xag_usd  = si_info.last_price
        eur_usd  = eur_info.last_price
        if xag_usd and eur_usd and xag_usd > 0 and eur_usd > 0:
            return xag_usd / eur_usd, xag_usd, eur_usd, datetime.now()
    except Exception:
        pass
    return None, None, None, None


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


# ── Conseil personnalisé ──────────────────────────────────────────────────────

def generate_advice(score: int, label: str, ind: dict, df: pd.DataFrame) -> list[dict]:
    """
    Génère une liste de blocs de conseil en langage simple, basés sur les
    indicateurs actuels. Chaque bloc a : type (info/success/warning/error),
    titre, et texte.
    """
    advice = []
    rs_spy  = ind.get("rs_spy")
    rs_gld  = ind.get("rs_gld")
    s30_xag = ind.get("sharpe_xag_30")
    s30_spy = ind.get("sharpe_spy_30")
    dxy_mom = ind.get("dxy_mom_1m")
    price   = df["XAG_EUR"].iloc[-1]
    ratio   = df["ratio_xau_xag"].iloc[-1]
    ratio_m = df["ratio_xau_xag"].mean()

    # ── Conseil principal selon le signal ───────────────────────────────────
    if label == "ACHAT FORT":
        advice.append({"type": "success", "titre": "Action recommandée : Acheter de l'XAG ce mois-ci",
            "texte": (
                "Les conditions sont très favorables pour renforcer ta position en argent métal. "
                "Plusieurs indicateurs s'alignent en ta faveur : le prix est bas par rapport à ses moyennes historiques, "
                "l'argent est sous-évalué et le momentum est positif. "
                "C'est le type de moment où acheter régulièrement (même une petite somme) a du sens."
            )})
    elif label == "ACHETER":
        advice.append({"type": "success", "titre": "Action recommandée : Envisager un achat ce mois-ci",
            "texte": (
                "Les conditions sont globalement favorables. Ce n'est pas le moment parfait mais c'est un "
                "bon moment pour investir si tu avais prévu de le faire ce mois-ci. "
                "Tu peux y aller avec une somme normale, sans surpondérer."
            )})
    elif label == "NEUTRE":
        advice.append({"type": "warning", "titre": "Action recommandée : Attendre ou investir prudemment",
            "texte": (
                "Les signaux sont mitigés ce mois-ci — certains indicateurs sont positifs, d'autres négatifs. "
                "Si tu pratiques le DCA (investissement régulier fixe chaque mois), tu peux maintenir "
                "ta mise habituelle. Sinon, mieux vaut attendre un signal plus clair le mois prochain."
            )})
    elif label == "PRIVILÉGIER SPY":
        advice.append({"type": "error", "titre": "Action recommandée : Pause sur XAG, regarder le S&P500",
            "texte": (
                "En ce moment, le marché actions (S&P500 via SPY) performe nettement mieux que l'argent. "
                "Si tu as la possibilité d'investir dans un ETF S&P500 depuis Revolut, "
                "c'est probablement le meilleur placement ce mois-ci. "
                "Pas besoin de vendre ton XAG actuel — juste de ne pas en acheter davantage."
            )})
    else:  # ATTENDRE
        advice.append({"type": "error", "titre": "Action recommandée : Ne pas acheter ce mois-ci",
            "texte": (
                "Les conditions ne sont pas réunies pour un achat ce mois-ci. "
                "Le prix est élevé ou la tendance est défavorable. "
                "Garde ta poudre sèche et reviens consulter le mois prochain. "
                "Ton XAG existant n'est pas à vendre pour autant — juste ne pas en ajouter maintenant."
            )})

    # ── Conseil sur la comparaison XAG vs actifs ─────────────────────────────
    if rs_spy is not None and rs_gld is not None:
        best = "XAG (Argent)" if rs_spy > 1 and rs_gld > 1 else ("SPY (S&P500)" if rs_spy < rs_gld else "Or (GLD)")
        if rs_spy > 1.05 and rs_gld > 1.05:
            advice.append({"type": "info", "titre": "Comparaison actifs : L'argent domine en ce moment",
                "texte": (
                    f"Sur les 90 derniers jours, l'argent (XAG) surperforme à la fois le S&P500 (RS={rs_spy:.2f}) "
                    f"et l'or (RS={rs_gld:.2f}). C'est un signal de momentum fort — l'argent est l'actif le plus "
                    f"porteur en ce moment dans ton univers de comparaison."
                )})
        elif rs_spy < 0.90:
            advice.append({"type": "info", "titre": "Comparaison actifs : Les actions font mieux que l'argent",
                "texte": (
                    f"Sur 90 jours, le S&P500 surperforme l'argent (RS XAG/SPY = {rs_spy:.2f}). "
                    f"Si tu cherches à maximiser ton rendement à court terme, un ETF S&P500 serait plus performant. "
                    f"L'argent reste un bon actif de diversification long terme, mais ce n'est pas son meilleur moment relatif."
                )})
        elif rs_gld < 0.92:
            advice.append({"type": "info", "titre": "Comparaison actifs : L'or fait mieux que l'argent",
                "texte": (
                    f"Sur 90 jours, l'or (GLD) surperforme l'argent (RS XAG/GLD = {rs_gld:.2f}). "
                    f"Si tu veux t'exposer aux métaux précieux, l'or est actuellement le métal le plus dynamique. "
                    f"L'argent reste intéressant sur le long terme (ratio or/argent à {ratio:.0f})."
                )})

    # ── Conseil sur le Dollar ──────────────────────────────────────────────────
    if dxy_mom is not None and not np.isnan(dxy_mom):
        if dxy_mom > 2.0:
            advice.append({"type": "warning", "titre": "Attention : Le dollar US se renforce",
                "texte": (
                    f"Le dollar américain a progressé de +{dxy_mom:.1f}% ce mois-ci (indice DXY). "
                    f"Comme les métaux sont cotés en dollars, un dollar fort rend l'argent plus cher "
                    f"pour les acheteurs mondiaux et tends à faire baisser les prix. "
                    f"Ce contexte pèse sur l'XAG à court terme."
                )})
        elif dxy_mom < -2.0:
            advice.append({"type": "info", "titre": "Contexte favorable : Le dollar US s'affaiblit",
                "texte": (
                    f"Le dollar a baissé de {dxy_mom:.1f}% ce mois-ci. "
                    f"Un dollar faible est historiquement bon pour les métaux précieux : "
                    f"ils deviennent moins chers en monnaie locale pour les acheteurs du monde entier, "
                    f"ce qui stimule la demande et tend à faire monter les prix."
                )})

    # ── Conseil sur le ratio or/argent ────────────────────────────────────────
    if ratio > ratio_m * 1.15:
        advice.append({"type": "info", "titre": "Opportunité structurelle : Ratio or/argent très élevé",
            "texte": (
                f"Il faut actuellement {ratio:.0f} oz d'argent pour acheter 1 oz d'or "
                f"(moyenne historique : {ratio_m:.0f} oz). "
                f"Historiquement, quand ce ratio est très élevé, l'argent tend à rattraper l'or. "
                f"C'est un argument de long terme en faveur de l'argent, indépendamment du signal mensuel."
            )})

    # ── Conseil sur le Sharpe ────────────────────────────────────────────────
    if s30_xag is not None and s30_spy is not None and not (np.isnan(s30_xag) or np.isnan(s30_spy)):
        if s30_xag > 1.5:
            advice.append({"type": "info", "titre": "Qualité du rendement : Bon ratio risque/rendement pour l'XAG",
                "texte": (
                    f"Le Sharpe de l'XAG à 30 jours est de {s30_xag:.2f}. "
                    f"Cela signifie que pour chaque unité de risque prise, tu obtiens un rendement positif solide. "
                    f"Un Sharpe > 1 est considéré comme bon, > 2 est excellent."
                )})
        elif s30_xag < 0:
            advice.append({"type": "warning", "titre": "Rendement récent décevant pour l'XAG",
                "texte": (
                    f"Le Sharpe XAG à 30 jours est négatif ({s30_xag:.2f}), ce qui signifie que "
                    f"le rendement récent de l'argent n'a même pas compensé le risque pris. "
                    f"C'est un signal de faiblesse à court terme — pas forcément durable, "
                    f"mais à surveiller."
                )})

    return advice


# ── Interface ─────────────────────────────────────────────────────────────────

col_title, col_refresh = st.columns([5, 1])
col_title.title("🥈 XAG/EUR — Advisor")

if col_refresh.button("🔄 Rafraîchir", use_container_width=True, help="Force le rechargement du prix en temps réel"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Chargement des données marché..."):
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        st.stop()

# ── Prix live (cache 5 min) — remplace le dernier prix historique ─────────────
live_eur, live_usd, live_fx, live_ts = get_live_price()

if live_eur and live_eur > 0:
    price = live_eur   # on écrase avec le prix le plus frais disponible
    price_source = f"Live ~{live_ts.strftime('%H:%M:%S')}"
else:
    price = df["XAG_EUR"].iloc[-1]
    price_source = "Historique (J-1)"

prev_price = df["XAG_EUR"].iloc[-2]
change_pct = (price - prev_price) / prev_price * 100
ratio      = df["ratio_xau_xag"].iloc[-1]
rsi_now    = rsi(df["XAG_EUR"]).iloc[-1]
ind        = compute_indicators(df)

# ── Calculs globaux ───────────────────────────────────────────────────────────

portfolio    = load_portfolio()
qty          = portfolio.get("quantity", 0.0)
avg_price    = portfolio.get("avg_price", 0.0)
last_updated = portfolio.get("last_updated", None)
score, label, reasons, _ = compute_score(df, ind)

# ── 1. SIGNAL D'ACHAT — en premier ───────────────────────────────────────────

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
    st.caption("Score 0–100 · Aujourd'hui · Au-dessus de 60 = conditions favorables à l'achat.")

with col_why:
    st.markdown("**Analyse détaillée des 8 indicateurs :**")
    for icon, text in reasons:
        st.markdown(f"{icon} {text}")

st.divider()

# ── 2. PORTEFEUILLE + évolution depuis entrée ─────────────────────────────────

if qty > 0:
    current_val = qty * price
    invested    = qty * avg_price
    pnl         = current_val - invested
    pnl_pct     = (pnl / invested * 100) if invested > 0 else 0.0

    st.subheader("💼 Mon portefeuille XAG")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantité", f"{qty:.4f} oz", help="1 oz troy = 31,1 grammes")
    c2.metric("Valeur actuelle", f"{current_val:.2f} €",
        help="Quantité × prix spot actuel")
    c3.metric("Prix moyen d'achat", f"{avg_price:.2f} €/oz",
        help="Prix que tu as saisi lors de ta dernière mise à jour")
    c4.metric("P&L total", f"{pnl:+.2f} €", f"{pnl_pct:+.2f}%",
        help="Gain ou perte par rapport à ton prix d'achat moyen")

    # ── Graphique évolution prix + P&L depuis la date d'entrée ──────────────
    if last_updated:
        entry_date = pd.to_datetime(last_updated)
        df_since   = df[df.index >= entry_date].copy()
        date_label = entry_date.strftime("%d/%m/%Y")
    else:
        # Pas de date stockée → on prend les 90 derniers jours par défaut
        df_since   = df.tail(90).copy()
        date_label = "90 derniers jours"

    if len(df_since) >= 2:
        pnl_series = qty * (df_since["XAG_EUR"] - avg_price)  # P&L en € à chaque jour

        fig_pf = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.08,
            subplot_titles=(
                f"Prix XAG/EUR depuis le {date_label}",
                "Gain / Perte en € depuis ton entrée",
            ),
        )

        # ── Subplot 1 : prix + ligne prix d'achat ───────────────────────────
        # Zone colorée : verte si au-dessus du prix d'achat, rouge si en-dessous
        fig_pf.add_trace(go.Scatter(
            x=df_since.index, y=[avg_price] * len(df_since),
            name=f"Ton prix d'achat ({avg_price:.2f} €)",
            line=dict(color="orange", dash="dash", width=1.5),
        ), row=1, col=1)
        fig_pf.add_trace(go.Scatter(
            x=df_since.index, y=df_since["XAG_EUR"],
            name="Prix XAG/EUR",
            fill="tonexty",
            fillcolor="rgba(0,200,100,0.12)" if price >= avg_price else "rgba(255,80,80,0.12)",
            line=dict(color="#c0c0c0", width=2),
        ), row=1, col=1)

        # Point d'entrée
        fig_pf.add_trace(go.Scatter(
            x=[df_since.index[0]], y=[avg_price],
            mode="markers",
            name="Point d'entrée",
            marker=dict(color="orange", size=10, symbol="diamond"),
        ), row=1, col=1)

        # ── Subplot 2 : P&L cumulé en € ─────────────────────────────────────
        pnl_color = ["rgba(0,200,100,0.8)" if v >= 0 else "rgba(255,80,80,0.8)"
                     for v in pnl_series]
        fig_pf.add_trace(go.Bar(
            x=df_since.index, y=pnl_series,
            name="P&L (€)",
            marker_color=pnl_color,
            showlegend=False,
        ), row=2, col=1)
        fig_pf.add_hline(y=0, line_color="white", line_width=1, row=2, col=1)

        fig_pf.update_yaxes(title_text="€ / oz", row=1, col=1)
        fig_pf.update_yaxes(title_text="€", row=2, col=1)
        fig_pf.update_layout(
            height=500, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_pf, use_container_width=True)

        if last_updated:
            st.caption(f"Données depuis la dernière mise à jour du portefeuille ({date_label}).")
        else:
            st.caption("Aucune date de saisie enregistrée — affichage sur 90 jours. "
                       "Sauvegarde ton portefeuille ci-dessous pour ancrer la date.")
    st.divider()

# ── 3. MÉTRIQUES MARCHÉ ───────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric(f"Prix XAG/EUR · {price_source}", f"{price:.2f} €", f"{change_pct:+.2f}% vs hier",
    help=(
        "Prix d'une once troy d'argent en euros. "
        "1 once troy = 31,1 grammes. "
        f"Source : {price_source}. "
        "Yahoo Finance a environ 15 min de délai sur les données gratuites. "
        "Clique sur 🔄 Rafraîchir pour forcer la mise à jour."
    ))
col2.metric("Ratio Or/Argent", f"{ratio:.1f}",
    help=(
        "Indique combien d'onces d'argent il faut pour acheter 1 once d'or. "
        f"Actuellement {ratio:.0f} oz d'argent = 1 oz d'or. "
        f"La moyenne historique est ~{df['ratio_xau_xag'].mean():.0f}. "
        "Plus ce chiffre est élevé, plus l'argent est bon marché par rapport à l'or — "
        "signal d'achat structurel."
    ))
col3.metric("RSI 14j", f"{rsi_now:.1f}",
    help=(
        "RSI = Relative Strength Index. Mesure si un actif est suracheté ou survendu. "
        "Va de 0 à 100. "
        "Sous 30 : l'argent a beaucoup baissé → bon moment potentiel pour acheter. "
        "Au-dessus de 70 : l'argent a beaucoup monté → mieux vaut attendre. "
        "Entre 30 et 70 : zone neutre."
    ))

st.divider()

# ── 4. INDICATEURS AVANCÉS ────────────────────────────────────────────────────

st.divider()
st.subheader("📐 Indicateurs avancés")
st.caption("Comparaison de l'argent à d'autres actifs pour savoir si c'est le meilleur placement en ce moment.")

s30    = ind.get("sharpe_xag_30")
s90    = ind.get("sharpe_xag_90")
ss30   = ind.get("sharpe_spy_30")
rs_spy = ind.get("rs_spy")
rs_gld = ind.get("rs_gld")

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Sharpe XAG 30j", f"{s30:.2f}" if s30 and not np.isnan(s30) else "—",
    help=(
        "Mesure le rendement obtenu par rapport au risque pris sur 30 jours. "
        "> 1 = bon | > 2 = excellent | < 0 = perte réelle."
    ))
mc2.metric("Sharpe XAG 90j", f"{s90:.2f}" if s90 and not np.isnan(s90) else "—",
    help="Même calcul sur 90 jours — vision moyen terme plus stable.")
mc3.metric("Sharpe SPY 30j", f"{ss30:.2f}" if ss30 and not np.isnan(ss30) else "—",
    help="Sharpe du S&P500 sur 30j — référence : si XAG > SPY, l'argent est plus efficace.")
mc4.metric("RS XAG/SPY 90j", f"{rs_spy:.2f}" if rs_spy else "—",
    help="> 1 = XAG surperforme le S&P500 sur 90j | < 1 = les actions font mieux.")
mc5.metric("RS XAG/Or 90j",  f"{rs_gld:.2f}" if rs_gld else "—",
    help="> 1 = Argent surperforme l'or sur 90j | < 1 = l'or fait mieux.")

# ── Statut des flux (discret) ─────────────────────────────────────────────────
with st.expander("📡 Statut des données marché", expanded=False):
    status_cols = st.columns(5)
    eur_rate = df["EUR_USD"].iloc[-1]
    assets   = {
        "XAG (Argent)": ("XAG_USD", "€"),
        "SPY (S&P500)":  ("SPY",     "€"),
        "GLD (Or ETF)":  ("GLD",     "€"),
        "TLT (Oblig.)":  ("TLT",     "€"),
        "DXY (Dollar)":  ("DXY",     "pts"),
    }
    for col, (lbl, (key, unit)) in zip(status_cols, assets.items()):
        if key in df.columns and df[key].notna().sum() > 10:
            last_val = df[key].iloc[-1]
            display  = last_val / eur_rate if unit == "€" else last_val
            col.metric(lbl, f"{display:.2f} {unit}", "✅ OK")
        else:
            col.metric(lbl, "—", "❌ Manquant")

st.divider()

# ── 6. GRAPHIQUES TECHNIQUES ──────────────────────────────────────────────────

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

# ── Glossaire pédagogique ────────────────────────────────────────────────────

st.divider()
with st.expander("📖 Glossaire — comprendre les indicateurs"):
    st.markdown("""
### Les actifs suivis

| Actif | Ce que c'est | Pourquoi on le suit |
|---|---|---|
| **XAG** | Argent métal (Silver). Coté en oz troy (1 oz = 31,1 g). | C'est l'actif que tu achètes sur Revolut. |
| **SPY** | ETF qui réplique le S&P500, l'indice des 500 plus grandes entreprises américaines. | Référence du marché actions mondiale. On compare l'argent à ça. |
| **GLD** | ETF adossé à l'or physique. | Permet de comparer argent vs or. |
| **TLT** | ETF d'obligations US à 20+ ans. | Quand les taux montent, TLT baisse. Il reflète le contexte de taux. |
| **DXY** | Indice Dollar US. Mesure la force du dollar face à un panier de devises (euro, yen...). | Un dollar fort fait baisser les métaux précieux. |

---

### Les indicateurs techniques

**RSI (Relative Strength Index)**
Mesure la vitesse et l'amplitude des variations de prix. Va de 0 à 100.
- **< 30** : l'actif a trop baissé trop vite (survente) → les vendeurs s'épuisent → rebond probable → **signal d'achat**
- **> 70** : l'actif a trop monté trop vite (surachat) → les acheteurs s'essoufflent → correction probable → **signal de vente ou d'attente**
- **Entre 30 et 70** : zone neutre, ni chaud ni froid.

**Bandes de Bollinger**
Enveloppe autour du prix composée de 3 lignes : une moyenne mobile centrale (SMA20) et deux bandes à 2 écarts-types au-dessus et en dessous.
- Le prix touche la **bande basse** → il est statistiquement bas → potentiel de rebond → **signal d'achat**
- Le prix touche la **bande haute** → il est statistiquement haut → risque de correction → **signal de prudence**
- 95% du temps le prix reste à l'intérieur des bandes.

**SMA20 / SMA50 (Simple Moving Average)**
Moyenne des 20 (ou 50) derniers jours de prix. Lisse les fluctuations et révèle la tendance.
- Prix **sous** la SMA50 → tendance baissière → le prix est "en solde" par rapport à sa moyenne récente → **opportunité d'achat potentielle**
- Prix **au-dessus** → tendance haussière → prix déjà élevé → **prudence**

---

### Les indicateurs avancés

**Ratio Or/Argent**
Nombre d'onces d'argent nécessaires pour acheter 1 once d'or.
- Historiquement entre 40 et 120.
- **Ratio élevé (ex: 90+)** : l'argent est bon marché par rapport à l'or → l'argent tend à rattraper → **opportunité long terme**
- **Ratio bas (ex: 40)** : l'argent est cher par rapport à l'or → attention.

**Force Relative (RS)**
Compare la performance de l'argent à un autre actif sur une période donnée, en normalisant les deux à 1 au départ.
- **RS > 1** : l'argent a fait mieux que l'actif de référence → momentum favorable à l'argent
- **RS < 1** : l'autre actif a fait mieux → l'argent sous-performe en relatif

**Ratio de Sharpe**
Mesure le rendement obtenu par rapport au risque pris (la volatilité).
`Sharpe = (Rendement moyen - Taux sans risque) / Volatilité × √252`
- **> 1** : bon — tu es bien rémunéré pour le risque pris
- **> 2** : excellent
- **< 0** : mauvais — tu perds de l'argent en termes réels

**Momentum DXY**
Variation du Dollar Index sur 1 mois.
- Les métaux sont cotés en dollars → **dollar fort = métal plus cher pour le monde = demande baisse = prix baisse**
- **Dollar faible = métal moins cher pour le monde = demande monte = prix monte**

---

### Le score global (0–100)

Agrège tous les indicateurs avec des pondérations :

| Signal | Score | Signification |
|---|---|---|
| 💚 ACHAT FORT | ≥ 75 | Conditions très favorables, plusieurs signaux alignés |
| 🟢 ACHETER | 60–74 | Conditions favorables |
| 🟡 NEUTRE | 40–59 | Signaux mitigés, pas de conviction claire |
| 🔴 ATTENDRE | 25–39 | Conditions défavorables |
| 📈 PRIVILÉGIER SPY | < 25 + RS faible | Les actions dominent, mieux vaut ne pas acheter d'argent ce mois |
""")

# ── Pied de page ──────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Données : Yahoo Finance (~15 min de délai) · Prix live mis en cache 5 min · "
    f"Indicateurs mis en cache 15 min · Clique 🔄 pour forcer le rafraîchissement · "
    f"Mise à jour : {datetime.today().strftime('%d/%m/%Y %H:%M:%S')} · "
    f"⚠️ Outil d'aide à la décision uniquement — pas un conseil financier"
)
