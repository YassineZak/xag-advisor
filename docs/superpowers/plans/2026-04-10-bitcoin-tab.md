# Bitcoin Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un onglet Bitcoin au dashboard XAG Advisor avec solde Binance en temps réel, signal d'achat 0–100 (RSI + Bollinger + Fear & Greed + cycle halving), et graphiques techniques.

**Architecture:** On extrait le code XAG existant de `app.py` vers `xag_tab.py`, on crée `btc_tab.py` pour tout le code Bitcoin, et `app.py` ne sert plus que de point d'entrée avec `st.tabs()`.

**Tech Stack:** Streamlit, yfinance (BTC-USD), python-binance (balance), alternative.me API (Fear & Greed), Plotly, pandas, pytest

---

## File Map

| Fichier | Action | Responsabilité |
|---------|--------|----------------|
| `app.py` | Modifier | Point d'entrée unique : page config + st.tabs() |
| `xag_tab.py` | Créer | Tout le code XAG actuel (extrait de app.py) |
| `btc_tab.py` | Créer | Données BTC, score, UI |
| `portfolio.json` | Modifier | Ajouter champ `btc_avg_price` |
| `requirements.txt` | Modifier | Ajouter `python-binance>=1.0.17` |
| `.gitignore` | Créer/Modifier | Exclure `.streamlit/secrets.toml` |
| `tests/test_btc_score.py` | Créer | Tests unitaires du scoring BTC |

---

## Task 1 : Setup (gitignore, requirements, portfolio.json)

**Files:**
- Modify: `requirements.txt`
- Modify: `portfolio.json`
- Modify or create: `.gitignore`

- [ ] **Step 1: Ajouter python-binance à requirements.txt**

Contenu final de `requirements.txt` :
```
streamlit>=1.32.0
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
PyGithub>=2.1.1
python-binance>=1.0.17
```

- [ ] **Step 2: Ajouter btc_avg_price à portfolio.json**

Contenu final de `portfolio.json` :
```json
{
  "quantity": 16.7594,
  "avg_price": 0.0,
  "last_updated": "2026-04-09",
  "btc_avg_price": 0.0
}

```

- [ ] **Step 3: Ajouter .gitignore pour protéger les secrets**

Créer ou compléter `.gitignore` :
```
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt portfolio.json .gitignore
git commit -m "chore: add python-binance, btc_avg_price field, gitignore secrets"
```

---

## Task 2 : Extraire le code XAG dans xag_tab.py

**Files:**
- Create: `xag_tab.py`
- Modify: `app.py`

Le but est de déplacer TOUT le contenu actuel de `app.py` dans `xag_tab.py`, encapsulé dans une fonction `render()`. `app.py` devient minimaliste.

- [ ] **Step 1: Créer xag_tab.py avec le contenu actuel de app.py**

Copier l'intégralité du contenu de `app.py` dans `xag_tab.py`, puis :
1. Retirer les lignes `st.set_page_config(...)` et le bloc CSS `st.markdown(...)` (ils resteront dans `app.py`)
2. Encapsuler tout le code UI (à partir de la ligne `col_title, col_refresh = st.columns(...)`) dans une fonction `render()` :

```python
# xag_tab.py
import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from github import Github, GithubException

# ── Toutes les fonctions existantes restent ici (load_portfolio, save_portfolio,
#    get_data, get_live_price, rsi, bollinger, force_relative, sharpe_rolling,
#    compute_indicators, compute_score, generate_advice) ──────────────────────


def render():
    """Point d'entrée du tab XAG — appelé par app.py."""
    # ── Tout le code UI actuel de app.py va ici (lignes ~579 à 978) ──────────
    col_title, col_refresh = st.columns([5, 1])
    col_title.title("🥈 XAG/EUR — Advisor")
    # ... (rest of existing UI code)
```

- [ ] **Step 2: Vérifier que xag_tab.py est complet**

```bash
wc -l xag_tab.py
# Attendu : ~950 lignes (978 originales moins page_config + CSS ~30 lignes)
```

- [ ] **Step 3: Réécrire app.py**

```python
# app.py
import streamlit as st
import xag_tab
import btc_tab  # sera créé dans la prochaine tâche — ne pas importer encore

st.set_page_config(
    page_title="Portfolio Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🥈 Argent (XAG)", "₿ Bitcoin (BTC)"])

with tab1:
    xag_tab.render()

with tab2:
    st.info("Tab BTC en cours de construction...")
```

> Note : On laisse `btc_tab` commenté pour l'instant, le tab2 affiche un placeholder. On le branchera dans la Task 6.

- [ ] **Step 4: Tester localement que le tab XAG fonctionne**

```bash
streamlit run app.py
```

Attendu : L'app se lance avec 2 onglets. Onglet "🥈 Argent (XAG)" fonctionne identiquement à avant. Onglet "₿ Bitcoin (BTC)" affiche le message "en cours de construction".

- [ ] **Step 5: Commit**

```bash
git add app.py xag_tab.py
git commit -m "refactor: extract XAG code to xag_tab.py, add st.tabs() in app.py"
```

---

## Task 3 : Fonctions de données BTC (btc_tab.py — partie 1)

**Files:**
- Create: `btc_tab.py`

- [ ] **Step 1: Créer btc_tab.py avec les fonctions de données**

```python
# btc_tab.py
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
def get_btc_live_price() -> tuple[float | None, datetime | None]:
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


# ── Solde Binance ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_btc_balance() -> float:
    """
    Récupère le solde BTC (free + locked) depuis Binance.
    Nécessite BINANCE_API_KEY et BINANCE_API_SECRET dans st.secrets.
    Retourne 0.0 si non configuré ou en cas d'erreur.
    """
    try:
        from binance.client import Client
        api_key = st.secrets["BINANCE_API_KEY"]
        api_secret = st.secrets["BINANCE_API_SECRET"]
        client = Client(api_key, api_secret)
        balance = client.get_asset_balance(asset="BTC")
        return float(balance["free"]) + float(balance["locked"])
    except KeyError:
        return 0.0
    except Exception:
        return 0.0


# ── Fear & Greed Index ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_fear_greed() -> tuple[int | None, str | None, pd.DataFrame]:
    """
    Récupère le Fear & Greed Index depuis alternative.me (gratuit, pas de clé API).
    Retourne (valeur_actuelle, label_actuel, dataframe_30j).
    dataframe_30j a colonnes : date (index), value, label
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
        current_value = df_fg["value"].iloc[-1]
        current_label = df_fg["label"].iloc[-1]
        return int(current_value), current_label, df_fg

    except Exception:
        return None, None, pd.DataFrame()
```

- [ ] **Step 2: Commit**

```bash
git add btc_tab.py
git commit -m "feat: add BTC data functions (yfinance, Binance balance, Fear & Greed)"
```

---

## Task 4 : Fonction de scoring BTC

**Files:**
- Modify: `btc_tab.py`
- Create: `tests/test_btc_score.py`

- [ ] **Step 1: Écrire les tests du scoring avant l'implémentation**

Créer `tests/test_btc_score.py` :

```python
# tests/test_btc_score.py
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_tab import compute_btc_score


def make_df(rsi_val, close, bb_lower, bb_upper):
    """Helper : crée un DataFrame minimal pour compute_btc_score."""
    n = 60
    df = pd.DataFrame({
        "Close": [close] * n,
        "RSI": [rsi_val] * n,
        "BB_lower": [bb_lower] * n,
        "BB_upper": [bb_upper] * n,
        "BB_mid": [(bb_lower + bb_upper) / 2] * n,
        "SMA20": [close] * n,
        "SMA50": [close] * n,
    })
    return df


def test_extreme_fear_low_rsi_gives_high_score():
    """Extreme Fear + RSI oversold + prix sous bande basse → score élevé (≥60)."""
    df = make_df(rsi_val=25, close=80_000, bb_lower=85_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=15)
    assert score >= 60, f"Score attendu ≥60, obtenu {score}"


def test_extreme_greed_high_rsi_gives_low_score():
    """Extreme Greed + RSI surachat + prix au-dessus bande haute → score faible (≤40)."""
    df = make_df(rsi_val=78, close=100_000, bb_lower=80_000, bb_upper=95_000)
    score, label, _ = compute_btc_score(df, fear_greed=85)
    assert score <= 40, f"Score attendu ≤40, obtenu {score}"


def test_neutral_conditions_give_neutral_score():
    """Conditions neutres → pas de signal fort dans un sens ou l'autre."""
    df = make_df(rsi_val=50, close=90_000, bb_lower=85_000, bb_upper=95_000)
    score, label, _ = compute_btc_score(df, fear_greed=50)
    assert score < 60, f"Score attendu <60 pour conditions neutres, obtenu {score}"
    assert label not in ("ACHAT FORT", "ÉVITER"), f"Label inattendu: {label}"


def test_label_strong_buy():
    """Score ≥75 → label 'ACHAT FORT'."""
    df = make_df(rsi_val=20, close=75_000, bb_lower=85_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=10)
    assert label == "ACHAT FORT", f"Label attendu 'ACHAT FORT', obtenu '{label}' (score={score})"


def test_label_avoid():
    """Score ≤24 → label 'ÉVITER'."""
    df = make_df(rsi_val=82, close=105_000, bb_lower=80_000, bb_upper=100_000)
    score, label, _ = compute_btc_score(df, fear_greed=90)
    assert label == "ÉVITER", f"Label attendu 'ÉVITER', obtenu '{label}' (score={score})"


def test_score_clamped_0_100():
    """Score toujours entre 0 et 100."""
    df = make_df(rsi_val=5, close=50_000, bb_lower=90_000, bb_upper=100_000)
    score, _, _ = compute_btc_score(df, fear_greed=5)
    assert 0 <= score <= 100

    df2 = make_df(rsi_val=95, close=120_000, bb_lower=80_000, bb_upper=100_000)
    score2, _, _ = compute_btc_score(df2, fear_greed=95)
    assert 0 <= score2 <= 100


def test_missing_fear_greed_still_works():
    """Si fear_greed=None, la fonction ne plante pas et retourne un score valide."""
    df = make_df(rsi_val=50, close=90_000, bb_lower=85_000, bb_upper=95_000)
    score, label, reasons = compute_btc_score(df, fear_greed=None)
    assert 0 <= score <= 100
    assert isinstance(label, str)
    assert isinstance(reasons, list)
```

- [ ] **Step 2: Lancer les tests — ils doivent échouer (fonction pas encore écrite)**

```bash
cd /home/yassine/xag-advisor && python -m pytest tests/test_btc_score.py -v 2>&1 | head -30
```

Attendu : erreur `ImportError: cannot import name 'compute_btc_score' from 'btc_tab'`

- [ ] **Step 3: Implémenter compute_btc_score dans btc_tab.py**

Ajouter après les fonctions de données dans `btc_tab.py` :

```python
# ── Score d'achat BTC ─────────────────────────────────────────────────────────

HALVING_DATE = date(2024, 4, 19)   # dernier halving Bitcoin

def _halving_cycle_score() -> tuple[int, str]:
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
    fear_greed: int | None,
) -> tuple[int, str, list[tuple[str, str]]]:
    """
    Score d'achat BTC 0–100 basé sur 4 indicateurs pondérés.

    Pondérations :
      RSI 14j           : 25 pts max
      Bollinger position : 20 pts max
      Fear & Greed Index : 35 pts max
      Cycle post-halving : 20 pts max

    Retourne (score, label, reasons) où reasons est une liste de (icon, texte).
    """
    score = 0
    reasons = []

    rsi_val = df["RSI"].iloc[-1]
    close   = df["Close"].iloc[-1]
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
```

- [ ] **Step 4: Lancer les tests — ils doivent passer**

```bash
python -m pytest tests/test_btc_score.py -v
```

Attendu :
```
tests/test_btc_score.py::test_extreme_fear_low_rsi_gives_high_score PASSED
tests/test_btc_score.py::test_extreme_greed_high_rsi_gives_low_score PASSED
tests/test_btc_score.py::test_neutral_conditions_give_neutral_score PASSED
tests/test_btc_score.py::test_label_strong_buy PASSED
tests/test_btc_score.py::test_label_avoid PASSED
tests/test_btc_score.py::test_score_clamped_0_100 PASSED
tests/test_btc_score.py::test_missing_fear_greed_still_works PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add btc_tab.py tests/test_btc_score.py
git commit -m "feat: add BTC scoring function with TDD (RSI + Bollinger + Fear&Greed + halving cycle)"
```

---

## Task 5 : Interface UI du tab BTC

**Files:**
- Modify: `btc_tab.py` (ajouter la fonction `render()`)

- [ ] **Step 1: Ajouter la fonction render() à btc_tab.py**

```python
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
        current_price = float(df["Close"].iloc[-1])
        # Approximate EUR conversion (fallback)
        import yfinance as yf
        try:
            eur_usd = yf.Ticker("EURUSD=X").fast_info.last_price or 1.08
            current_price = current_price / eur_usd
        except Exception:
            current_price = current_price / 1.08
        price_label = "Estimé (EUR)"

    btc_balance = get_btc_balance()
    fg_value, fg_label, df_fg = get_fear_greed()
    score, signal_label, reasons = compute_btc_score(df, fear_greed=fg_value)

    # Lire avg_price depuis portfolio.json via GitHub (même logique que XAG)
    try:
        from github import Github, GithubException
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo("YassineZak/xag-advisor")
        import json
        f = repo.get_contents("portfolio.json")
        portfolio = json.loads(f.decoded_content)
        btc_avg_price = float(portfolio.get("btc_avg_price", 0.0))
    except Exception:
        btc_avg_price = 0.0

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
        st.warning("⚠️ Solde Binance non configuré. Ajoute BINANCE_API_KEY et BINANCE_API_SECRET dans les secrets Streamlit.")

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
        fg_color = "#22c55e" if fg_value <= 25 else ("#86efac" if fg_value <= 40 else
                   ("#facc15" if fg_value <= 60 else ("#f97316" if fg_value <= 75 else "#ef4444")))
        col_fg, col_fg_chart = st.columns([1, 3])
        col_fg.markdown(f"""
        <div style="text-align:center; padding:15px; background:#1e1e2e; border-radius:10px;">
            <div style="font-size:2.5rem; font-weight:bold; color:{fg_color};">{fg_value}</div>
            <div style="color:{fg_color}; font-weight:600;">{fg_label}</div>
        </div>
        """, unsafe_allow_html=True)

        if not df_fg.empty:
            import plotly.graph_objects as go
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
    st.subheader("📈 Analyse technique")
    df_6m = df.tail(180).copy()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03,
    )

    # Prix + Bollinger + SMAs
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_upper"], name="BB Upper",
        line=dict(color="rgba(148,163,184,0.4)", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_lower"], name="BB Lower",
        line=dict(color="rgba(148,163,184,0.4)", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(148,163,184,0.06)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["BB_mid"], name="BB Mid",
        line=dict(color="rgba(148,163,184,0.3)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["SMA20"], name="SMA20",
        line=dict(color="#60a5fa", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["SMA50"], name="SMA50",
        line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["Close"], name="BTC/EUR",
        line=dict(color="#f97316", width=2.5)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_6m.index, y=df_6m["RSI"], name="RSI 14",
        line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=500,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Bloc 5 : Évolution du portefeuille ────────────────────────────────────
    if btc_balance > 0 and btc_avg_price > 0:
        st.subheader("💼 Évolution du portefeuille BTC")
        # Approximation EUR via conversion USD→EUR sur l'historique
        try:
            eur_usd_hist = yf.Ticker("EURUSD=X").history(period="400d")["Close"]
            eur_usd_hist.index = eur_usd_hist.index.tz_localize(None)
            df_port = df[["Close"]].join(eur_usd_hist.rename("EUR_USD"), how="left")
            df_port["EUR_USD"] = df_port["EUR_USD"].ffill()
            df_port["Close_EUR"] = df_port["Close"] / df_port["EUR_USD"]
        except Exception:
            df_port = df.copy()
            df_port["Close_EUR"] = df_port["Close"] / 1.08

        purchase_line = btc_avg_price
        df_port["value_eur"] = df_port["Close_EUR"] * btc_balance
        purchase_value = purchase_line * btc_balance

        fig_port = go.Figure()
        fig_port.add_hline(y=purchase_value, line_dash="dash", line_color="#94a3b8",
                           annotation_text=f"Prix moyen {btc_avg_price:,.0f} €")
        fig_port.add_trace(go.Scatter(
            x=df_port.index, y=df_port["value_eur"],
            mode="lines", line=dict(color="#f97316", width=2),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
            name="Valeur portefeuille BTC (€)"
        ))
        fig_port.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_port, use_container_width=True)
```

- [ ] **Step 2: Commit**

```bash
git add btc_tab.py
git commit -m "feat: add BTC tab UI (metrics, signal, Fear&Greed chart, technical charts, portfolio)"
```

---

## Task 6 : Brancher btc_tab dans app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Mettre à jour app.py pour importer et utiliser btc_tab**

```python
# app.py
import streamlit as st
import xag_tab
import btc_tab

st.set_page_config(
    page_title="Portfolio Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🥈 Argent (XAG)", "₿ Bitcoin (BTC)"])

with tab1:
    xag_tab.render()

with tab2:
    btc_tab.render()
```

- [ ] **Step 2: Tester l'intégration complète localement**

```bash
streamlit run app.py
```

Checklist de vérification :
- [ ] Tab XAG fonctionne identiquement à avant
- [ ] Tab BTC s'affiche sans erreur (même sans clés Binance → warning affiché)
- [ ] Prix BTC visible
- [ ] Fear & Greed Index visible
- [ ] Graphiques techniques visibles
- [ ] Score 0-100 visible avec label

- [ ] **Step 3: Lancer tous les tests**

```bash
python -m pytest tests/ -v
```

Attendu : 7 passed

- [ ] **Step 4: Commit final**

```bash
git add app.py
git commit -m "feat: wire btc_tab into app.py — Bitcoin tab fully integrated"
```

---

## Task 7 : Configuration Streamlit Cloud & push

**Files:**
- Aucun fichier code — configuration dans l'interface Streamlit Cloud

- [ ] **Step 1: Créer .streamlit/secrets.toml localement pour test (ne pas committer)**

```toml
# .streamlit/secrets.toml  ← déjà dans .gitignore
GITHUB_TOKEN = "ton_github_token"
BINANCE_API_KEY = "ta_binance_api_key"
BINANCE_API_SECRET = "ton_binance_api_secret"
```

Pour créer une clé Binance read-only :
1. Binance → Account → API Management
2. "Create API" → type "System Generated"
3. Cocher uniquement "Enable Reading" (décocher tout le reste)
4. Copier API Key et Secret Key

- [ ] **Step 2: Tester avec les vraies clés en local**

```bash
streamlit run app.py
```

Vérifier que le solde BTC s'affiche correctement dans les métriques.

- [ ] **Step 3: Push sur main (déclenche le redéploiement Streamlit Cloud)**

```bash
git push origin main
```

- [ ] **Step 4: Configurer les secrets sur Streamlit Cloud**

1. Aller sur share.streamlit.io → ton app xag-advisor
2. Settings → Secrets
3. Coller le contenu de `.streamlit/secrets.toml` (avec les vraies valeurs)
4. Save → l'app redémarre automatiquement

- [ ] **Step 5: Vérifier le déploiement**

Ouvrir l'URL de l'app sur mobile.
Vérifier : tab BTC visible, prix live, Fear & Greed, score, solde Binance.
