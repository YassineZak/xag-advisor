import json
import streamlit as st
from google import genai as _gemini
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── Univers d'actifs ────────────────────────────────────────────────────────

_GEMINI_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
]

# ETFs éligibles PEA — réplication synthétique (swap) sauf mention
ETF_PEA_UNIVERSE = [
    {"ticker": "EWLD.PA", "name": "Amundi MSCI World",    "zone": "Monde",        "ter": "0.38%", "description": "Exposition mondiale diversifiée via swap (PEA)"},
    {"ticker": "PSP5.PA", "name": "Lyxor S&P 500",        "zone": "USA",          "ter": "0.15%", "description": "500 plus grandes entreprises américaines"},
    {"ticker": "PANX.PA", "name": "Amundi NASDAQ-100",    "zone": "USA Tech",     "ter": "0.23%", "description": "100 valeurs tech du NASDAQ via swap"},
    {"ticker": "C40.PA",  "name": "Amundi CAC 40",        "zone": "France",       "ter": "0.25%", "description": "40 plus grandes capitalisations françaises"},
    {"ticker": "PAEEM.PA","name": "Amundi MSCI Emerging", "zone": "Émergents",    "ter": "0.20%", "description": "Marchés émergents via swap (PEA éligible)"},
    {"ticker": "CE9.PA",  "name": "Amundi MSCI Europe",   "zone": "Europe",       "ter": "0.15%", "description": "Actions européennes diversifiées, réplication physique"},
    {"ticker": "RS2K.PA", "name": "Amundi Russell 2000",  "zone": "USA Small Cap","ter": "0.35%", "description": "Petites capitalisations américaines via swap"},
    {"ticker": "UST.PA",  "name": "Lyxor MSCI USA",       "zone": "USA Large Cap","ter": "0.15%", "description": "Grandes capitalisations américaines via swap"},
]

# Actions européennes éligibles PEA — Euronext Paris (CAC 40 / SBF 120)
STOCK_PEA_UNIVERSE = [
    {"ticker": "MC.PA",  "name": "LVMH",              "sector": "Luxe"},
    {"ticker": "TTE.PA", "name": "TotalEnergies",     "sector": "Énergie"},
    {"ticker": "AIR.PA", "name": "Airbus",            "sector": "Aéronautique"},
    {"ticker": "SAN.PA", "name": "Sanofi",            "sector": "Pharmacie"},
    {"ticker": "BNP.PA", "name": "BNP Paribas",      "sector": "Banque"},
    {"ticker": "CS.PA",  "name": "AXA",               "sector": "Assurance"},
    {"ticker": "OR.PA",  "name": "L'Oréal",           "sector": "Cosmétiques"},
    {"ticker": "DSY.PA", "name": "Dassault Systèmes", "sector": "Technologie"},
    {"ticker": "CAP.PA", "name": "Capgemini",         "sector": "IT Services"},
    {"ticker": "SAF.PA", "name": "Safran",            "sector": "Aéronautique"},
    {"ticker": "SU.PA",  "name": "Schneider Electric","sector": "Électrique"},
    {"ticker": "RMS.PA", "name": "Hermès",            "sector": "Luxe"},
    {"ticker": "AI.PA",  "name": "Air Liquide",       "sector": "Chimie"},
    {"ticker": "EL.PA",  "name": "EssilorLuxottica",  "sector": "Santé"},
    {"ticker": "SGO.PA", "name": "Saint-Gobain",      "sector": "Matériaux"},
]

_MACRO_TICKERS = {
    "vix":   "^VIX",
    "sp500": "SPY",
    "eurusd":"EURUSD=X",
    "bonds": "^TNX",
}

# ─── Indicateurs techniques ──────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + 2 * std, sma - 2 * std


def _score_asset(close: pd.Series) -> tuple:
    """Retourne (score 0-100, label, color hex)."""
    if len(close) < 30:
        return 50, "NEUTRE", "#ffd600"

    rsi_val = float(_rsi(close).iloc[-1])
    sma20   = float(close.rolling(20).mean().iloc[-1])
    sma50   = float(close.rolling(50).mean().iloc[-1])
    _, bb_up, bb_lo = _bollinger(close)
    bb_range = float(bb_up.iloc[-1] - bb_lo.iloc[-1])
    bb_pos   = float((close.iloc[-1] - float(bb_lo.iloc[-1])) / (bb_range + 1e-9) * 100)
    price    = float(close.iloc[-1])

    perf_1m = float((close.iloc[-1] / close.iloc[-21]  - 1) * 100) if len(close) > 21  else 0.0
    perf_3m = float((close.iloc[-1] / close.iloc[-63]  - 1) * 100) if len(close) > 63  else 0.0
    perf_6m = float((close.iloc[-1] / close.iloc[-126] - 1) * 100) if len(close) > 126 else 0.0

    score = 50  # base neutre

    # RSI (25 pts)
    if rsi_val < 30:    score += 25
    elif rsi_val < 40:  score += 15
    elif rsi_val < 50:  score += 5
    elif rsi_val > 70:  score -= 20
    elif rsi_val > 60:  score -= 10

    # Position dans les bandes de Bollinger (20 pts)
    if bb_pos < 20:    score += 20
    elif bb_pos < 40:  score += 10
    elif bb_pos > 80:  score -= 15
    elif bb_pos > 65:  score -= 7

    # Momentum 3 mois (15 pts) — légère correction = opportunité d'entrée LT
    if -20 < perf_3m < -5:  score += 15
    elif -5 <= perf_3m < 0: score += 8
    elif 0 <= perf_3m < 8:  score += 4
    elif perf_3m < -20:     score -= 5   # krach = prudence accrue

    # Tendance de fond 6 mois (10 pts)
    if perf_6m > 5:      score += 10
    elif perf_6m > 0:    score += 5
    elif perf_6m < -25:  score -= 10

    # Structure des moyennes mobiles (10 pts)
    if price > sma20 > sma50:   score += 10
    elif price < sma20 < sma50: score -= 10

    score = max(0, min(100, score))

    if score >= 75:    label, color = "ACHAT FORT", "#00c853"
    elif score >= 60:  label, color = "ACHETER",    "#76ff03"
    elif score >= 40:  label, color = "NEUTRE",     "#ffd600"
    else:              label, color = "ATTENDRE",   "#ff6d00"

    return score, label, color

# ─── Récupération des données ────────────────────────────────────────────────

@st.cache_data(ttl=900)
def get_macro_data() -> dict:
    end   = datetime.now()
    start = end - timedelta(days=400)
    results = {}
    for key, ticker in _MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                continue
            close = df["Close"].squeeze()
            results[key] = {
                "current": float(close.iloc[-1]),
                "1m": float((close.iloc[-1] / close.iloc[-21]  - 1) * 100) if len(close) > 21 else 0.0,
                "3m": float((close.iloc[-1] / close.iloc[-63]  - 1) * 100) if len(close) > 63 else 0.0,
            }
        except Exception:
            pass
    return results


@st.cache_data(ttl=900)
def get_asset_signals(universe_key: str) -> list:
    """Télécharge, score et retourne le top 5 des ETF ou actions."""
    universe = ETF_PEA_UNIVERSE if universe_key == "etf" else STOCK_PEA_UNIVERSE
    end      = datetime.now()
    start    = end - timedelta(days=400)
    tickers  = [item["ticker"] for item in universe]

    try:
        raw = yf.download(
            tickers, start=start, end=end,
            group_by="ticker", progress=False, auto_adjust=True,
        )
    except Exception:
        raw = pd.DataFrame()

    signals = []
    for item in universe:
        ticker = item["ticker"]
        try:
            if len(tickers) == 1:
                close = raw["Close"].squeeze()
            else:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                close = raw[ticker]["Close"].squeeze()

            close = close.dropna()
            if len(close) < 30:
                continue

            score, label, color = _score_asset(close)
            rsi_val  = float(_rsi(close).iloc[-1])
            _, bb_up, bb_lo = _bollinger(close)
            bb_range = float(bb_up.iloc[-1] - bb_lo.iloc[-1])
            bb_pos   = float((close.iloc[-1] - float(bb_lo.iloc[-1])) / (bb_range + 1e-9) * 100)
            price    = float(close.iloc[-1])
            perf_1m  = float((close.iloc[-1] / close.iloc[-21]  - 1) * 100) if len(close) > 21  else 0.0
            perf_3m  = float((close.iloc[-1] / close.iloc[-63]  - 1) * 100) if len(close) > 63  else 0.0
            perf_6m  = float((close.iloc[-1] / close.iloc[-126] - 1) * 100) if len(close) > 126 else 0.0

            chart_df = close.tail(180).reset_index()
            chart_df.columns = ["Date", "Close"]

            signals.append({
                **item,
                "price":   price,
                "score":   score,
                "label":   label,
                "color":   color,
                "rsi":     rsi_val,
                "bb_pos":  bb_pos,
                "perf_1m": perf_1m,
                "perf_3m": perf_3m,
                "perf_6m": perf_6m,
                "chart":   chart_df,
            })
        except Exception:
            continue

    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals[:5]

# ─── Analyse IA du marché ─────────────────────────────────────────────────────

@st.cache_data(ttl=21600)  # 6 h — le contexte géopolitique évolue lentement
def get_market_analysis(vix: float, sp500_3m: float, eurusd: float, bonds: float, date_str: str) -> dict:
    try:
        client = _gemini.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        prompt = f"""Tu es un analyste financier senior spécialisé en gestion de patrimoine. Date : {date_str}.

Données de marché en temps réel :
- VIX (volatilité implicite) : {vix:.1f} → {'stress élevé' if vix > 25 else 'modéré' if vix > 15 else 'faible'}
- Performance S&P 500 sur 3 mois : {sp500_3m:+.1f}%
- EUR/USD : {eurusd:.4f}
- Taux obligataire US 10 ans : {bonds:.2f}%

En tenant compte de ces données et du contexte économique et géopolitique mondial actuel, génère une analyse JSON avec exactement ces champs :
- "regime" : régime de marché actuel en 2-4 mots (ex : "Correction technique", "Bull run tardif", "Bear market", "Consolidation latérale")
- "contexte" : 2-3 phrases décrivant le contexte macro et géopolitique du moment (tensions, politique monétaire, thèmes dominants)
- "risques" : liste de 3 risques clés pour les marchés (max 12 mots chacun)
- "opportunites" : liste de 3 opportunités pour un investisseur PEA long terme (max 12 mots chacun)
- "verdict" : 1 phrase de recommandation concrète pour un investisseur PEA long terme (DCA, patience, secteurs favoris…)
- "score_macro" : entier 0-100 (100 = conditions idéales pour investir, 0 = très défavorable)

Réponds UNIQUEMENT avec le JSON brut, sans markdown ni balises de code."""

        for model in _GEMINI_MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                text = response.text.strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.split("```")[0]
                return json.loads(text.strip())
            except Exception as e:
                err = str(e)
                if any(x in err for x in ("404", "NOT_FOUND", "503", "UNAVAILABLE", "overloaded")):
                    continue
                break
    except Exception:
        pass

    return {
        "regime":       "Analyse indisponible",
        "contexte":     "API d'analyse non accessible. Les signaux techniques ci-dessous restent fiables.",
        "risques":      ["Données macro non accessibles"],
        "opportunites": ["Consultez les signaux techniques ci-dessous"],
        "verdict":      "Basez votre décision sur les indicateurs techniques affichés.",
        "score_macro":  50,
    }


# ─── Portefeuille Trade Republic (import via screenshot) ─────────────────────

_TR_PORTFOLIO_FILE = "tr_portfolio.json"
_TR_REPO = "YassineZak/xag-advisor"


def parse_tr_statement(image_bytes: bytes, mime_type: str = "image/png") -> dict | None:
    """
    Parse une capture d'écran de relevé Trade Republic ("Valeur nette" /
    "État du patrimoine net") via Gemini Vision.
    Retourne {date, cash_eur, holdings: [...]} ou None.
    """
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("Clé GOOGLE_API_KEY manquante dans les secrets Streamlit.")
        return None

    prompt = """Tu es un analyste financier. Cette image est une capture d'écran d'un relevé "Valeur nette" / "État du patrimoine net" de Trade Republic.

Extrais les données au format JSON STRICT (réponds UNIQUEMENT avec le JSON brut, sans markdown, sans commentaire) :

{
  "date": "YYYY-MM-DD",
  "cash_eur": 0.00,
  "holdings": [
    {
      "isin": "FR...",
      "name": "Nom du titre tel qu'affiché",
      "qty": 0,
      "snapshot_price": 0.00,
      "snapshot_value": 0.00,
      "yf_ticker": "..."
    }
  ]
}

Règles :
- "cash_eur" = solde Espèces / Compte courant en EUR
- "qty" = nombre de pièces / parts détenues
- "snapshot_price" = prix par pièce affiché à droite (EUR)
- "snapshot_value" = valeur EUR totale affichée pour la ligne
- "yf_ticker" = ticker Yahoo Finance pour le pricing live. Mappings connus :
    * Amundi PEA Monde MSCI World UCITS ETF Acc (ISIN FR001400U5Q4) → "WPEA.PA"
    * Amundi Stoxx Europe 600 PEA UCITS ETF Acc (ISIN FR0011550193) → "C6E.PA"
    * Amundi PEA S&P 500 UCITS ETF Acc (ISIN FR0011871128) → "PE500.PA"
    * Amundi PEA NASDAQ-100 UCITS ETF Acc → "PANX.PA"
    * Amundi PEA MSCI Emerging Markets → "PAEEM.PA"
    Sinon déduis-le du nom et de la zone (Euronext Paris, suffixe .PA). Si vraiment incertain, mets null.

N'invente pas de positions, n'extrais que ce qui est lisible."""

    try:
        from google.genai import types
        client = _gemini.Client(api_key=api_key)
        img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        for model in _GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[img_part, prompt],
                )
                text = (response.text or "").strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.split("```")[0]
                return json.loads(text.strip())
            except Exception as e:
                err = str(e)
                if any(x in err for x in ("404", "NOT_FOUND", "503", "UNAVAILABLE", "overloaded")):
                    continue
                st.error(f"Erreur Gemini ({model}) : {e}")
                return None
    except Exception as e:
        st.error(f"Erreur d'analyse : {e}")
        return None

    return None


@st.cache_data(ttl=300)
def load_tr_portfolio() -> dict:
    """Charge tr_portfolio.json depuis GitHub. Cache 5 min."""
    try:
        from github import Github
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            return {"cash_eur": 0.0, "holdings": [], "last_updated": None}
        g = Github(token)
        repo = g.get_repo(_TR_REPO)
        f = repo.get_contents(_TR_PORTFOLIO_FILE)
        return json.loads(f.decoded_content)
    except Exception:
        return {"cash_eur": 0.0, "holdings": [], "last_updated": None}


def save_tr_portfolio(data: dict) -> bool:
    """Écrit tr_portfolio.json sur GitHub (create ou update)."""
    try:
        from github import Github
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            st.error("GITHUB_TOKEN manquant dans les secrets.")
            return False
        g = Github(token)
        repo = g.get_repo(_TR_REPO)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        try:
            f = repo.get_contents(_TR_PORTFOLIO_FILE)
            repo.update_file(_TR_PORTFOLIO_FILE, "Update Trade Republic portfolio", content, f.sha)
        except Exception:
            repo.create_file(_TR_PORTFOLIO_FILE, "Create Trade Republic portfolio", content)
        return True
    except Exception as e:
        st.error(f"Erreur GitHub : {e}")
        return False


@st.cache_data(ttl=300)
def _get_tr_live_prices(tickers: tuple) -> dict:
    """Récupère les prix EUR live via yfinance. Cache 5 min."""
    prices = {}
    for t in tickers:
        if not t:
            continue
        try:
            p = yf.Ticker(t).fast_info.last_price
            if p and p > 0:
                prices[t] = float(p)
        except Exception:
            pass
    return prices


def get_tr_live_value() -> dict:
    """
    Combine portfolio importé + prix live yfinance.
    Retourne {cash_eur, savings_eur, total_eur, holdings_detail, last_updated, has_data}.
    Pour chaque holding : prix live si yf_ticker fonctionne, sinon snapshot.
    """
    portfolio = load_tr_portfolio()
    holdings = portfolio.get("holdings", []) or []

    tickers = tuple(h.get("yf_ticker") for h in holdings if h.get("yf_ticker"))
    live_prices = _get_tr_live_prices(tickers) if tickers else {}

    detail = []
    savings = 0.0
    for h in holdings:
        ticker = h.get("yf_ticker")
        snap_price = float(h.get("snapshot_price", 0) or 0)
        snap_value = float(h.get("snapshot_value", 0) or 0)
        qty = float(h.get("qty", 0) or 0)

        live_price = live_prices.get(ticker, 0) if ticker else 0
        if live_price > 0:
            value = qty * live_price
            price = live_price
            is_live = True
        else:
            value = snap_value if snap_value > 0 else qty * snap_price
            price = snap_price
            is_live = False

        savings += value
        detail.append({
            "isin": h.get("isin", ""),
            "name": h.get("name", ""),
            "qty": qty,
            "price": price,
            "snapshot_price": snap_price,
            "value": value,
            "is_live": is_live,
            "ticker": ticker or "",
        })

    cash = float(portfolio.get("cash_eur", 0) or 0)
    return {
        "cash_eur": cash,
        "savings_eur": savings,
        "total_eur": cash + savings,
        "holdings_detail": detail,
        "last_updated": portfolio.get("last_updated"),
        "imported_at": portfolio.get("imported_at"),
        "has_data": bool(holdings) or cash > 0,
    }


def _render_tr_portfolio() -> None:
    """Bloc UI : balances Trade Republic + uploader de relevé."""
    st.markdown("## 💼 Mon portefeuille Trade Republic")

    tr = get_tr_live_value()

    if tr["has_data"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("💵 Espèces", f"{tr['cash_eur']:,.2f} €", help="Solde du compte courant Trade Republic")
        n_live = sum(1 for h in tr["holdings_detail"] if h["is_live"])
        n_total = len(tr["holdings_detail"])
        savings_help = (
            f"Valorisation live ({n_live}/{n_total} positions via yfinance, reste snapshot)"
            if n_live < n_total else "Valorisation live via yfinance"
        ) if n_total else "Aucune position importée"
        c2.metric("📈 Plan d'Épargne", f"{tr['savings_eur']:,.2f} €", help=savings_help)
        c3.metric("Total Trade Republic", f"{tr['total_eur']:,.2f} €")

        if tr.get("last_updated"):
            st.caption(f"Dernier relevé importé : **{tr['last_updated']}**")

        if tr["holdings_detail"]:
            with st.expander("📊 Détail des positions PEA", expanded=True):
                rows = []
                for h in tr["holdings_detail"]:
                    if h["is_live"] and h["snapshot_price"] > 0:
                        diff_pct = (h["price"] - h["snapshot_price"]) / h["snapshot_price"] * 100
                        diff_str = f"{diff_pct:+.2f}%"
                    else:
                        diff_str = "—"
                    qty_str = f"{int(h['qty'])}" if abs(h["qty"] - int(h["qty"])) < 1e-6 else f"{h['qty']:.4f}"
                    rows.append({
                        "Titre": h["name"],
                        "ISIN": h["isin"],
                        "Pièces": qty_str,
                        "Prix unité (€)": f"{h['price']:.4f}" if h["price"] < 10 else f"{h['price']:.2f}",
                        "Valeur (€)": f"{h['value']:,.2f}",
                        "Δ depuis relevé": diff_str,
                        "Source": "🟢 Live" if h["is_live"] else "📸 Snapshot",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun relevé Trade Republic importé. Charge une capture d'écran ci-dessous pour démarrer.")

    with st.expander("📤 Importer un nouveau relevé Trade Republic", expanded=not tr["has_data"]):
        st.caption(
            "Dans l'app Trade Republic, ouvre **Profil → Valeur nette / État du patrimoine net**, "
            "fais une capture d'écran et upload-la ci-dessous. Gemini Vision extraira "
            "automatiquement ton solde Espèces et tes positions PEA."
        )

        uploaded = st.file_uploader(
            "Capture d'écran du relevé",
            type=["png", "jpg", "jpeg", "webp"],
            key="tr_upload",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            col_img, col_action = st.columns([1, 1])
            with col_img:
                st.image(uploaded, caption="Aperçu", use_container_width=True)
            with col_action:
                st.markdown("&nbsp;")
                if st.button("🔍 Analyser et sauvegarder", type="primary", use_container_width=True, key="tr_parse"):
                    with st.spinner("Analyse Gemini en cours..."):
                        image_bytes = uploaded.getvalue()
                        mime = uploaded.type or "image/png"
                        parsed = parse_tr_statement(image_bytes, mime)

                    if parsed is None or not isinstance(parsed, dict):
                        st.error("Échec de l'analyse. Réessaie ou vérifie la lisibilité de l'image.")
                    else:
                        payload = {
                            "last_updated": parsed.get("date") or datetime.now().strftime("%Y-%m-%d"),
                            "imported_at": datetime.now().isoformat(),
                            "cash_eur": float(parsed.get("cash_eur", 0) or 0),
                            "holdings": [
                                {
                                    "isin": str(h.get("isin", "")),
                                    "name": str(h.get("name", "")),
                                    "qty": float(h.get("qty", 0) or 0),
                                    "snapshot_price": float(h.get("snapshot_price", 0) or 0),
                                    "snapshot_value": float(h.get("snapshot_value", 0) or 0),
                                    "yf_ticker": h.get("yf_ticker") or None,
                                }
                                for h in parsed.get("holdings", []) if isinstance(h, dict)
                            ],
                        }

                        st.success(f"✅ Analyse réussie — {len(payload['holdings'])} position(s) détectée(s).")
                        with st.expander("Aperçu du parsing JSON", expanded=False):
                            st.json(payload)

                        if save_tr_portfolio(payload):
                            st.success("💾 Sauvegardé sur GitHub. Rechargement…")
                            st.cache_data.clear()
                            st.rerun()


# ─── Composants UI ───────────────────────────────────────────────────────────

def _perf_html(val: float) -> str:
    if val >= 0:
        return f'<span style="color:#00c853;font-weight:bold">▲ {val:.1f}%</span>'
    return f'<span style="color:#ff6d00;font-weight:bold">▼ {abs(val):.1f}%</span>'


def _render_card(item: dict, rank: int, tag_key: str):
    tag  = item.get(tag_key, "")
    ter  = item.get("ter", "")
    desc = item.get("description", "")

    ter_html  = (f'<span style="background:#1a3a5c;padding:2px 6px;border-radius:4px;'
                 f'font-size:11px;color:#90caf9;margin-left:4px">TER {ter}</span>') if ter else ""
    desc_html = (f'<div style="color:#888;font-size:12px;margin-bottom:8px">{desc}</div>') if desc else ""

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e1e2e,#16213e);border:1px solid #2d2d3d;
                border-radius:12px;padding:16px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
            <div>
                <span style="font-size:19px;font-weight:bold;color:#e0e0e0">#{rank}&nbsp;{item['name']}</span>
                <span style="background:#2d2d3d;padding:2px 8px;border-radius:4px;
                             font-size:11px;color:#aaa;margin-left:6px">{item['ticker']}</span>
                <span style="background:#1e3a3a;padding:2px 8px;border-radius:4px;
                             font-size:11px;color:#80cbc4;margin-left:4px">{tag}</span>{ter_html}
            </div>
            <div style="text-align:right">
                <span style="font-size:15px;font-weight:bold;color:{item['color']}">{item['label']}</span><br>
                <span style="font-size:22px;font-weight:bold;color:#ffffff">{item['price']:.2f} €</span>
            </div>
        </div>{desc_html}
        <div style="background:#1a1a2e;border-radius:6px;padding:3px;margin-bottom:10px">
            <div style="background:{item['color']};width:{item['score']}%;height:7px;border-radius:5px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{item['score']}/100")
    c2.metric("RSI 14j", f"{item['rsi']:.0f}")
    c3.markdown(f"**Perf 1M**<br>{_perf_html(item['perf_1m'])}", unsafe_allow_html=True)
    c4.markdown(f"**Perf 3M**<br>{_perf_html(item['perf_3m'])}", unsafe_allow_html=True)

    if "chart" in item and not item["chart"].empty:
        fig = go.Figure(go.Scatter(
            x=item["chart"]["Date"],
            y=item["chart"]["Close"],
            mode="lines",
            line=dict(color=item["color"], width=1.8),
            fill="tozeroy",
            fillcolor="rgba({},{},{},0.13)".format(
                int(item["color"][1:3], 16),
                int(item["color"][3:5], 16),
                int(item["color"][5:7], 16),
            ),
        ))
        fig.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── Point d'entrée ──────────────────────────────────────────────────────────

def render():
    _render_tr_portfolio()
    st.divider()

    st.markdown("## 📈 ETF & Actions — PEA Long Terme")

    with st.spinner("Chargement des indicateurs macro..."):
        macro = get_macro_data()

    vix    = macro.get("vix",    {}).get("current", 20.0) or 20.0
    sp3m   = macro.get("sp500",  {}).get("3m",       0.0) or 0.0
    eurusd = macro.get("eurusd", {}).get("current",  1.08) or 1.08
    bonds  = macro.get("bonds",  {}).get("current",  4.5)  or 4.5

    # ── Contexte de marché ─────────────────────────────────────────────────
    st.markdown("### 🌍 Contexte de marché & prévisions")

    c1, c2, c3, c4 = st.columns(4)
    vix_delta = macro.get("vix", {}).get("1m")
    c1.metric("VIX", f"{vix:.1f}",
              delta=f"{vix_delta:+.1f}" if vix_delta is not None else None,
              help="Indice de volatilité. <15 = calme · >25 = stress · >35 = panique")
    c2.metric("S&P 500 (3M)", f"{sp3m:+.1f}%")
    c3.metric("EUR/USD", f"{eurusd:.4f}")
    c4.metric("Taux 10Y US", f"{bonds:.2f}%")

    with st.spinner("Analyse IA du contexte géopolitique et macro en cours..."):
        analysis = get_market_analysis(
            vix=round(vix, 1),
            sp500_3m=round(sp3m, 1),
            eurusd=round(eurusd, 4),
            bonds=round(bonds, 2),
            date_str=datetime.now().strftime("%Y-%m-%d"),
        )

    mscore = analysis.get("score_macro", 50)
    rc = "#00c853" if mscore >= 65 else "#ffd600" if mscore >= 45 else "#ff6d00"

    risks_html = "".join(
        f'<div style="color:#ccc;font-size:13px;margin:3px 0">⚠️ {r}</div>'
        for r in analysis.get("risques", [])
    )
    opps_html = "".join(
        f'<div style="color:#ccc;font-size:13px;margin:3px 0">✅ {o}</div>'
        for o in analysis.get("opportunites", [])
    )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a2744,#0d1b2a);border-left:4px solid {rc};
                border-radius:10px;padding:18px;margin:12px 0 22px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span style="font-size:17px;font-weight:bold;color:{rc}">📊 {analysis.get('regime','—')}</span>
            <span style="background:{rc};color:#000;padding:3px 12px;border-radius:20px;
                         font-weight:bold;font-size:13px">Score macro : {mscore}/100</span>
        </div>
        <p style="color:#c8d8e8;margin-bottom:14px;font-size:14px;line-height:1.6">
            {analysis.get('contexte','')}
        </p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px">
            <div>
                <div style="color:#ff8a65;font-weight:bold;margin-bottom:6px;font-size:13px">Risques</div>
                {risks_html}
            </div>
            <div>
                <div style="color:#69f0ae;font-weight:bold;margin-bottom:6px;font-size:13px">Opportunités</div>
                {opps_html}
            </div>
        </div>
        <div style="padding:10px 14px;background:rgba(0,0,0,0.3);border-radius:6px">
            <span style="color:#90caf9;font-weight:bold;font-size:13px">💡 Verdict LT : </span>
            <span style="color:#e0e0e0;font-size:13px">{analysis.get('verdict','')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 5 ETF PEA ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Top 5 ETF éligibles PEA — Long terme")
    st.caption(
        "📌 ETFs éligibles au PEA (réplication synthétique swap ou physique avec ≥75 % d'actifs EEE). "
        "Données avec ~15 min de délai · Classement par score technique."
    )

    with st.spinner("Analyse des ETF PEA..."):
        etf_signals = get_asset_signals("etf")

    if not etf_signals:
        st.warning("Impossible de récupérer les données ETF. Vérifiez votre connexion.")
    else:
        for i, etf in enumerate(etf_signals, 1):
            _render_card(etf, i, tag_key="zone")

    # ── Top 5 Actions PEA ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Top 5 Actions PEA — Long terme")
    st.caption(
        "📌 Actions européennes éligibles PEA, sélectionnées parmi les grandes capitalisations "
        "françaises (CAC 40 / Euronext Paris) · Mêmes critères de scoring que les ETF."
    )

    with st.spinner("Analyse des actions PEA..."):
        stock_signals = get_asset_signals("stock")

    if not stock_signals:
        st.warning("Impossible de récupérer les données des actions.")
    else:
        for i, stock in enumerate(stock_signals, 1):
            _render_card(stock, i, tag_key="sector")

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:24px;padding:12px 16px;background:#1a1a2e;border-radius:8px;
                border:1px solid #2d2d3d;font-size:11px;color:#666;line-height:1.6">
        ⚠️ <strong style="color:#888">Avertissement :</strong> Ces informations sont fournies à titre
        indicatif uniquement et ne constituent pas des conseils en investissement. Les performances
        passées ne préjugent pas des performances futures. Tout investissement comporte un risque de
        perte en capital. Consultez un conseiller financier agréé avant toute décision d'investissement.
    </div>
    """, unsafe_allow_html=True)
