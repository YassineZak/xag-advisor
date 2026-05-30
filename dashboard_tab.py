"""
Onglet Dashboard — vue consolidée de l'ensemble des avoirs.

Agrège trois sources sans double comptage :
  • Silver WisdomTree   → parts (relevé Trade Republic) × prix live WisdomTree
  • ETF/PEA + cash TR   → relevé Trade Republic, SILVER EXCLU (compté ci-dessus)
  • Crypto Bitpanda     → soldes × prix live EUR

Évolution jour / mois / année (€ et %) via un historique hybride :
  • backfill approximatif depuis les prix de marché (quantités actuelles),
  • snapshots quotidiens réels stockés dans portfolio_history.json (GitHub),
    qui prennent le pas sur le backfill aux dates où ils existent.

Aucun graphique — uniquement des chiffres et des tableaux.
"""

import json
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

import xag_tab
import btc_tab
import etf_pea_tab

HISTORY_FILE = "portfolio_history.json"
REPO_NAME    = "YassineZak/xag-advisor"


# ── Snapshot live de tous les avoirs ──────────────────────────────────────────

def compute_snapshot() -> dict:
    """Valorise tous les avoirs maintenant. Retourne les composantes + total."""
    # Silver WisdomTree (live)
    holding = xag_tab.get_silver_holding()
    price, _ = xag_tab.get_silver_live()
    if price is None:
        hist = xag_tab.get_silver_history()
        price = float(hist.iloc[-1]) if not hist.empty else 0.0
    silver_qty = holding["qty"]
    silver_eur = silver_qty * price
    silver_snapshot = holding["snapshot_value"]

    # Trade Republic (ETF/PEA + cash), silver exclu pour ne pas le compter 2×
    try:
        tr = etf_pea_tab.get_tr_live_value()
    except Exception:
        tr = {"cash_eur": 0.0, "savings_eur": 0.0}
    cash_eur = float(tr.get("cash_eur", 0) or 0)
    etf_eur  = max(float(tr.get("savings_eur", 0) or 0) - silver_snapshot, 0.0)

    # Crypto Bitpanda
    try:
        bp = btc_tab.get_bitpanda_values()
        crypto_eur = float(bp.get("total_eur", 0) or 0)
        bp_holdings = bp.get("holdings", {})
    except Exception:
        crypto_eur = 0.0
        bp_holdings = {}

    total = silver_eur + etf_eur + cash_eur + crypto_eur
    return {
        "silver_eur": silver_eur,
        "silver_qty": silver_qty,
        "silver_price": price,
        "etf_eur": etf_eur,
        "cash_eur": cash_eur,
        "crypto_eur": crypto_eur,
        "crypto_holdings": bp_holdings,
        "total": total,
        "tr": tr,
    }


# ── Historique réel (snapshots quotidiens sur GitHub) ─────────────────────────

@st.cache_data(ttl=300)
def load_history() -> list:
    try:
        from github import Github
        g    = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(REPO_NAME)
        f    = repo.get_contents(HISTORY_FILE)
        data = json.loads(f.decoded_content)
        return data.get("snapshots", []) if isinstance(data, dict) else (data or [])
    except Exception:
        return []


def append_today_snapshot(snap: dict) -> None:
    """
    Enregistre la valeur du jour UNE seule fois par jour (premier chargement).
    On n'écrit pas à chaque rerun pour éviter une avalanche de commits GitHub :
    la valeur live du jour reste exacte côté affichage via build_merged_series().
    """
    today = date.today().strftime("%Y-%m-%d")
    history = load_history()
    if any(s.get("date") == today for s in history):
        return  # déjà enregistré aujourd'hui → rien à faire
    history.append({
        "date": today,
        "silver": round(snap["silver_eur"], 2),
        "etf": round(snap["etf_eur"], 2),
        "cash": round(snap["cash_eur"], 2),
        "crypto": round(snap["crypto_eur"], 2),
        "total": round(snap["total"], 2),
    })
    try:
        from github import Github, GithubException
        g    = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(REPO_NAME)
        content = json.dumps({"snapshots": history}, indent=2, ensure_ascii=False)
        try:
            f = repo.get_contents(HISTORY_FILE)
            repo.update_file(HISTORY_FILE, "Update portfolio history", content, f.sha)
        except GithubException:
            repo.create_file(HISTORY_FILE, "Create portfolio history", content)
        load_history.clear()
    except Exception:
        pass  # ne jamais casser l'UI pour le logging d'historique


# ── Backfill approximatif depuis les prix de marché ───────────────────────────

@st.cache_data(ttl=3600)
def _price_history_eur(ticker: str, days: int = 400) -> pd.Series:
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d")
        if not hist.empty and "Close" in hist.columns:
            s = hist["Close"].copy()
            s.index = s.index.tz_localize(None).normalize()
            return s.dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def backfill_total_series(snap: dict, days: int = 400) -> pd.Series:
    """
    Reconstruit une série quotidienne approximative de la valeur totale, en
    supposant les quantités actuelles constantes dans le passé. ETF/PEA + cash
    sont figés à leur valeur courante (pas d'historique de snapshot disponible).
    """
    frames = {}

    silver_qty = snap["silver_qty"]
    if silver_qty > 0:
        sh = xag_tab.get_silver_history(days)
        if not sh.empty:
            sh = sh.copy(); sh.index = sh.index.normalize()
            frames["silver"] = sh * silver_qty

    for sym, info in snap.get("crypto_holdings", {}).items():
        if info.get("type") != "crypto":
            continue
        bal = float(info.get("balance", 0) or 0)
        if bal <= 0:
            continue
        ph = _price_history_eur(f"{sym}-EUR", days)
        if not ph.empty:
            frames[f"crypto_{sym}"] = ph * bal

    if not frames:
        return pd.Series(dtype=float)

    df = pd.DataFrame(frames).sort_index().ffill()
    # Composantes sans historique : constantes
    flat = snap["etf_eur"] + snap["cash_eur"]
    # Cash crypto fiat (déjà inclus dans crypto_eur mais pas dans les frames crypto_*)
    crypto_fiat = sum(
        float(i.get("balance", 0) or 0)
        for i in snap.get("crypto_holdings", {}).values()
        if i.get("type") == "fiat"
    )
    total = df.sum(axis=1) + flat + crypto_fiat
    return total.dropna()


def _value_on_or_before(series: pd.Series, target: date):
    """Dernière valeur dont la date est <= target. None si aucune."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(target)
    sub = series[series.index <= ts]
    return float(sub.iloc[-1]) if len(sub) else None


def build_merged_series(snap: dict) -> pd.Series:
    """Backfill recouvert par les snapshots réels (qui priment), + valeur du jour."""
    series = backfill_total_series(snap)
    series = series.copy() if series is not None else pd.Series(dtype=float)

    for s in load_history():
        try:
            d = pd.Timestamp(s["date"]).normalize()
            series.loc[d] = float(s["total"])
        except Exception:
            continue

    # Point du jour = valeur live (la plus fiable)
    series.loc[pd.Timestamp(date.today())] = float(snap["total"])
    return series.sort_index()


# ── Rendu ─────────────────────────────────────────────────────────────────────

def _variation(series: pd.Series, current: float, target: date):
    past = _value_on_or_before(series, target)
    if past is None or past <= 0:
        return "—", None
    eur = current - past
    pct = eur / past * 100
    return f"{eur:+,.2f} €", f"{pct:+.2f}%"


def render():
    col_title, col_refresh = st.columns([5, 1])
    col_title.title("📊 Dashboard — Vue d'ensemble")
    if col_refresh.button("🔄 Rafraîchir", key="dash_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Valorisation de tes avoirs..."):
        snap = compute_snapshot()
        append_today_snapshot(snap)
        series = build_merged_series(snap)

    # ── Total ─────────────────────────────────────────────────────────────────
    today = date.today()
    v_day,  p_day  = _variation(series, snap["total"], today - timedelta(days=1))
    v_mon,  p_mon  = _variation(series, snap["total"], today - timedelta(days=30))
    v_year, p_year = _variation(series, snap["total"], today - timedelta(days=365))

    st.markdown(f"""
    <div style="text-align:center;padding:22px;background:linear-gradient(135deg,#1e1e2e,#16213e);
                border-radius:14px;margin-bottom:14px">
        <div style="color:#94a3b8;font-size:0.8rem;letter-spacing:.1em;text-transform:uppercase">Patrimoine total</div>
        <div style="font-size:2.8rem;font-weight:800;color:#fbbf24">{snap['total']:,.2f} €</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Évolution (chiffres) ───────────────────────────────────────────────────
    st.subheader("📈 Évolution")
    e1, e2, e3 = st.columns(3)
    e1.metric("Jour", v_day,  delta=p_day,  help="Variation vs hier.")
    e2.metric("Mois", v_mon,  delta=p_mon,  help="Variation vs il y a ~30 jours.")
    e3.metric("Année", v_year, delta=p_year, help="Variation vs il y a ~1 an.")
    st.caption(
        "Historique hybride : approximé depuis les prix de marché au démarrage, "
        "puis fiabilisé par un snapshot réel enregistré chaque jour. "
        "Les lignes ETF/PEA et cash sont figées à leur dernière valeur de relevé."
    )

    # ── Évolution (tableau) ────────────────────────────────────────────────────
    evo_rows = [
        {"Période": "Jour (J-1)",  "Variation (€)": v_day,  "Variation (%)": p_day  or "—"},
        {"Période": "Mois (30j)",  "Variation (€)": v_mon,  "Variation (%)": p_mon  or "—"},
        {"Période": "Année (12m)", "Variation (€)": v_year, "Variation (%)": p_year or "—"},
    ]
    st.dataframe(pd.DataFrame(evo_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Répartition par classe d'actif ─────────────────────────────────────────
    st.subheader("🧱 Répartition")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("🥈 Silver WisdomTree", f"{snap['silver_eur']:,.2f} €",
              help=f"{snap['silver_qty']:g} parts × {snap['silver_price']:.2f} € (prix live).")
    b2.metric("📈 ETF / PEA", f"{snap['etf_eur']:,.2f} €", help="Positions Trade Republic hors silver.")
    b3.metric("💵 Cash TR", f"{snap['cash_eur']:,.2f} €")
    b4.metric("₿ Crypto", f"{snap['crypto_eur']:,.2f} €", help="Avoirs Bitpanda valorisés en EUR.")

    total = snap["total"] or 1.0
    rep_rows = [
        {"Classe": "🥈 Silver WisdomTree", "Valeur (€)": f"{snap['silver_eur']:,.2f}", "Part": f"{snap['silver_eur']/total*100:.1f}%"},
        {"Classe": "📈 ETF / PEA",         "Valeur (€)": f"{snap['etf_eur']:,.2f}",    "Part": f"{snap['etf_eur']/total*100:.1f}%"},
        {"Classe": "💵 Cash Trade Republic","Valeur (€)": f"{snap['cash_eur']:,.2f}",   "Part": f"{snap['cash_eur']/total*100:.1f}%"},
        {"Classe": "₿ Crypto Bitpanda",    "Valeur (€)": f"{snap['crypto_eur']:,.2f}", "Part": f"{snap['crypto_eur']/total*100:.1f}%"},
    ]
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

    st.caption(
        f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} · "
        "Sources : Trade Republic (relevé importé), Bitpanda API, Yahoo Finance · "
        "⚠️ Outil d'analyse — pas un conseil financier."
    )

    st.divider()
    _render_widget_section(snap, series)


# ── Widget iPhone (gist secret lu par Scriptable) ─────────────────────────────

def _render_widget_section(snap: dict, series: pd.Series) -> None:
    """
    Pousse le récap vers un gist secret (lu par le widget Scriptable iPhone)
    et affiche l'URL + la procédure d'installation.

    Publication automatique une fois par session (pour éviter une avalanche de
    révisions de gist à chaque rerun), + bouton pour forcer la mise à jour.
    """
    import widget_export

    hist   = xag_tab.get_silver_history()
    signal = xag_tab.compute_silver_signal(hist, snap.get("silver_price", 0))
    today  = date.today()
    v_day, p_day = _variation(series, snap["total"], today - timedelta(days=1))

    with st.expander("📱 Widget iPhone (Scriptable)", expanded=False):
        if not st.session_state.get("_widget_pushed"):
            st.session_state["_widget_res"] = widget_export.push_widget(
                snap, signal, v_day, p_day)
            st.session_state["_widget_pushed"] = True

        if st.button("🔄 Mettre à jour le widget maintenant", key="widget_push",
                     use_container_width=True):
            st.session_state["_widget_res"] = widget_export.push_widget(
                snap, signal, v_day, p_day)

        res = st.session_state.get("_widget_res", {})
        if res.get("ok"):
            st.success("✅ Récap publié dans le gist secret.")
            st.caption("URL à coller dans le champ `WIDGET_URL` du script Scriptable :")
            st.code(res["url"], language=None)
        elif res.get("error"):
            st.error(f"Échec de publication : {res['error']}")
            st.caption(
                "Le plus souvent, ton `GITHUB_TOKEN` n'a pas le scope `gist`. "
                "Régénère un token GitHub avec la case **gist** cochée et "
                "remplace-le dans les secrets Streamlit."
            )

        st.markdown(
            "**Mise en place (1 fois) :**\n"
            "1. Installe l'app **Scriptable** (gratuite, App Store).\n"
            "2. Récupère le script `scriptable_widget.js` du repo, "
            "colle-le dans un nouveau script Scriptable.\n"
            "3. Remplace `WIDGET_URL` par l'URL ci-dessus, renomme « Patrimoine ».\n"
            "4. Écran d'accueil **ou** verrouillage → appui long → **+** → "
            "**Scriptable** → choisis la taille → édite le widget → "
            "Script = « Patrimoine ».\n\n"
            "Le widget se rafraîchit tout seul (cadence gérée par iOS, ~15-30 min). "
            "Les chiffres reflètent ton dernier passage ici."
        )
