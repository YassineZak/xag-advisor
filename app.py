import streamlit as st
import xag_tab
import btc_tab
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta

COOKIE_HOURS = 8

st.set_page_config(
    page_title="Portfolio Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 1rem; }
    .stMetric { background: #1e1e2e; border-radius: 10px; padding: 10px; }

    /* Onglets — texte toujours visible */
    [data-baseweb="tab"] { color: #cbd5e1 !important; font-size: 1rem !important; font-weight: 600 !important; }
    [data-baseweb="tab"][aria-selected="true"] { color: #f1f5f9 !important; border-bottom: 3px solid #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Cookies (doit être initialisé tôt, avant tout rendu conditionnel) ──────────

_cookie_pw = st.secrets.get("APP_PASSWORD", "default") + "-xag-cookie-v1"
cookies = EncryptedCookieManager(prefix="xag_", password=_cookie_pw)

if not cookies.ready():
    st.stop()

# ── Authentification ──────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Restaurer la session depuis le cookie
if not st.session_state["authenticated"]:
    _stored = cookies.get("auth_expiry")
    if _stored:
        try:
            if datetime.now() < datetime.fromisoformat(_stored):
                st.session_state["authenticated"] = True
        except ValueError:
            pass

if not st.session_state["authenticated"]:
    st.title("🔒 Portfolio Advisor")
    st.markdown("---")
    col, _ = st.columns([1, 2])
    with col:
        with st.form("login_form"):
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
            if submitted:
                if password == st.secrets.get("APP_PASSWORD", ""):
                    expiry = (datetime.now() + timedelta(hours=COOKIE_HOURS)).isoformat()
                    cookies["auth_expiry"] = expiry
                    cookies.save()
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect.")
    st.stop()

# ── Header portfolio ─────────────────────────────────────────────────────────

def _fmt(v, dec=2):
    return f"{v:,.{dec}f} €" if v is not None else "—"

def _pnl_html(current, avg):
    if not avg or avg <= 0 or current is None:
        return ""
    pct = (current - avg) / avg * 100
    color = "#22c55e" if pct >= 0 else "#ef4444"
    sign  = "+" if pct >= 0 else ""
    return f' <span style="color:{color};font-size:0.8rem">({sign}{pct:.1f}%)</span>'

_portfolio        = xag_tab.load_portfolio()
_xag_eur, _, _, _ = xag_tab.get_live_price()
_xag_qty  = _portfolio.get("quantity", 0.0)
_xag_avg  = _portfolio.get("avg_price", 0.0)
_xag_val  = _xag_qty * _xag_eur if _xag_eur else None

_bp       = btc_tab.get_bitpanda_values()
_bp_total = _bp["total_eur"]
_total    = (_xag_val or 0) + _bp_total

# Construire les items crypto/fiat Bitpanda — crypto d'abord, fiat à la fin
_crypto_items_html = []
for _sym, _info in _bp["holdings"].items():
    if _info["type"] == "crypto":
        _val_str = _fmt(_info["value_eur"]) if _info["value_eur"] > 0 else "—"
        _crypto_items_html.append(
            f'<span><b style="color:#94a3b8">{_sym}</b>&nbsp;'
            f'<b style="color:#e2e8f0">{_info["balance"]:.5f}</b>'
            f'&nbsp;→&nbsp;<b style="color:#22c55e">{_val_str}</b></span>'
        )
for _sym, _info in _bp["holdings"].items():
    if _info["type"] == "fiat":
        _crypto_items_html.append(
            f'<span><span style="color:#64748b">{_sym}</span>&nbsp;'
            f'<b style="color:#22c55e">{_info["balance"]:,.2f} €</b></span>'
        )
_sep = '<span style="color:#334155">│</span>'
_crypto_html = f'&nbsp;{_sep}&nbsp;'.join(_crypto_items_html) if _crypto_items_html else '<span style="color:#64748b">—</span>'

st.markdown(f"""
<div style="background:#1e1e2e;border-radius:10px;padding:10px 20px;margin-bottom:14px;
            display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:0.88rem;line-height:1.8">
  <span style="color:#64748b;font-weight:700;font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase">Portefeuille</span>
  <span>🥈&nbsp;<b style="color:#e2e8f0">{_xag_qty:.3f} oz</b>
        <span style="color:#64748b">&nbsp;·&nbsp;{_fmt(_xag_eur)}/oz</span>
        &nbsp;→&nbsp;<b style="color:#22c55e">{_fmt(_xag_val)}</b>{_pnl_html(_xag_eur, _xag_avg)}</span>
  {_sep}
  {_crypto_html}
  {_sep}
  <span style="font-weight:700;color:#94a3b8">Total&nbsp;&nbsp;<span style="color:#fbbf24;font-size:0.95rem">{_fmt(_total)}</span></span>
  <span style="color:#1e293b;font-size:0.65rem;margin-left:auto">v2.1</span>
</div>
""", unsafe_allow_html=True)

# ── App principale ────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🥈 Métaux", "₿ Cryptos"])

with tab1:
    xag_tab.render()

with tab2:
    btc_tab.render()
