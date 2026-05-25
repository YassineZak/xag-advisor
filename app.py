import hmac
import hashlib
import time
import streamlit as st
from streamlit_cookies_controller import CookieController
import xag_tab
import btc_tab
import etf_pea_tab

SESSION_DAYS  = 30
SESSION_SEC   = SESSION_DAYS * 24 * 3600
_AUTH_COOKIE  = "xag_auth"

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
    [data-baseweb="tab"] { color: #cbd5e1 !important; font-size: 1rem !important; font-weight: 600 !important; }
    [data-baseweb="tab"][aria-selected="true"] { color: #f1f5f9 !important; border-bottom: 3px solid #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Auth via cookie HTTP (persiste sur PWA iOS, même après force-quit) ────────

_cookies = CookieController(key="auth_cookies")


def _sign(ts: int) -> str:
    pw  = st.secrets.get("APP_PASSWORD", "").encode()
    msg = f"{ts}:xag-v2".encode()
    return hmac.new(pw, msg, hashlib.sha256).hexdigest()[:24]


def _make_token() -> str:
    ts = int(time.time())
    return f"{_sign(ts)}.{ts}"


def _is_token_valid(token: str) -> bool:
    if not token or "." not in token:
        return False
    try:
        tok, ts_str = token.rsplit(".", 1)
        ts = int(ts_str)
        if time.time() - ts > SESSION_SEC:
            return False
        return hmac.compare_digest(tok, _sign(ts))
    except Exception:
        return False


def _check_auth_from_cookie() -> bool:
    token = _cookies.get(_AUTH_COOKIE)
    return _is_token_valid(token) if token else False

# ── Authentification ──────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):
    st.session_state["authenticated"] = _check_auth_from_cookie()

if not st.session_state["authenticated"]:
    st.title("🔒 Portfolio Advisor")
    st.markdown("---")
    col, _ = st.columns([1, 2])
    with col:
        with st.form("login_form"):
            password  = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
            if submitted:
                if password == st.secrets.get("APP_PASSWORD", ""):
                    token = _make_token()
                    _cookies.set(
                        _AUTH_COOKIE,
                        token,
                        max_age=SESSION_SEC,
                        same_site="lax",
                        secure=True,
                    )
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
    pct   = (current - avg) / avg * 100
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

try:
    _tr = etf_pea_tab.get_tr_live_value()
except Exception:
    _tr = {"cash_eur": 0.0, "savings_eur": 0.0, "total_eur": 0.0, "has_data": False}
_tr_total = _tr.get("total_eur", 0.0)

_total    = (_xag_val or 0) + _bp_total + _tr_total

_sep = '<span style="color:#334155">│</span>'
_sections: list[str] = []

# Section XAG (uniquement si on en détient)
if _xag_qty and _xag_qty > 0:
    _sections.append(
        f'<span>🥈&nbsp;<b style="color:#e2e8f0">{_xag_qty:.3f} oz</b>'
        f'<span style="color:#64748b">&nbsp;·&nbsp;{_fmt(_xag_eur)}/oz</span>'
        f'&nbsp;→&nbsp;<b style="color:#22c55e">{_fmt(_xag_val)}</b>{_pnl_html(_xag_eur, _xag_avg)}</span>'
    )

# Cryptos (uniquement celles avec valeur > 0)
_crypto_items_html = []
for _sym, _info in _bp["holdings"].items():
    if _info["type"] == "crypto" and _info["value_eur"] > 0:
        _crypto_items_html.append(
            f'<span><b style="color:#94a3b8">{_sym}</b>&nbsp;'
            f'<b style="color:#e2e8f0">{_info["balance"]:.5f}</b>'
            f'&nbsp;→&nbsp;<b style="color:#22c55e">{_fmt(_info["value_eur"])}</b></span>'
        )
for _sym, _info in _bp["holdings"].items():
    if _info["type"] == "fiat" and _info["balance"] > 0:
        _crypto_items_html.append(
            f'<span><span style="color:#64748b">{_sym}</span>&nbsp;'
            f'<b style="color:#22c55e">{_info["balance"]:,.2f} €</b></span>'
        )
if _crypto_items_html:
    _sections.append(f'&nbsp;{_sep}&nbsp;'.join(_crypto_items_html))

# Trade Republic (PEA et/ou Espèces, uniquement si > 0)
_tr_parts = []
if _tr.get("savings_eur", 0) > 0:
    _tr_parts.append(
        f'📈&nbsp;<span style="color:#94a3b8">PEA</span>&nbsp;'
        f'<b style="color:#22c55e">{_fmt(_tr["savings_eur"])}</b>'
    )
if _tr.get("cash_eur", 0) > 0:
    _tr_parts.append(
        f'💵&nbsp;<b style="color:#22c55e">{_fmt(_tr["cash_eur"])}</b>'
    )
if _tr_parts:
    _sections.append(f'<span>{"&nbsp;·&nbsp;".join(_tr_parts)}</span>')

# Total (toujours affiché)
_sections.append(
    f'<span style="font-weight:700;color:#94a3b8">Total&nbsp;&nbsp;'
    f'<span style="color:#fbbf24;font-size:0.95rem">{_fmt(_total)}</span></span>'
)

_body = f'&nbsp;{_sep}&nbsp;'.join(_sections) if _sections else '<span style="color:#64748b">Portefeuille vide</span>'

st.markdown(f"""
<div style="background:#1e1e2e;border-radius:10px;padding:10px 20px;margin-bottom:14px;
            display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:0.88rem;line-height:1.8">
  <span style="color:#64748b;font-weight:700;font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase">Portefeuille</span>
  {_body}
  <span style="color:#475569;font-size:0.65rem;margin-left:auto">v2.6</span>
</div>
""", unsafe_allow_html=True)

# ── App principale ────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🥈 Métaux", "₿ Cryptos", "📈 ETF & Actions PEA"])

with tab1:
    xag_tab.render()

with tab2:
    btc_tab.render()

with tab3:
    etf_pea_tab.render()

# ── Pied de page : déconnexion ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ Compte", expanded=False):
    if st.button("🚪 Se déconnecter", key="logout_btn"):
        _cookies.remove(_AUTH_COOKIE)
        st.session_state["authenticated"] = False
        st.rerun()
