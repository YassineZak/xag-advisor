import hmac
import hashlib
import time
import streamlit as st
import streamlit.components.v1 as components
import xag_tab
import btc_tab
import etf_pea_tab
import dashboard_tab

SESSION_DAYS = 365  # 1 an
SESSION_SEC  = SESSION_DAYS * 24 * 3600
_AUTH_PARAM  = "auth"
_STORAGE_KEY = "xag_auth_v3"

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

# ── Auth via query param dans l'URL ──────────────────────────────────────────
# Pourquoi pas cookies/localStorage ? iOS PWA en mode standalone applique des
# politiques ITP très strictes : les cookies et localStorage sont souvent purgés
# au force-quit. Solution la plus fiable : laisser le token dans l'URL — si tu
# bookmarkes la PWA APRÈS login, l'URL contient le token et la PWA reste
# connectée tant que le token est valide (1 an par défaut).

def _sign(ts: int) -> str:
    pw  = st.secrets.get("APP_PASSWORD", "").encode()
    msg = f"{ts}:xag-v2".encode()
    return hmac.new(pw, msg, hashlib.sha256).hexdigest()[:24]

def _check_auth() -> bool:
    val = st.query_params.get(_AUTH_PARAM, "")
    if not val or "." not in val:
        return False
    try:
        tok, ts_str = val.rsplit(".", 1)
        ts = int(ts_str)
        if time.time() - ts > SESSION_SEC:
            return False
        return hmac.compare_digest(tok, _sign(ts))
    except Exception:
        return False

def _set_auth():
    ts = int(time.time())
    st.query_params[_AUTH_PARAM] = f"{_sign(ts)}.{ts}"


def _inject_storage_fallback_js():
    """
    Restauration agressive multi-storage : localStorage + cookies.
    Pour les cas où le manifest Streamlit écrase l'URL bookmarkée de la PWA iOS.
    Tente d'accéder au window.top (parent réel) pour partager le storage.
    """
    components.html(f"""
    <script>
    (function() {{
      const KEY = '{_STORAGE_KEY}';
      const PARAM = '{_AUTH_PARAM}';
      const MAX_AGE_MS = {SESSION_SEC} * 1000;

      function getCtx() {{
        try {{
          if (window.top && window.top.localStorage) return window.top;
        }} catch (e) {{}}
        try {{
          if (window.parent && window.parent.localStorage) return window.parent;
        }} catch (e) {{}}
        return window;
      }}

      function isValid(t) {{
        if (!t || !t.includes('.')) return false;
        const ts = parseInt(t.split('.').pop()) * 1000;
        return !isNaN(ts) && (Date.now() - ts) < MAX_AGE_MS;
      }}

      function readToken(ctx) {{
        try {{
          const v = ctx.localStorage.getItem(KEY);
          if (v && isValid(v)) return v;
        }} catch (e) {{}}
        try {{
          const m = ctx.document.cookie.match(new RegExp('(?:^|; )' + KEY + '=([^;]*)'));
          if (m) {{
            const v = decodeURIComponent(m[1]);
            if (isValid(v)) return v;
          }}
        }} catch (e) {{}}
        return null;
      }}

      function writeToken(ctx, v) {{
        try {{ ctx.localStorage.setItem(KEY, v); }} catch (e) {{}}
        try {{
          const exp = new Date(Date.now() + MAX_AGE_MS).toUTCString();
          ctx.document.cookie = KEY + '=' + encodeURIComponent(v) +
            '; expires=' + exp + '; path=/; SameSite=Lax; Secure';
        }} catch (e) {{}}
      }}

      try {{
        const ctx = getCtx();
        const loc = ctx.location;
        const params = new URLSearchParams(loc.search);
        const urlAuth = params.get(PARAM);

        if (urlAuth && isValid(urlAuth)) {{
          writeToken(ctx, urlAuth);
        }} else {{
          const stored = readToken(ctx);
          if (stored) {{
            params.set(PARAM, stored);
            loc.replace(loc.origin + loc.pathname + '?' + params.toString());
          }}
        }}
      }} catch (e) {{ console.error('Auth restore failed:', e); }}
    }})();
    </script>
    """, height=0)


def _get_current_url_diagnostic() -> str:
    """Renvoie l'URL courante détectable côté serveur (pour diagnostic PWA)."""
    try:
        url = st.context.url
        if url:
            return url
    except Exception:
        pass
    try:
        host = st.context.headers.get("Host", "?")
        params = "&".join(f"{k}={v}" for k, v in st.query_params.to_dict().items())
        return f"https://{host}/" + (f"?{params}" if params else "")
    except Exception:
        return "(URL non détectable côté serveur)"


# Injection au plus tôt : tente de restaurer le token avant l'écran de login
_inject_storage_fallback_js()

# ── Authentification ──────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):
    st.session_state["authenticated"] = _check_auth()

if not st.session_state["authenticated"]:
    st.title("🔒 Portfolio Advisor")
    st.markdown("---")

    # Diagnostic : montre l'URL avec laquelle la PWA a été lancée
    current_url = _get_current_url_diagnostic()
    has_auth_in_url = bool(st.query_params.get(_AUTH_PARAM, ""))
    with st.expander("🔍 Diagnostic (utile pour debug PWA iPhone)", expanded=False):
        st.caption("URL actuelle détectée par le serveur :")
        st.code(current_url, language=None)
        if has_auth_in_url:
            st.warning("Token présent dans l'URL mais invalide (expiré ou corrompu).")
        else:
            st.info(
                "**Pas de token dans l'URL.** Si tu viens de relancer une PWA bookmarkée, "
                "ça signifie que iOS / Streamlit a écrasé l'URL d'origine. "
                "Le storage local n'a pas non plus de token valide. → re-login obligatoire."
            )

    col, _ = st.columns([1, 2])
    with col:
        with st.form("login_form"):
            password  = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)
            if submitted:
                if password == st.secrets.get("APP_PASSWORD", ""):
                    _set_auth()
                    st.session_state["authenticated"] = True
                    st.session_state["just_logged_in"] = True
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect.")
    st.stop()

# ── Helpers : construction de l'URL d'auth complète ──────────────────────────

def _build_authed_url() -> str:
    """Reconstruit l'URL complète avec le token courant (pour bookmark PWA)."""
    token = st.query_params.get(_AUTH_PARAM, "")
    # Essaie de récupérer l'host depuis le contexte ; fallback Streamlit Cloud
    host = "xag-advisor.streamlit.app"
    try:
        ctx_host = st.context.headers.get("Host", "")
        if ctx_host:
            host = ctx_host
    except Exception:
        pass
    return f"https://{host}/?{_AUTH_PARAM}={token}"


def _render_pwa_install_block():
    """Affiche les instructions de (ré)installation de la PWA iPhone."""
    full_url = _build_authed_url()
    st.markdown("### 📱 Pour rester connecté sur la PWA iPhone")
    st.markdown(
        "**Pourquoi tu dois faire ça ?** iOS fige l'URL de la PWA au moment de "
        "l'installation. Si tu installes la PWA AVANT de te connecter (ou si tu "
        "te connectes dans l'app), le bookmark pointe sur l'URL **sans token** → "
        "à chaque kill, tu retombes sur l'écran de login. "
        "Il faut donc réinstaller la PWA depuis Safari APRÈS être connecté."
    )
    st.markdown("**Étapes (1 minute) :**")
    st.markdown(
        "1. **Désinstalle l'actuelle PWA** : long-press l'icône sur ton écran "
        "d'accueil → *Retirer l'app* → *Supprimer du dock*\n"
        "2. **Tap le bouton bleu ci-dessous** pour ouvrir l'URL d'auth dans Safari :"
    )
    st.markdown(
        f'<a href="{full_url}" target="_blank" rel="noopener" '
        f'style="display:inline-block;background:#3b82f6;color:white;padding:12px 24px;'
        f'border-radius:8px;text-decoration:none;font-weight:600;margin:8px 0;'
        f'font-size:0.95rem">🌐 Ouvrir dans Safari</a>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ou copie cette URL et colle-la dans Safari manuellement :"
    )
    st.code(full_url, language=None)
    st.markdown(
        "3. **Dans Safari**, vérifie que l'URL contient bien `?auth=...`\n"
        "4. Tap **Partager** (carré avec flèche ↑ en bas de l'écran) → "
        "scroll → **Sur l'écran d'accueil** → **Ajouter** (en haut à droite)\n"
        "5. C'est fini — la nouvelle PWA reste connectée 1 an, même au force-quit."
    )


# ── Banner post-login : instructions PWA iOS ─────────────────────────────────
if st.session_state.pop("just_logged_in", False):
    st.success("✅ Connecté pour 1 an !")
    with st.container(border=True):
        _render_pwa_install_block()

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

# Valeurs de repli — le header ne doit jamais faire planter toute la page
_silver = {"qty": 0.0, "avg_price": 0.0, "snapshot_value": 0.0}
_silver_price = 0.0
_xag_qty = _xag_avg = 0.0
_xag_val = None
_bp = {"holdings": {}, "total_eur": 0.0}
_bp_total = 0.0
_tr_etf = _tr_cash = 0.0

try:
    # Silver WisdomTree : parts depuis le relevé Trade Republic × prix live WisdomTree
    _silver = xag_tab.get_silver_holding()
    _silver_price, _ = xag_tab.get_silver_live()
    if _silver_price is None:
        _sh = xag_tab.get_silver_history()
        _silver_price = float(_sh.iloc[-1]) if not _sh.empty else 0.0
    _xag_qty = _silver["qty"]
    _xag_avg = _silver["avg_price"]
    _xag_val = _xag_qty * _silver_price if _silver_price else None

    _bp       = btc_tab.get_bitpanda_values()
    _bp_total = _bp["total_eur"]

    _tr = etf_pea_tab.get_tr_live_value()
    # ETF/PEA hors silver (le silver est valorisé en live ci-dessus → pas de double comptage)
    _tr_etf  = max(_tr.get("savings_eur", 0.0) - _silver.get("snapshot_value", 0.0), 0.0)
    _tr_cash = _tr.get("cash_eur", 0.0)
except Exception as _e:
    st.caption(f"⚠️ Résumé du portefeuille temporairement indisponible ({type(_e).__name__}).")

_total    = (_xag_val or 0) + _bp_total + _tr_etf + _tr_cash

_sep = '<span style="color:#334155">│</span>'
_sections: list[str] = []

# Section Silver WisdomTree (uniquement si on en détient)
if _xag_qty and _xag_qty > 0:
    _sections.append(
        f'<span>🥈&nbsp;<b style="color:#e2e8f0">{_xag_qty:g} parts</b>'
        f'<span style="color:#64748b">&nbsp;·&nbsp;{_fmt(_silver_price)}/part</span>'
        f'&nbsp;→&nbsp;<b style="color:#22c55e">{_fmt(_xag_val)}</b>{_pnl_html(_silver_price, _xag_avg)}</span>'
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

# Trade Republic (PEA hors silver et/ou Espèces, uniquement si > 0)
_tr_parts = []
if _tr_etf > 0:
    _tr_parts.append(
        f'📈&nbsp;<span style="color:#94a3b8">PEA</span>&nbsp;'
        f'<b style="color:#22c55e">{_fmt(_tr_etf)}</b>'
    )
if _tr_cash > 0:
    _tr_parts.append(
        f'💵&nbsp;<b style="color:#22c55e">{_fmt(_tr_cash)}</b>'
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
  <span style="color:#475569;font-size:0.65rem;margin-left:auto">v3.0</span>
</div>
""", unsafe_allow_html=True)

# ── App principale ────────────────────────────────────────────────────────────

tab0, tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🥈 Silver", "₿ Cryptos", "📈 ETF & Actions PEA"])

with tab0:
    dashboard_tab.render()

with tab1:
    xag_tab.render()

with tab2:
    btc_tab.render()

with tab3:
    etf_pea_tab.render()

# ── Pied de page : déconnexion + procédure PWA ──────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ Compte & installation PWA", expanded=False):
    st.markdown("**Session valide 1 an** depuis le dernier login.")
    st.divider()
    _render_pwa_install_block()
    st.divider()
    if st.button("🚪 Se déconnecter", key="logout_btn"):
        st.query_params.clear()
        st.session_state["authenticated"] = False
        st.rerun()
