"""
Export du récap portefeuille vers un gist secret GitHub, lu par le widget
Scriptable sur iPhone (écran d'accueil / verrouillage).

Pourquoi un gist secret et pas un « vrai » gist privé ?
GitHub n'a pas de gist réellement privé : un gist « secret » est simplement
non-listé, mais accessible par quiconque dispose de l'URL (un hash de 32
caractères impossible à deviner). Avantage : Scriptable lit juste l'URL raw,
on ne met AUCUN token sur le téléphone, et les montants ne sont pas exposés
publiquement.

Le gist est créé une seule fois (repéré par sa description) puis mis à jour.
Nécessite que st.secrets["GITHUB_TOKEN"] possède le scope `gist`.
"""

import json
from datetime import datetime

import streamlit as st

GIST_DESC     = "xag-advisor-widget"   # marqueur pour retrouver le gist
GIST_FILENAME = "widget.json"


def _user():
    from github import Github
    return Github(st.secrets["GITHUB_TOKEN"]).get_user()


def _find_widget_gist(user):
    """Retrouve le gist du widget par sa description ; None s'il n'existe pas."""
    for gist in user.get_gists():
        if gist.description == GIST_DESC:
            return gist
    return None


def build_payload(snap: dict, signal: dict,
                  day_var: str | None = None, day_pct: str | None = None) -> dict:
    """Construit le JSON minimal consommé par le widget Scriptable."""
    rsi = signal.get("rsi")
    return {
        "total":        round(float(snap.get("total", 0) or 0), 2),
        "silver":       round(float(snap.get("silver_eur", 0) or 0), 2),
        "etf":          round(float(snap.get("etf_eur", 0) or 0), 2),
        "cash":         round(float(snap.get("cash_eur", 0) or 0), 2),
        "crypto":       round(float(snap.get("crypto_eur", 0) or 0), 2),
        "signal_label": signal.get("label", "—"),
        "signal_color": signal.get("color", "#94a3b8"),
        "rsi":          round(float(rsi), 1) if rsi is not None else None,
        "day_var":      day_var,
        "day_pct":      day_pct,
        "updated_at":   datetime.now().strftime("%d/%m %H:%M"),
        "currency":     "EUR",
    }


def push_widget(snap: dict, signal: dict,
                day_var: str | None = None, day_pct: str | None = None) -> dict:
    """
    Crée ou met à jour le gist secret avec le récap courant.
    Retourne {ok, url, gist_id, error}.
    """
    payload = build_payload(snap, signal, day_var, day_pct)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        from github import InputFileContent
        user = _user()
        gist = _find_widget_gist(user)
        if gist is None:
            gist = user.create_gist(
                public=False,
                files={GIST_FILENAME: InputFileContent(content)},
                description=GIST_DESC,
            )
        else:
            gist.edit(files={GIST_FILENAME: InputFileContent(content)})
        # URL raw « dernière révision » (sans sha → toujours la version courante)
        raw_url = (
            f"https://gist.githubusercontent.com/"
            f"{user.login}/{gist.id}/raw/{GIST_FILENAME}"
        )
        return {"ok": True, "url": raw_url, "gist_id": gist.id, "error": None}
    except Exception as e:
        return {"ok": False, "url": None, "gist_id": None,
                "error": f"{type(e).__name__}: {e}"}
