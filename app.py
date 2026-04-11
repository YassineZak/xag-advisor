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

    /* Onglets — texte toujours visible */
    [data-baseweb="tab"] { color: #cbd5e1 !important; font-size: 1rem !important; font-weight: 600 !important; }
    [data-baseweb="tab"][aria-selected="true"] { color: #f1f5f9 !important; border-bottom: 3px solid #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🥈 Métaux", "₿ Cryptos"])

with tab1:
    xag_tab.render()

with tab2:
    btc_tab.render()
