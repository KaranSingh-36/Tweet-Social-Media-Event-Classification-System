"""EventPulse – Intelligent Tweet / Social Media Event Classification System.

Entry point of the Streamlit dashboard:  ``streamlit run app.py``
"""
from __future__ import annotations

import streamlit as st

from config import settings
from src import ui

st.set_page_config(
    page_title="EventPulse · Tweet Event Intelligence",
    page_icon=str(settings.ASSETS_DIR / "logo_icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
st.logo(str(settings.ASSETS_DIR / "logo.svg"), size="large", icon_image=str(settings.ASSETS_DIR / "logo_icon.svg"))

PAGES = {
    "Workspace": [
        st.Page("pages/0_Overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("pages/1_Live_Classifier.py", url_path="live-classifier", title="Live Classifier", icon=":material/bolt:"),
        st.Page("pages/2_Bulk_Analysis.py", url_path="bulk-analysis", title="Bulk Analysis", icon=":material/upload_file:"),
    ],
    "Insights": [
        st.Page("pages/3_Event_Analytics.py", url_path="event-analytics", title="Event Analytics", icon=":material/insights:"),
    ],
    "Models": [
        st.Page("pages/4_Model_Comparison.py", url_path="model-comparison", title="Model Comparison", icon=":material/leaderboard:"),
        st.Page("pages/5_Model_Evaluation.py", url_path="model-evaluation", title="Model Evaluation", icon=":material/fact_check:"),
        st.Page("pages/6_Error_Analysis.py", url_path="error-analysis", title="Error Analysis", icon=":material/troubleshoot:"),
    ],
}
nav = st.navigation(PAGES, position="sidebar", expanded=True)


def sidebar_status() -> None:
    models = {
        "TF-IDF + SVM": settings.SVM_FILE.exists(),
        "TF-IDF + LogReg": settings.LOGREG_FILE.exists(),
        "DistilBERT": (settings.TRANSFORMER_FINAL_DIR / "config.json").exists(),
        "Disaster type model": settings.EVENT_TYPE_MODEL_FILE.exists(),
    }
    rows = "".join(
        f'<div class="status-row"><span><span class="status-dot {"on" if ok else "off"}"></span>{name}</span>'
        f'<span class="status-text">{"Ready" if ok else "Not trained"}</span></div>'
        for name, ok in models.items()
    )
    with st.sidebar:
        st.html(f'<div class="sidebar-card"><div class="title">Model status</div>{rows}</div>')
        st.html(
            '<div class="sidebar-card"><div class="title">About</div>'
            'MCA AI &amp; ML · Semester III capstone<br>ETMMTI303 – Language Technologies &amp; Text Intelligence</div>'
        )


sidebar_status()
nav.run()
