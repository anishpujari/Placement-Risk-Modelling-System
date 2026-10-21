"""
PlacementAI - Streamlit Dashboard
Entry point: page config, shared theme and navigation for the lender-facing UI.
"""

import os, sys

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui import apply_theme

st.set_page_config(
    page_title="PlacementAI",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")

nav = st.navigation({
    "Portfolio": [
        st.Page(os.path.join(PAGES_DIR, "1_Portfolio_Overview.py"), title="Portfolio overview",
                icon=":material/monitoring:", url_path="Portfolio_Overview", default=True),
        st.Page(os.path.join(PAGES_DIR, "4_Institute_Benchmarking.py"), title="Institute benchmarking",
                icon=":material/account_balance:", url_path="Institute_Benchmarking"),
    ],
    "Students": [
        st.Page(os.path.join(PAGES_DIR, "2_Student_Risk_Card.py"), title="Student lookup",
                icon=":material/person_search:", url_path="Student_Risk_Card"),
        st.Page(os.path.join(PAGES_DIR, "3_Predict_New.py"), title="New assessment",
                icon=":material/assignment_add:", url_path="Predict_New"),
    ],
    "Model": [
        st.Page(os.path.join(PAGES_DIR, "5_Model_Insights.py"), title="Model insights",
                icon=":material/insights:", url_path="Model_Insights"),
    ],
})

st.sidebar.caption(
    "Risk intelligence only. Scores support a loan officer's review and are not "
    "automated credit decisions, in line with the RBI Fair Practices Code."
)

nav.run()
