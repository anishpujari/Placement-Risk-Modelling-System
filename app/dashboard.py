"""
PlacementAI - Streamlit Dashboard
Main entry point for the lender-facing UI.
"""

import streamlit as st
import os, sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="PlacementAI - Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .risk-badge-LOW {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .risk-badge-MEDIUM {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .risk-badge-HIGH {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Main landing
st.markdown("""
<div class="main-header">
    <h1>🎓 PlacementAI</h1>
    <p>AI-Powered Placement Risk Intelligence for Education Lenders</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Welcome to PlacementAI

Navigate using the sidebar to access:

- **📊 Portfolio Overview** — Real-time portfolio risk monitoring with KPIs and heatmaps
- **🎯 Student Risk Card** — Individual student risk assessment with SHAP explanations
- **🔮 Predict New Student** — Run live predictions on new student applications
- **🏫 Institute Benchmarking** — Compare university placement performance

---

*PlacementAI produces risk intelligence only — not automated credit decisions. Compliant with RBI Fair Practices Code.*
""")
