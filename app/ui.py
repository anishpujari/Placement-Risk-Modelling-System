"""
PlacementAI - Shared UI building blocks for the Streamlit dashboard.
Theme CSS, Plotly template, number formatting and reusable page sections.
"""

import os
import re
from html import escape

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "placement_100k_final.csv")

# ── Palette ──────────────────────────────────────────────────────────────────
# One accent (ink blue). Red / amber / green are reserved for risk state only.
ACCENT = "#1F4B7A"
TEXT = "#1C2430"
MUTED = "#5B6573"
BORDER = "#DDE1E6"
GRID = "#ECEFF2"

RISK_ORDER = ["HIGH", "MEDIUM", "LOW"]
RISK_LABEL = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
RISK_COLOR = {"HIGH": "#B3261E", "MEDIUM": "#B7791F", "LOW": "#2F7D57"}

FONT = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"

CSS = f"""
<style>
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1320px; }}

h1 {{ font-size: 1.65rem !important; font-weight: 600 !important; letter-spacing: -0.01em; padding-bottom: 0 !important; }}
h3 {{ font-size: 1.02rem !important; font-weight: 600 !important; padding-bottom: 0.15rem !important; }}

[data-testid="stSidebarNav"]::before {{
    content: "PlacementAI";
    display: block;
    font-weight: 600;
    font-size: 1.05rem;
    color: {TEXT};
    padding: 0.25rem 1rem 1rem;
}}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}

.page-sub {{ color: {MUTED}; font-size: 0.92rem; margin: -0.4rem 0 1.4rem; }}

.kpi-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    margin: 0.25rem 0 1.75rem;
}}
.kpi {{ padding: 0.9rem 1.25rem 0.95rem 0; }}
.kpi + .kpi {{ border-left: 1px solid {BORDER}; padding-left: 1.25rem; }}
.kpi-label {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: 0.3rem; }}
.kpi-value {{
    font-variant-numeric: tabular-nums;
    font-size: 1.6rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: {TEXT};
    line-height: 1.15;
}}
.kpi-value small {{ font-size: 0.85rem; color: {MUTED}; margin-left: 0.2rem; }}
.kpi-note {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.25rem; }}
@media (max-width: 760px) {{
    .kpi + .kpi {{ border-left: 0; padding-left: 0; }}
    .kpi {{ border-bottom: 1px solid {GRID}; }}
}}

.risk {{
    display: inline-block;
    font-weight: 600;
    border-left: 3px solid currentColor;
    padding-left: 0.5rem;
    line-height: 1.2;
}}
.risk-HIGH {{ color: {RISK_COLOR["HIGH"]}; }}
.risk-MEDIUM {{ color: {RISK_COLOR["MEDIUM"]}; }}
.risk-LOW {{ color: {RISK_COLOR["LOW"]}; }}

.record-head {{
    display: grid;
    grid-template-columns: minmax(0, 2.4fr) minmax(0, 1fr) minmax(0, 1fr);
    gap: 1.5rem;
    align-items: end;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.25rem;
}}
.record-id {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: {MUTED}; }}
.record-title {{ font-size: 1.35rem; font-weight: 600; color: {TEXT}; margin: 0.15rem 0; }}
.record-meta {{ color: {MUTED}; font-size: 0.9rem; }}
.record-head .risk {{ font-size: 1.35rem; }}
@media (max-width: 760px) {{ .record-head {{ grid-template-columns: 1fr 1fr; }} .record-head > :first-child {{ grid-column: 1 / -1; }} }}

.rows {{ margin: 0.2rem 0 0.5rem; }}
.row {{
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid {GRID};
    font-size: 0.92rem;
}}
.row:last-child {{ border-bottom: 0; }}
.row-label {{ color: {TEXT}; }}
.row-value {{ font-variant-numeric: tabular-nums; font-weight: 500; white-space: nowrap; color: {TEXT}; }}
.row-value.up {{ color: {RISK_COLOR["HIGH"]}; }}
.row-value.down {{ color: {RISK_COLOR["LOW"]}; }}
.row-value.muted {{ color: {MUTED}; }}

.note {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.5; max-width: 72ch; }}
</style>
"""


def apply_theme():
    """Inject the shared stylesheet and register the Plotly template."""
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["placementai"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12.5, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[ACCENT, "#7A8796", "#A9B4C0"],
        xaxis=dict(automargin=True, showgrid=False, linecolor=BORDER, ticks="", zeroline=False,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12))),
        yaxis=dict(automargin=True, gridcolor=GRID, zeroline=False, ticks="", showline=False,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12))),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", title=dict(text=""),
                    font=dict(color=MUTED)),
        margin=dict(l=4, r=4, t=36, b=4),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
        bargap=0.35,
    ))
    pio.templates.default = "placementai"


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading portfolio")
def load_data():
    return pd.read_csv(DATA_PATH)


# ── Formatting ───────────────────────────────────────────────────────────────

def inr(n, decimals=0):
    """Format a number with Indian digit grouping (12,34,567)."""
    neg = n < 0
    n = abs(n)
    whole = int(n)
    frac = f"{n - whole:.{decimals}f}"[1:] if decimals else ""
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        s = f"{head},{tail}"
    return ("-" if neg else "") + s + frac


def lakh(n):
    return f"₹{n / 1e5:.1f} L"


def crore(n):
    return f"₹{inr(n / 1e7)} Cr"


# ── Page blocks ──────────────────────────────────────────────────────────────

def page_header(title, sub=None):
    st.title(title, anchor=False)
    if sub:
        st.markdown(f'<p class="page-sub">{escape(sub)}</p>', unsafe_allow_html=True)


def section(title, caption=None):
    st.subheader(title, anchor=False)
    if caption:
        st.caption(caption)


def kpi_strip(items):
    """items: list of (label, value, note) tuples. value may contain trusted <small> markup."""
    cells = []
    for label, value, note in items:
        note_html = f'<div class="kpi-note">{escape(note)}</div>' if note else ""
        cells.append(
            f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{value}</div>{note_html}</div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


def risk_tag(tier, suffix=" risk"):
    tier = str(tier).upper()
    return f'<span class="risk risk-{tier}">{RISK_LABEL.get(tier, tier.title())}{suffix}</span>'


def record_header(record_id, title, meta, tier, score_label, score_value, tier_label="Risk tier"):
    st.markdown(
        f"""<div class="record-head">
            <div>
                <div class="record-id">{escape(str(record_id))}</div>
                <div class="record-title">{escape(str(title))}</div>
                <div class="record-meta">{escape(meta)}</div>
            </div>
            <div><div class="kpi-label">{escape(tier_label)}</div>{risk_tag(tier, "")}</div>
            <div><div class="kpi-label">{escape(score_label)}</div><div class="kpi-value">{score_value}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def value_rows(rows):
    """rows: list of (label, value, tone) with tone in {'', 'up', 'down', 'muted'}."""
    html = "".join(
        f'<div class="row"><span class="row-label">{escape(str(label))}</span>'
        f'<span class="row-value {tone}">{escape(str(value))}</span></div>'
        for label, value, tone in rows
    )
    st.markdown(f'<div class="rows">{html}</div>', unsafe_allow_html=True)


def driver_rows(drivers):
    """drivers: list of (label, impact_text). '+' impacts raise risk, '-' lower it."""
    rows = []
    for label, impact in drivers:
        impact = str(impact).strip()
        tone = "up" if impact.startswith("+") else "down" if impact.startswith("-") else "muted"
        rows.append((label, impact, tone))
    value_rows(rows)


def note(text):
    st.markdown(f'<p class="note">{escape(text)}</p>', unsafe_allow_html=True)


def chart(fig, height=320):
    # Set explicitly: Streamlit's chart theme otherwise overrides template values.
    fig.update_layout(
        height=height,
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color=TEXT),
        margin=dict(l=16, r=8, t=36, b=8),
    )
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, width="stretch")


# ── Model and SHAP ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model")
def load_models():
    from model.predict import load_model
    from model.shap_explainer import get_explainer
    model, encoders, feature_cols = load_model()
    return model, encoders, feature_cols, get_explainer(model)


def explain(student_dict):
    """Model prediction plus full SHAP breakdown for one student dict."""
    from model.predict import predict_student, prepare_features
    from model.shap_explainer import explain_full
    model, encoders, feature_cols, explainer = load_models()
    result = predict_student(student_dict, model, encoders, feature_cols)
    X = prepare_features(student_dict, encoders, feature_cols)
    contrib, p_high, _ = explain_full(X, explainer, feature_cols, student_dict)
    return result, contrib, X


def model_rows(result):
    """Predicted tier and per-tier probabilities as value rows."""
    probs = result["class_probabilities"]
    tier = result["risk_tier"]
    value_rows([
        ("Predicted tier", RISK_LABEL.get(tier, tier), {"HIGH": "up", "LOW": "down"}.get(tier, "")),
        ("Probability of high risk", f"{probs.get('HIGH', 0) * 100:.1f}%", ""),
        ("Probability of medium risk", f"{probs.get('MEDIUM', 0) * 100:.1f}%", ""),
        ("Probability of low risk", f"{probs.get('LOW', 0) * 100:.1f}%", ""),
    ])


def _value_text(v):
    from model.shap_explainer import _format_value
    return _format_value(v)


def shap_chart(contrib, top=12):
    """Horizontal SHAP bars for the HIGH class: right pushes toward high risk, left away from it."""
    top_df = contrib.head(top).iloc[::-1]
    rest = contrib.iloc[top:]
    labels = [f"{n} = {_value_text(v)}" for n, v in zip(top_df["name"], top_df["value"])]
    values = top_df["shap_high"].tolist()
    pps = top_df["effect_pp"].tolist()
    if len(rest):
        labels.insert(0, f"{len(rest)} other features")
        values.insert(0, float(rest["shap_high"].sum()))
        pps.insert(0, float("nan"))
    colors = [RISK_COLOR["HIGH"] if v > 0 else RISK_COLOR["LOW"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        customdata=pps,
        hovertemplate="%{y}<br>SHAP %{x:+.3f}<br>%{customdata:+.1f} pts high-risk probability<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=BORDER)
    fig.update_layout(showlegend=False, bargap=0.3)
    fig.update_xaxes(title=dict(text="SHAP value for high risk (log-odds)"), showgrid=True, gridcolor=GRID,
                     zeroline=False)
    fig.update_yaxes(showgrid=False)
    chart(fig, height=max(300, 30 * len(labels) + 70))


def shap_table(contrib):
    """Every feature's contribution, for an expander."""
    table = contrib[["name", "value", "shap_high", "effect_pp"]].copy()
    table["value"] = table["value"].map(_value_text)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "name": st.column_config.TextColumn("Feature"),
            "value": st.column_config.TextColumn("Value"),
            "shap_high": st.column_config.NumberColumn("SHAP (log-odds)", format="%+.3f"),
            "effect_pp": st.column_config.NumberColumn("Effect on high-risk probability", format="%+.1f pts"),
        },
    )


def shap_section(contrib, title="What drives the high-risk probability"):
    section(
        title,
        "Each bar is one feature's SHAP contribution toward the high-risk class. Red bars push toward high "
        "risk, green bars push away. Hover a bar to see its effect in probability points.",
    )
    shap_chart(contrib)
    with st.expander(f"All {len(contrib)} feature contributions"):
        shap_table(contrib)
