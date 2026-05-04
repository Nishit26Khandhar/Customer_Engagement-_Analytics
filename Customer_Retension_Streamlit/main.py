# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         RetentIQ — Customer Retention Intelligence Dashboard                ║
# ║         Unified Mentors Internship | ML Project                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import sys
import warnings
import io
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetentIQ | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "navy":    "#1B3B6F",
    "teal":    "#0891B2",
    "coral":   "#E05252",
    "amber":   "#D97706",
    "green":   "#059669",
    "bg":      "#F0F4F8",
    "card":    "#FFFFFF",
    "text":    "#0F172A",
    "muted":   "#64748B",
    "border":  "#E2E8F0",
}
CHURN_COLORS   = {"Retained": C["teal"], "Churned": C["coral"]}
RISK_COLORS    = {"Low Risk": C["green"], "Medium Risk": C["amber"], "High Risk": C["coral"]}
GEO_COLORS     = ["#1B3B6F", "#0891B2", "#0E7490"]

# ── Injected CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}
.main .block-container {{
    background: {C['bg']};
    padding-top: 1.5rem;
    max-width: 1280px;
}}
#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, {C['navy']} 0%, #0D2A52 60%, #0A1F3E 100%);
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15);
}}
[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
[data-testid="stSidebar"] .stRadio > label {{
    color: #94A3B8 !important;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}}
[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {{
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #E2E8F0 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* ── KPI Cards ── */
.kpi-wrap {{
    background: {C['card']};
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 2px 16px rgba(27,59,111,0.07);
    border-top: 3px solid;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    height: 100%;
}}
.kpi-wrap:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(27,59,111,0.14);
}}
.kpi-icon {{ font-size: 1.7rem; margin-bottom: 10px; line-height:1; }}
.kpi-val {{
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {C['text']};
    line-height: 1.1;
}}
.kpi-lbl {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['muted']};
    margin-top: 4px;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {C['muted']};
    margin-top: 6px;
}}

/* ── Section Titles ── */
.sec-title {{
    font-family: 'Lora', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: {C['navy']};
    margin-bottom: 2px;
}}
.sec-sub {{
    font-size: 0.875rem;
    color: {C['muted']};
    margin-bottom: 20px;
}}

/* ── Chart Card ── */
.chart-card {{
    background: {C['card']};
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 2px 16px rgba(27,59,111,0.06);
    margin-bottom: 16px;
}}
.chart-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {C['navy']};
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid {C['border']};
}}

/* ── Insight Box ── */
.insight-box {{
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDFA 100%);
    border-left: 3px solid {C['teal']};
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.875rem;
    color: {C['navy']};
    line-height: 1.55;
}}
.insight-box strong {{ color: {C['teal']}; }}

/* ── Risk Score Display ── */
.risk-hero {{
    background: linear-gradient(135deg, {C['navy']} 0%, #0D2A52 100%);
    color: white;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(27,59,111,0.3);
}}
.risk-pct {{
    font-family: 'Lora', serif;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
}}
.risk-lbl {{
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-top: 6px;
}}

/* ── Badges ── */
.badge {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.badge-high  {{ background:#FEE2E2; color:#991B1B; }}
.badge-med   {{ background:#FEF3C7; color:#92400E; }}
.badge-low   {{ background:#D1FAE5; color:#065F46; }}

/* ── Page Hero Banner ── */
.page-hero {{
    background: linear-gradient(135deg, {C['navy']} 0%, #1D4ED8 50%, {C['teal']} 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}}
.page-hero::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}}
.page-hero::after {{
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}}
.hero-title {{
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 6px;
    position: relative;
    z-index:1;
}}
.hero-sub {{
    font-size: 0.92rem;
    opacity: 0.82;
    max-width: 520px;
    line-height: 1.55;
    position: relative;
    z-index:1;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {C['navy']}, #1D4ED8);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    padding: 10px 28px;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(27,59,111,0.28);
    transition: all 0.18s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 7px 20px rgba(27,59,111,0.38);
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    gap: 4px;
}}

/* ── Dataframe ── */
.dataframe {{ font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 0.82rem !important; }}

/* ── Divider ── */
.grad-divider {{
    height: 2px;
    background: linear-gradient(90deg, {C['navy']}, {C['teal']}, transparent);
    border-radius: 2px;
    margin: 20px 0 24px 0;
}}

/* ── Logo in sidebar ── */
.sidebar-logo {{
    text-align: center;
    padding: 20px 16px 24px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}}
.logo-main {{
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: white !important;
    letter-spacing: -0.01em;
}}
.logo-tag {{
    font-size: 0.65rem;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}}

/* ── Model badge ── */
.model-badge {{
    background: rgba(8,145,178,0.15);
    border: 1px solid rgba(8,145,178,0.35);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.78rem;
}}
.model-badge span {{ color: {C['teal']} !important; font-weight:700; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)

    paths = [
        os.path.join(BASE_DIR,   "artifacts", "raw.csv"),
        os.path.join(BASE_DIR,   "notebook",  "data", "European_Bank.csv"),
        os.path.join(BASE_DIR,   "data",      "European_Bank.csv"),
        os.path.join(PARENT_DIR, "artifacts", "raw.csv"),
        os.path.join(PARENT_DIR, "notebook",  "data", "European_Bank.csv"),
        os.path.join(PARENT_DIR, "data",      "European_Bank.csv"),
    ]

    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Exited" in df.columns:
                df["Churn_Label"] = df["Exited"].map({0: "Retained", 1: "Churned"})
                return df

    return None


@st.cache_resource(show_spinner=False)
def load_model():
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    for base in [BASE_DIR, PARENT_DIR]:
        p = os.path.join(base, "artifacts", "model.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    for base in [BASE_DIR, PARENT_DIR]:
        p = os.path.join(base, "artifacts", "preprocessor.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def fmt_num(n, decimals=0):
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def kpi_card(icon, value, label, sub="", color=C["navy"]):
    return f"""
    <div class="kpi-wrap" style="border-top-color:{color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-lbl">{label}</div>
        {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>"""


def plotly_defaults(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=C["text"]),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        title_font=dict(family="Plus Jakarta Sans", size=12, color=C["navy"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"], tickfont_color=C["muted"])
    fig.update_yaxes(gridcolor=C["border"], linecolor="rgba(0,0,0,0)", tickfont_color=C["muted"])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar(df, model_obj):
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-main">🏦 RetentIQ</div>
            <div class="logo-tag">Customer Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "NAVIGATION",
            options=[
                "📊  Overview",
                "💬  Engagement Analytics",
                "📦  Product Utilization",
                "💰  Financial Risk",
                "🔍  Disengaged Detector",
                "🎯  Risk Scorer",
            ],
        )

        st.markdown("<div style='margin-top:20px;margin-bottom:8px'></div>", unsafe_allow_html=True)

        if df is not None:
            total = len(df)
            churn_rate = df["Exited"].mean() * 100
            st.markdown(f"""
            <div class="model-badge">
                📁 Dataset: <span>{fmt_num(total)} rows</span><br>
                📉 Churn Rate: <span>{churn_rate:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        if model_obj is not None:
            mname = model_obj.get("best_model_name", "CatBoost")
            thresh = model_obj.get("threshold", 0.20)
            st.markdown(f"""
            <div class="model-badge" style="margin-top:8px">
                🤖 Model: <span>{mname}</span><br>
                🎚️ Threshold: <span>{thresh:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="position:absolute;bottom:20px;left:0;right:0;text-align:center;
                    font-size:0.65rem;color:rgba(255,255,255,0.25);padding:0 16px;">
            Unified Mentors Internship Project<br>Customer Engagement Analytics
        </div>
        """, unsafe_allow_html=True)

    return page.strip()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

def page_overview(df):
    st.markdown("""
    <div class="page-hero">
        <div class="hero-title">Customer Retention Intelligence</div>
        <div class="hero-sub">
            An end-to-end behavioral analytics platform that moves beyond demographics
            to reveal the engagement and product drivers behind customer churn.
        </div>
    </div>
    """, unsafe_allow_html=True)

    total        = len(df)
    churned      = df["Exited"].sum()
    churn_rate   = churned / total * 100
    retained     = total - churned
    high_risk    = df["RetentionRiskScore"].quantile(0.75) if "RetentionRiskScore" in df.columns else 0
    high_risk_n  = (df["RetentionRiskScore"] >= high_risk).sum() if "RetentionRiskScore" in df.columns else 0
    avg_rsi      = df["RelationshipStrengthIndex"].mean() if "RelationshipStrengthIndex" in df.columns else 0
    at_risk_prem = df["AtRiskPremiumCustomer"].sum() if "AtRiskPremiumCustomer" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("👥", fmt_num(total), "Total Customers", f"{fmt_num(retained)} retained", C["navy"]), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("📉", f"{churn_rate:.1f}%", "Churn Rate", f"{fmt_num(churned)} customers lost", C["coral"]), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("⚠️", fmt_num(high_risk_n), "High-Risk Customers", "Top 25% by retention risk score", C["amber"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("🤝", f"{avg_rsi:.3f}", "Avg. Relationship Index", f"{fmt_num(at_risk_prem)} at-risk premium", C["teal"]), unsafe_allow_html=True)

    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1.6])

    with col_a:
        st.markdown("<div class='chart-card'><div class='chart-title'>Churn Distribution</div>", unsafe_allow_html=True)
        donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[retained, churned],
            hole=0.62,
            marker_colors=[C["teal"], C["coral"]],
            textinfo="percent",
            textfont_size=13,
            pull=[0, 0.04],
        ))
        donut.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px'>Churn</span>",
                             x=0.5, y=0.5, showarrow=False, font_size=18, font_color=C["coral"])
        plotly_defaults(donut, 280)
        donut.update_layout(margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                            legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
        st.plotly_chart(donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if "Geography" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Churn Rate by Geography</div>", unsafe_allow_html=True)
            geo_df = df.groupby("Geography").agg(
                Total=("Exited", "count"),
                Churned=("Exited", "sum")
            ).reset_index()
            geo_df["Churn_Rate"] = geo_df["Churned"] / geo_df["Total"] * 100
            geo_df["Retained"]   = geo_df["Total"] - geo_df["Churned"]

            fig_geo = go.Figure()
            fig_geo.add_trace(go.Bar(name="Retained", x=geo_df["Geography"], y=geo_df["Retained"],
                                     marker_color=C["teal"], marker_line_width=0))
            fig_geo.add_trace(go.Bar(name="Churned", x=geo_df["Geography"], y=geo_df["Churned"],
                                     marker_color=C["coral"], marker_line_width=0))
            for _, row in geo_df.iterrows():
                fig_geo.add_annotation(x=row["Geography"], y=row["Total"]+30,
                                       text=f"{row['Churn_Rate']:.1f}%",
                                       showarrow=False, font=dict(size=11, color=C["coral"], family="Plus Jakarta Sans"))
            fig_geo.update_layout(barmode="stack")
            plotly_defaults(fig_geo, 280)
            st.plotly_chart(fig_geo, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Engagement segment + Risk tier row
    col_c, col_d = st.columns(2)

    with col_c:
        if "EngagementSegment" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Churn by Engagement Segment</div>", unsafe_allow_html=True)
            seg_df = df.groupby("EngagementSegment")["Exited"].agg(["mean","count"]).reset_index()
            seg_df.columns = ["Segment","Churn_Rate","Count"]
            seg_df["Churn_Rate"] *= 100
            seg_df = seg_df.sort_values("Churn_Rate", ascending=True)
            colors = [C["coral"] if r > 30 else C["amber"] if r > 15 else C["teal"] for r in seg_df["Churn_Rate"]]
            fig_seg = go.Figure(go.Bar(
                y=seg_df["Segment"], x=seg_df["Churn_Rate"],
                orientation="h", marker_color=colors,
                text=[f"{r:.1f}%" for r in seg_df["Churn_Rate"]],
                textposition="outside", marker_line_width=0,
            ))
            plotly_defaults(fig_seg, 260)
            fig_seg.update_xaxes(title="Churn Rate (%)")
            st.plotly_chart(fig_seg, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        if "RiskTier" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Risk Tier Distribution</div>", unsafe_allow_html=True)
            risk_df = df["RiskTier"].value_counts().reset_index()
            risk_df.columns = ["Tier","Count"]
            order = ["Low Risk","Medium Risk","High Risk"]
            risk_df["Tier"] = pd.Categorical(risk_df["Tier"], categories=order, ordered=True)
            risk_df = risk_df.sort_values("Tier")
            rcolors = [RISK_COLORS.get(t, C["navy"]) for t in risk_df["Tier"]]
            fig_risk = go.Figure(go.Bar(
                x=risk_df["Tier"], y=risk_df["Count"],
                marker_color=rcolors, marker_line_width=0,
                text=risk_df["Count"], textposition="outside",
            ))
            plotly_defaults(fig_risk, 260)
            st.plotly_chart(fig_risk, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Key insights
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>📌 Key Findings</div>", unsafe_allow_html=True)
    ia, ib, ic = st.columns(3)
    with ia:
        st.markdown(f"""<div class="insight-box">
        <strong>🇩🇪 Germany Effect</strong><br>
        Germany shows the highest churn rate at ~32%, nearly double that of France (~16%). 
        High-balance German customers are disproportionately at risk of silent churn.
        </div>""", unsafe_allow_html=True)
    with ib:
        st.markdown(f"""<div class="insight-box">
        <strong>🚪 Inactivity is the #1 Signal</strong><br>
        Inactive members churn at 3× the rate of active members. Engagement status 
        is the single strongest behavioural predictor in this dataset.
        </div>""", unsafe_allow_html=True)
    with ic:
        st.markdown(f"""<div class="insight-box">
        <strong>📦 Product Paradox</strong><br>
        Customers with 3–4 products churn at 80%+ — a counterintuitive finding 
        that likely reflects forced bundling without genuine engagement.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENGAGEMENT ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

def page_engagement(df):
    st.markdown("<div class='sec-title'>💬 Engagement Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>How customer activity and engagement behaviour drives retention outcomes</div>", unsafe_allow_html=True)

    # Active vs Inactive
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='chart-card'><div class='chart-title'>Active vs Inactive Member Churn</div>", unsafe_allow_html=True)
        act_df = df.groupby("IsActiveMember")["Exited"].agg(["mean","count"]).reset_index()
        act_df["Label"] = act_df["IsActiveMember"].map({0:"Inactive",1:"Active"})
        act_df["Churn_Rate"] = act_df["mean"] * 100
        act_df["Retained_Rate"] = 100 - act_df["Churn_Rate"]
        fig_act = go.Figure()
        fig_act.add_trace(go.Bar(name="Churn Rate", x=act_df["Label"], y=act_df["Churn_Rate"],
                                  marker_color=[C["coral"], C["coral"]], marker_line_width=0,
                                  text=[f"{v:.1f}%" for v in act_df["Churn_Rate"]], textposition="outside"))
        for _, row in act_df.iterrows():
            fig_act.add_annotation(x=row["Label"], y=-8,
                                   text=f"n={fmt_num(row['count'])}",
                                   showarrow=False, font=dict(size=10, color=C["muted"]))
        plotly_defaults(fig_act, 300)
        fig_act.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig_act, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "ActivityScore" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Activity Score Distribution by Churn Status</div>", unsafe_allow_html=True)
            fig_box = px.box(df, x="Churn_Label", y="ActivityScore",
                              color="Churn_Label",
                              color_discrete_map=CHURN_COLORS,
                              points=False)
            plotly_defaults(fig_box, 300)
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Engagement Segment deep dive
    if "EngagementSegment" in df.columns:
        st.markdown("<div class='chart-card'><div class='chart-title'>Segment Performance Matrix — Churn Rate vs Customer Count</div>", unsafe_allow_html=True)
        seg_df = df.groupby("EngagementSegment").agg(
            Churn_Rate=("Exited","mean"),
            Count=("Exited","count"),
            Avg_Balance=("Balance","mean"),
        ).reset_index()
        seg_df["Churn_Rate"] *= 100
        fig_bubble = px.scatter(
            seg_df, x="Churn_Rate", y="Count",
            size="Avg_Balance", color="Churn_Rate",
            text="EngagementSegment",
            color_continuous_scale=[[0, C["teal"]], [0.5, C["amber"]], [1, C["coral"]]],
            size_max=60,
        )
        fig_bubble.update_traces(textposition="top center", textfont_size=11)
        fig_bubble.update_coloraxes(showscale=False)
        plotly_defaults(fig_bubble, 360)
        fig_bubble.update_xaxes(title="Churn Rate (%)")
        fig_bubble.update_yaxes(title="Customer Count")
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tenure band
    col3, col4 = st.columns(2)
    with col3:
        if "TenureBand" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Churn Rate by Tenure Band</div>", unsafe_allow_html=True)
            ten_order = ["New (0-1y)","Early (2-3y)","Mid (4-6y)","Loyal (7y+)"]
            ten_df = df.groupby("TenureBand")["Exited"].mean().reset_index()
            ten_df.columns = ["Band","Churn_Rate"]
            ten_df["Churn_Rate"] *= 100
            ten_df["Band"] = pd.Categorical(ten_df["Band"], categories=ten_order, ordered=True)
            ten_df = ten_df.sort_values("Band")
            fig_ten = px.line(ten_df, x="Band", y="Churn_Rate",
                               markers=True, color_discrete_sequence=[C["navy"]])
            fig_ten.add_scatter(x=ten_df["Band"], y=ten_df["Churn_Rate"],
                                mode="markers", marker=dict(size=10, color=C["coral"]), showlegend=False)
            fig_ten.update_traces(line_width=2.5)
            plotly_defaults(fig_ten, 280)
            fig_ten.update_yaxes(title="Churn Rate (%)")
            st.plotly_chart(fig_ten, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        if "EngagementRetentionScore" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Engagement Retention Score vs Churn</div>", unsafe_allow_html=True)
            fig_ers = px.histogram(df, x="EngagementRetentionScore",
                                    color="Churn_Label",
                                    barmode="overlay",
                                    color_discrete_map=CHURN_COLORS,
                                    opacity=0.72, nbins=30)
            plotly_defaults(fig_ers, 280)
            fig_ers.update_xaxes(title="Engagement Retention Score")
            fig_ers.update_yaxes(title="Count")
            st.plotly_chart(fig_ers, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Engagement Retention Ratio KPI:</strong> Inactive members churn at ~27% vs ~14% for active members — a 2× multiplier.
    The <strong>Active_Engaged</strong> segment shows the lowest churn while <strong>Inactive_Disengaged</strong> shows the highest,
    confirming that behavioural engagement is a more reliable retention signal than financial variables alone.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT UTILIZATION
# ═════════════════════════════════════════════════════════════════════════════

def page_product(df):
    st.markdown("<div class='sec-title'>📦 Product Utilization Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>How product depth and engagement with banking products affects retention</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Filter by Product Count", expanded=False):
        prod_range = st.slider(
            "Number of Products", 
            min_value=1, max_value=4, value=(1, 4), step=1
        )
    df = df[df["NumOfProducts"].between(prod_range[0], prod_range[1])]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='chart-card'><div class='chart-title'>Churn Rate by Number of Products</div>", unsafe_allow_html=True)
        prod_df = df.groupby("NumOfProducts")["Exited"].agg(["mean","count"]).reset_index()
        prod_df.columns = ["NumProducts","Churn_Rate","Count"]
        prod_df["Churn_Rate"] *= 100
        bar_colors = [C["teal"] if r < 20 else C["amber"] if r < 50 else C["coral"] for r in prod_df["Churn_Rate"]]
        fig_prod = go.Figure(go.Bar(
            x=prod_df["NumProducts"].astype(str),
            y=prod_df["Churn_Rate"],
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{r:.1f}%<br>n={c}" for r, c in zip(prod_df["Churn_Rate"], prod_df["Count"])],
            textposition="outside",
        ))
        plotly_defaults(fig_prod, 300)
        fig_prod.update_xaxes(title="Number of Products")
        fig_prod.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig_prod, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "ProductDepth" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Product Depth vs Retention</div>", unsafe_allow_html=True)
            depth_df = df.groupby(["ProductDepth","Churn_Label"]).size().reset_index(name="Count")
            total_by_depth = depth_df.groupby("ProductDepth")["Count"].transform("sum")
            depth_df["Pct"] = depth_df["Count"] / total_by_depth * 100
            fig_depth = px.bar(depth_df, x="ProductDepth", y="Pct",
                                color="Churn_Label",
                                color_discrete_map=CHURN_COLORS,
                                barmode="stack",
                                text=[f"{p:.0f}%" for p in depth_df["Pct"]],
                                category_orders={"ProductDepth":["Single","Dual","Multi"]})
            fig_depth.update_traces(textposition="inside", textfont_size=11)
            plotly_defaults(fig_depth, 300)
            fig_depth.update_yaxes(title="Percentage (%)")
            st.plotly_chart(fig_depth, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Credit card stickiness
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='chart-card'><div class='chart-title'>Credit Card Ownership vs Churn</div>", unsafe_allow_html=True)
        cc_df = df.groupby("HasCrCard")["Exited"].agg(["mean","count"]).reset_index()
        cc_df["Label"] = cc_df["HasCrCard"].map({0:"No Card",1:"Has Card"})
        cc_df["Churn_Rate"] = cc_df["mean"] * 100
        fig_cc = go.Figure(go.Bar(
            x=cc_df["Label"], y=cc_df["Churn_Rate"],
            marker_color=[C["amber"], C["teal"]], marker_line_width=0,
            text=[f"{v:.1f}%" for v in cc_df["Churn_Rate"]], textposition="outside"
        ))
        plotly_defaults(fig_cc, 280)
        fig_cc.update_yaxes(title="Churn Rate (%)")
        st.plotly_chart(fig_cc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        if "CreditCardStickinessScore" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Credit Card Stickiness Score KPI</div>", unsafe_allow_html=True)
            fig_stick = px.histogram(df, x="CreditCardStickinessScore",
                                      color="Churn_Label",
                                      barmode="overlay", opacity=0.72,
                                      color_discrete_map=CHURN_COLORS, nbins=25)
            plotly_defaults(fig_stick, 280)
            fig_stick.update_xaxes(title="Stickiness Score")
            st.plotly_chart(fig_stick, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Product Paradox Insight:</strong> 3–4 product customers show 80%+ churn — likely due to
    over-bundled products without genuine engagement. The bank should audit whether multi-product customers
    are <em>actively using</em> all products. The <strong>Product Depth Index KPI</strong> (weighing active usage)
    reveals that product count alone does not guarantee loyalty — active engagement with those products does.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FINANCIAL RISK
# ═════════════════════════════════════════════════════════════════════════════

def page_financial(df):
    st.markdown("<div class='sec-title'>💰 Financial Commitment & Risk Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Balance, salary and financial behaviour cross-analysis for at-risk premium customers</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if "BalanceTier" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Churn Rate by Balance Tier</div>", unsafe_allow_html=True)
            tier_order = ["Zero","Low","Mid","High","Premium"]
            bt_df = df.groupby("BalanceTier")["Exited"].agg(["mean","count"]).reset_index()
            bt_df.columns = ["Tier","Churn_Rate","Count"]
            bt_df["Churn_Rate"] *= 100
            bt_df["Tier"] = pd.Categorical(bt_df["Tier"], categories=tier_order, ordered=True)
            bt_df = bt_df.sort_values("Tier")
            bar_c = [C["teal"] if r < 15 else C["amber"] if r < 25 else C["coral"] for r in bt_df["Churn_Rate"]]
            fig_bt = go.Figure(go.Bar(
                x=bt_df["Tier"], y=bt_df["Churn_Rate"],
                marker_color=bar_c, marker_line_width=0,
                text=[f"{r:.1f}%" for r in bt_df["Churn_Rate"]], textposition="outside"
            ))
            plotly_defaults(fig_bt, 300)
            fig_bt.update_yaxes(title="Churn Rate (%)")
            st.plotly_chart(fig_bt, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "WealthIndex" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Wealth Index vs Churn Distribution</div>", unsafe_allow_html=True)
            fig_wi = px.histogram(df, x="WealthIndex", color="Churn_Label",
                                   barmode="overlay", opacity=0.72,
                                   color_discrete_map=CHURN_COLORS, nbins=30)
            plotly_defaults(fig_wi, 300)
            fig_wi.update_xaxes(title="Wealth Index (0–1)")
            st.plotly_chart(fig_wi, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # At-Risk Premium Analysis
    col3, col4 = st.columns(2)

    with col3:
        if "AtRiskPremiumCustomer" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>At-Risk Premium Customers — Balance vs Activity</div>", unsafe_allow_html=True)
            at_risk_df = df[df["AtRiskPremiumCustomer"] == 1].copy()
            normal_df  = df[df["AtRiskPremiumCustomer"] == 0].sample(min(500, len(df)), random_state=42)
            sample     = pd.concat([at_risk_df.head(300), normal_df.head(200)])
            sample["Label"] = sample["AtRiskPremiumCustomer"].map({0:"Standard", 1:"At-Risk Premium"})
            fig_ar = px.scatter(sample, x="Balance", y="EstimatedSalary",
                                 color="Label", opacity=0.65,
                                 color_discrete_map={"Standard": C["teal"], "At-Risk Premium": C["coral"]},
                                 hover_data=["Age","NumOfProducts"])
            plotly_defaults(fig_ar, 300)
            fig_ar.update_xaxes(title="Account Balance (€)")
            fig_ar.update_yaxes(title="Estimated Salary (€)")
            st.plotly_chart(fig_ar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        if "SalaryBalanceMismatch" in df.columns:
            st.markdown("<div class='chart-card'><div class='chart-title'>Salary-Balance Mismatch Effect on Churn</div>", unsafe_allow_html=True)
            mm_df = df.groupby("SalaryBalanceMismatch")["Exited"].agg(["mean","count"]).reset_index()
            mm_df["Label"] = mm_df["SalaryBalanceMismatch"].map({0:"Aligned",1:"Mismatch"})
            mm_df["Churn_Rate"] = mm_df["mean"] * 100
            fig_mm = go.Figure(go.Bar(
                x=mm_df["Label"], y=mm_df["Churn_Rate"],
                marker_color=[C["teal"], C["coral"]], marker_line_width=0,
                text=[f"{v:.1f}%" for v in mm_df["Churn_Rate"]], textposition="outside"
            ))
            plotly_defaults(fig_mm, 300)
            fig_mm.update_yaxes(title="Churn Rate (%)")
            st.plotly_chart(fig_mm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Geography + Balance heatmap
    if "Geography" in df.columns and "BalanceTier" in df.columns:
        st.markdown("<div class='chart-card'><div class='chart-title'>Churn Heatmap — Geography × Balance Tier</div>", unsafe_allow_html=True)
        tier_order = ["Zero","Low","Mid","High","Premium"]
        heat_df = df.groupby(["Geography","BalanceTier"])["Exited"].mean().reset_index()
        heat_df.columns = ["Geography","BalanceTier","Churn_Rate"]
        heat_df["Churn_Rate"] *= 100
        heat_pivot = heat_df.pivot(index="Geography", columns="BalanceTier", values="Churn_Rate")
        heat_pivot = heat_pivot.reindex(columns=[c for c in tier_order if c in heat_pivot.columns])
        fig_heat = px.imshow(heat_pivot, color_continuous_scale=[[0,C["teal"]],[0.5,C["amber"]],[1,C["coral"]]],
                              text_auto=".1f", aspect="auto")
        fig_heat.update_coloraxes(colorbar_title="Churn %")
        plotly_defaults(fig_heat, 260)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DISENGAGED DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

def page_disengaged(df):
    st.markdown("<div class='sec-title'>🔍 High-Value Disengaged Customer Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Identify premium customers who are disengaged and at imminent churn risk — your highest-priority retention targets</div>", unsafe_allow_html=True)

    # Filters
    with st.expander("⚙️ Filter Controls", expanded=True):
        fc1, fc2, fc3, fc4 , fc5 = st.columns(5)
        with fc1:
            min_bal = st.slider("Min Account Balance (€)", 0, 250000, 50000, step=5000)
        with fc2:
            min_sal = st.slider("Min Estimated Salary (€)", 0, 200000, 0, step=10000)
        with fc3:
            geo_opts = ["All"] + sorted(df["Geography"].unique().tolist()) if "Geography" in df.columns else ["All"]
            geo_sel  = st.selectbox("Geography", geo_opts)
        with fc4:
            age_range = st.slider("Age Range", 18, 92, (25, 70))
        with fc5:
            prod_range = st.slider("Product Count", 1, 4, (1, 4), step=1)

    filt = df.copy()
    filt = filt[filt["Balance"] >= min_bal]
    filt = filt[filt["EstimatedSalary"] >= min_sal]
    if geo_sel != "All":
        filt = filt[filt["Geography"] == geo_sel]
    filt = filt[(filt["Age"] >= age_range[0]) & (filt["Age"] <= age_range[1])]
    filt = filt[filt["NumOfProducts"].between(prod_range[0], prod_range[1])]
    filt = filt[filt["IsActiveMember"] == 0]  # disengaged

    st.markdown(f"<div style='margin: 8px 0 16px 0; font-size:0.85rem; color:{C['muted']};'>Showing <b style='color:{C['navy']}'>{len(filt):,}</b> disengaged customers matching your criteria</div>", unsafe_allow_html=True)

    # Summary KPIs
    if len(filt) > 0:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(kpi_card("🚨", fmt_num(len(filt)), "At-Risk Customers", "Inactive & high-value", C["coral"]), unsafe_allow_html=True)
        with k2: st.markdown(kpi_card("💵", f"€{filt['Balance'].mean():,.0f}", "Avg Balance", "Potential revenue at stake", C["amber"]), unsafe_allow_html=True)
        with k3: st.markdown(kpi_card("📅", f"{filt['Tenure'].mean():.1f} yrs", "Avg Tenure", "Established relationships", C["navy"]), unsafe_allow_html=True)
        with k4:
            actual_churn = filt["Exited"].mean() * 100 if "Exited" in filt.columns else 0
            st.markdown(kpi_card("📉", f"{actual_churn:.1f}%", "Actual Churn Rate", "In filtered segment", C["teal"]), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Display table
        display_cols = ["Age","Geography","Balance","EstimatedSalary","NumOfProducts",
                        "Tenure","HasCrCard","Exited"]
        if "RetentionRiskScore" in filt.columns:
            display_cols.insert(0, "RetentionRiskScore")
        if "RelationshipStrengthIndex" in filt.columns:
            display_cols.insert(1, "RelationshipStrengthIndex")

        show_df = filt[[c for c in display_cols if c in filt.columns]].copy()
        if "RetentionRiskScore" in show_df.columns:
            show_df = show_df.sort_values("RetentionRiskScore", ascending=False)
            show_df["RetentionRiskScore"] = show_df["RetentionRiskScore"].round(2)
        if "RelationshipStrengthIndex" in show_df.columns:
            show_df["RelationshipStrengthIndex"] = show_df["RelationshipStrengthIndex"].round(3)
        if "Balance" in show_df.columns:
            show_df["Balance"] = show_df["Balance"].apply(lambda x: f"€{x:,.0f}")
        if "EstimatedSalary" in show_df.columns:
            show_df["EstimatedSalary"] = show_df["EstimatedSalary"].apply(lambda x: f"€{x:,.0f}")

        st.dataframe(show_df.head(200), use_container_width=True, hide_index=True)

        # Download
        csv_buf = io.BytesIO()
        filt.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️  Export Filtered List as CSV",
            data=csv_buf.getvalue(),
            file_name="high_value_disengaged_customers.csv",
            mime="text/csv",
        )

        # Scatter chart
        st.markdown("<div class='chart-card' style='margin-top:16px'><div class='chart-title'>Balance vs Tenure — Disengaged Customers</div>", unsafe_allow_html=True)
        color_col = "RetentionRiskScore" if "RetentionRiskScore" in filt.columns else "Age"
        fig_dis = px.scatter(filt.head(400), x="Tenure", y="Balance",
                              color=color_col,
                              color_continuous_scale=[[0, C["teal"]], [0.5, C["amber"]], [1, C["coral"]]],
                              size="Balance", size_max=22, opacity=0.7,
                              hover_data=["Age","Geography","NumOfProducts"])
        plotly_defaults(fig_dis, 380)
        fig_dis.update_xaxes(title="Tenure (years)")
        fig_dis.update_yaxes(title="Account Balance (€)")
        fig_dis.update_coloraxes(colorbar_title="Risk Score")
        st.plotly_chart(fig_dis, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No customers match the current filter criteria. Try relaxing the filters.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INDIVIDUAL RISK SCORER
# ═════════════════════════════════════════════════════════════════════════════

def compute_single_features(inp, ref_df):
    """Compute engineered features for a single customer using reference dataset stats."""
    df = pd.DataFrame([inp])

    balance_median = ref_df["Balance"].median()
    salary_median  = ref_df["EstimatedSalary"].median()
    bal_min, bal_max = ref_df["Balance"].min(), ref_df["Balance"].max()
    sal_min, sal_max = ref_df["EstimatedSalary"].min(), ref_df["EstimatedSalary"].max()
    tenure_max       = ref_df["Tenure"].max()

    b = inp["Balance"];  n = inp["NumOfProducts"];  a = inp["IsActiveMember"]
    c = inp["HasCrCard"]; t = inp["Tenure"];         cs = inp["CreditScore"]
    age = inp["Age"];    s = inp["EstimatedSalary"]; geo = inp["Geography"]

    # Engagement
    if   a == 1 and n >= 2: seg = "Active_Engaged"
    elif a == 0 and n <= 1: seg = "Inactive_Disengaged"
    elif a == 1 and n == 1: seg = "Active_LowProduct"
    elif a == 0 and b > balance_median: seg = "Inactive_HighBalance"
    else: seg = "Other"
    df["EngagementSegment"] = seg

    df["ActivityScore"]       = a * 3 + c * 1 + int(n >= 2) * 2
    df["IsLongTenureActive"]  = int(t >= 5 and a == 1)

    # Tenure band (ordinal)
    if t <= 1:   tb = 0
    elif t <= 3: tb = 1
    elif t <= 6: tb = 2
    else:        tb = 3
    df["TenureBand"] = tb

    # Product
    if n == 1:   pd_enc = 0
    elif n == 2: pd_enc = 1
    else:        pd_enc = 2
    df["ProductDepth"]              = pd_enc
    df["IsMultiProduct"]            = int(n >= 2)
    df["ProductActivityIndex"]      = n * a
    df["CardActiveCombo"]           = int(c == 1 and a == 1)
    df["ProductEngagementRatio"]    = n / 4.0
    df["CreditCardStickinessScore"] = c * 0.4 + a * 0.4 + (n / 4) * 0.2

    # Financial
    if b == 0:        bt = 0
    elif b <= 50000:  bt = 1
    elif b <= 100000: bt = 2
    elif b <= 150000: bt = 3
    else:             bt = 4
    df["BalanceTier"]           = bt
    df["IsZeroBalance"]         = int(b == 0)
    df["BalanceToSalaryRatio"]  = b / (s + 1)
    df["SalaryBalanceMismatch"] = int(s > salary_median and b < balance_median * 0.25)
    df["AtRiskPremiumCustomer"] = int(b > balance_median and a == 0)
    df["HighBalanceActive"]     = int(b > balance_median and a == 1)

    # Salary tier
    q25, q50, q75 = ref_df["EstimatedSalary"].quantile([0.25, 0.50, 0.75])
    if s <= q25:   st_enc = 0
    elif s <= q50: st_enc = 1
    elif s <= q75: st_enc = 2
    else:          st_enc = 3
    df["SalaryTier"] = st_enc

    # Wealth index
    bal_norm = (b - bal_min) / (bal_max - bal_min + 1e-9)
    sal_norm = (s - sal_min) / (sal_max - sal_min + 1e-9)
    df["WealthIndex"] = (bal_norm + sal_norm) / 2.0

    # Demographic
    if age <= 25:    ab = 0
    elif age <= 35:  ab = 1
    elif age <= 45:  ab = 2
    elif age <= 55:  ab = 3
    elif age <= 65:  ab = 4
    else:            ab = 5
    df["AgeBand"]          = ab
    df["IsSeniorRisk"]     = int(age > 55)
    df["GeographyEncoded"] = {"France":0,"Spain":1,"Germany":2}.get(geo, 0)
    df["GermanyHighBalance"] = int(geo == "Germany" and b > balance_median)

    from sklearn.preprocessing import LabelEncoder
    gender_enc = {"Male": 1, "Female": 0}.get(inp["Gender"], 0)
    df["GenderEncoded"] = gender_enc

    # Credit score band
    if cs <= 579:   csb = 0
    elif cs <= 669: csb = 1
    elif cs <= 739: csb = 2
    elif cs <= 799: csb = 3
    else:           csb = 4
    df["CreditScoreBand"]  = csb
    df["AgeTenureProduct"] = age * t
    df["YoungLowTenure"]   = int(age < 35 and t <= 2)

    # KPI composites
    df["EngagementRetentionScore"] = a * 0.5 + (t / (tenure_max + 1e-9)) * 0.3 + c * 0.2
    df["ProductDepthIndex"]        = (n / 4) * 0.6 + a * 0.4
    df["HighBalanceDisengaged"]    = int(b > ref_df["Balance"].quantile(0.75) and a == 0)

    credit_min = ref_df["CreditScore"].min(); credit_max = ref_df["CreditScore"].max()
    credit_norm = (cs - credit_min) / (credit_max - credit_min + 1e-9)
    tenure_norm = t / (tenure_max + 1e-9)
    balance_norm_rsi = (b - bal_min) / (bal_max - bal_min + 1e-9)
    pdi = (n / 4) * 0.6 + a * 0.4

    df["RelationshipStrengthIndex"] = (
        a * 0.25 + pdi * 0.25 + tenure_norm * 0.20 +
        balance_norm_rsi * 0.15 + credit_norm * 0.10 + c * 0.05
    )
    rsi_threshold = ref_df["RelationshipStrengthIndex"].quantile(0.70) if "RelationshipStrengthIndex" in ref_df.columns else 0.5
    df["IsStickyCustomer"] = int(df["RelationshipStrengthIndex"].iloc[0] >= rsi_threshold)

    # Risk score
    risk = (a == 0) * 2.0 + (n == 1) * 1.5 + (n > 2) * 2.5 + int(b == 0) * 1.0
    risk += int(b > balance_median and a == 0) * 2.0
    risk += int(s > salary_median and b < balance_median * 0.25) * 1.0
    risk += int(age > 55) * 1.0 + int(t <= 1) * 1.0
    risk += int(cs < 580) * 0.5 + int(geo == "Germany" and b > balance_median) * 1.0
    risk_min = 0; risk_max = 12.5
    df["RetentionRiskScore"] = (risk - risk_min) / (risk_max - risk_min + 1e-9) * 10.0
    if df["RetentionRiskScore"].iloc[0] <= 3.33: rt = 0
    elif df["RetentionRiskScore"].iloc[0] <= 6.66: rt = 1
    else: rt = 2
    df["RiskTier"] = rt

    # OHE for Geography / Gender / EngagementSegment
    geo_map  = {"France": (0,0), "Germany": (1,0), "Spain": (0,1)}
    gg, gs   = geo_map.get(geo, (0,0))
    df["Geography_Germany"] = gg
    df["Geography_Spain"]   = gs

    seg_cols = ["EngagementSegment_Inactive_Disengaged","EngagementSegment_Inactive_HighBalance",
                "EngagementSegment_Other","EngagementSegment_Active_LowProduct"]
    for sc in seg_cols: df[sc] = 0
    seg_col = f"EngagementSegment_{seg}"
    if seg_col in df.columns: df[seg_col] = 1

    df["Gender_Male"] = int(inp["Gender"] == "Male")
    return df


def page_risk_scorer(df, model_obj, preprocessor):
    st.markdown("<div class='sec-title'>🎯 Individual Customer Risk Scorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Enter customer details to get real-time churn probability, risk band, and explainability insights</div>", unsafe_allow_html=True)

    if model_obj is None:
        st.warning("⚠️ Model not found. Run `python src/components/data_ingestion.py` to train the pipeline first.")
        return

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Customer Profile Input</div>", unsafe_allow_html=True)
        with st.form("risk_form"):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                credit_score = st.number_input("Credit Score", 300, 850, 650)
                age          = st.number_input("Age", 18, 92, 38)
                balance      = st.number_input("Account Balance (€)", 0, 300000, 85000, step=5000)
                num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            with r1c2:
                geography    = st.selectbox("Geography", ["France", "Germany", "Spain"])
                gender       = st.selectbox("Gender", ["Male", "Female"])
                tenure       = st.slider("Tenure (Years)", 0, 10, 4)
                salary       = st.number_input("Estimated Salary (€)", 0, 250000, 80000, step=5000)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                has_cc = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
            with r2c2:
                is_active = st.radio("Active Member?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)

            submitted = st.form_submit_button("🔮  Predict Churn Risk", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            raw_input = {
                "CreditScore": credit_score, "Geography": geography, "Gender": gender,
                "Age": age, "Tenure": tenure, "Balance": balance,
                "NumOfProducts": num_products, "HasCrCard": has_cc,
                "IsActiveMember": is_active, "EstimatedSalary": salary,
            }

            try:
                # Build feature row
                feat_df  = compute_single_features(raw_input, df)
                model    = model_obj["model"]
                threshold = model_obj.get("threshold", 0.30)

                # Align to expected feature columns if preprocessor available
                if preprocessor:
                    expected_cols = preprocessor.get("feature_columns", [])
                    numeric_cols  = preprocessor.get("numeric_cols", [])
                    scaler        = preprocessor.get("scaler", None)

                    for col in expected_cols:
                        if col not in feat_df.columns:
                            feat_df[col] = 0
                    feat_df = feat_df[expected_cols]
                    if scaler and numeric_cols:
                        num_present = [c for c in numeric_cols if c in feat_df.columns]
                        feat_df[num_present] = scaler.transform(feat_df[num_present])
                else:
                    feat_df = feat_df.select_dtypes(include=[np.number])

                X_input = feat_df.values
                proba   = model.predict_proba(X_input)[0][1]
                pred    = int(proba >= threshold)

                # ── Risk result display ──
                if proba >= 0.70:   risk_band, badge_cls, band_emoji = "High Risk",   "badge-high", "🔴"
                elif proba >= 0.40: risk_band, badge_cls, band_emoji = "Medium Risk", "badge-med",  "🟡"
                else:               risk_band, badge_cls, band_emoji = "Low Risk",    "badge-low",  "🟢"

                st.markdown(f"""
                <div class="risk-hero">
                    <div style="font-size:0.7rem;letter-spacing:0.15em;opacity:0.65;text-transform:uppercase;margin-bottom:8px">Churn Probability</div>
                    <div class="risk-pct">{proba*100:.1f}%</div>
                    <div style="margin-top:14px">
                        <span class="badge {'badge-high' if risk_band=='High Risk' else 'badge-med' if risk_band=='Medium Risk' else 'badge-low'}">
                            {band_emoji} {risk_band}
                        </span>
                    </div>
                    <div style="font-size:0.78rem;opacity:0.65;margin-top:10px">Threshold: {threshold:.2f} | Prediction: {'⚠️ Will Churn' if pred else '✅ Will Retain'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability gauge
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%", "font": {"size": 28, "family": "Lora, serif", "color": C["navy"]}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickfont": {"size": 10}},
                        "bar":   {"color": C["coral"] if proba >= 0.5 else C["amber"] if proba >= 0.3 else C["teal"]},
                        "steps": [
                            {"range": [0,  40],  "color": "#D1FAE5"},
                            {"range": [40, 70],  "color": "#FEF3C7"},
                            {"range": [70, 100], "color": "#FEE2E2"},
                        ],
                        "threshold": {"line": {"color": C["navy"], "width": 3}, "value": threshold * 100},
                    },
                ))
                gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20),
                                    paper_bgcolor="rgba(0,0,0,0)", font_family="Plus Jakarta Sans")
                st.plotly_chart(gauge, use_container_width=True)

                # Key risk factors (rule-based, always available)
                st.markdown("<div class='chart-card'><div class='chart-title'>Key Risk Factors</div>", unsafe_allow_html=True)
                factors = []
                if is_active == 0:        factors.append(("🔴 Inactive Member", "Strongest churn predictor — 3× higher risk"))
                if num_products in [3,4]: factors.append(("🔴 Multi-Product Overload", "3+ products linked to 80%+ churn"))
                if geography == "Germany": factors.append(("🟡 Germany Region", "Highest regional churn rate (~32%)"))
                if balance > 100000 and is_active == 0: factors.append(("🔴 At-Risk Premium", "High-value but disengaged"))
                if age > 55:              factors.append(("🟡 Senior Risk Flag", "Age > 55 is an elevated risk indicator"))
                if tenure <= 1:           factors.append(("🟡 New Customer", "Low tenure increases exit probability"))
                if credit_score < 580:    factors.append(("🟡 Low Credit Score", "Below 580 threshold"))
                if not factors:           factors.append(("🟢 No Major Flags", "Customer profile appears stable"))

                for factor, desc in factors:
                    st.markdown(f"<div class='insight-box'><strong>{factor}</strong><br>{desc}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # SHAP (optional)
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    with st.spinner("Computing SHAP explainability..."):
                        bg_sample = df.drop(columns=["Exited","Churn_Label"], errors="ignore")
                        bg_sample = bg_sample.select_dtypes(include=[np.number]).sample(
                            min(80, len(bg_sample)), random_state=42)
                        if preprocessor:
                            for col in expected_cols:
                                if col not in bg_sample.columns: bg_sample[col] = 0
                            bg_sample = bg_sample.reindex(columns=expected_cols, fill_value=0)

                        explainer  = shap.KernelExplainer(model.predict_proba, bg_sample.values, link="identity")
                        shap_vals  = explainer.shap_values(X_input, nsamples=60)
                        sv_class1  = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                        feat_names = list(feat_df.columns)

                        fig_shap, ax = plt.subplots(figsize=(7, 4))
                        top_n = 12
                        indices  = np.argsort(np.abs(sv_class1))[-top_n:][::-1]
                        top_vals = sv_class1[indices]
                        top_feats = [feat_names[i] for i in indices]
                        colors_shap = [C["coral"] if v > 0 else C["teal"] for v in top_vals]
                        ax.barh(top_feats[::-1], top_vals[::-1], color=colors_shap[::-1])
                        ax.axvline(0, color=C["muted"], linewidth=0.8, linestyle="--")
                        ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=9)
                        ax.set_title("Feature Contributions (SHAP)", fontsize=11, fontweight="bold", color=C["navy"])
                        ax.tick_params(axis="y", labelsize=8)
                        ax.tick_params(axis="x", labelsize=8)
                        fig_shap.patch.set_alpha(0)
                        ax.set_facecolor("#F7F9FD")
                        plt.tight_layout()
                        st.markdown("<div class='chart-card'><div class='chart-title'>SHAP Explainability</div>", unsafe_allow_html=True)
                        st.pyplot(fig_shap)
                        st.markdown("</div>", unsafe_allow_html=True)
                        plt.close(fig_shap)
                except Exception:
                    pass  # SHAP is optional enhancement

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure the pipeline artifacts exist in the `artifacts/` folder.")

        else:
            # Placeholder
            st.markdown(f"""
            <div style="background:{C['card']};border-radius:16px;padding:48px 32px;
                        text-align:center;box-shadow:0 2px 16px rgba(27,59,111,0.06);
                        border: 2px dashed {C['border']}">
                <div style="font-size:3rem;margin-bottom:16px">🎯</div>
                <div style="font-family:'Lora',serif;font-size:1.3rem;color:{C['navy']};font-weight:700;margin-bottom:8px">
                    Ready to Score
                </div>
                <div style="font-size:0.875rem;color:{C['muted']};max-width:280px;margin:0 auto;line-height:1.6">
                    Fill in the customer profile on the left and click <strong>Predict Churn Risk</strong>
                    to get an instant assessment with explainability.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    with st.spinner("Loading data and model..."):
        df          = load_data()
        model_obj   = load_model()
        preprocessor = load_preprocessor()

    if df is None:
        st.error("📁 Dataset not found. Please ensure `artifacts/raw.csv` or `notebook/data/European_Bank.csv` exists.")
        st.code("# Run this first:\npython src/components/data_ingestion.py")
        return

    page = render_sidebar(df, model_obj)

    if   "Overview"     in page: page_overview(df)
    elif "Engagement"   in page: page_engagement(df)
    elif "Product"      in page: page_product(df)
    elif "Financial"    in page: page_financial(df)
    elif "Disengaged"   in page: page_disengaged(df)
    elif "Risk Scorer"  in page: page_risk_scorer(df, model_obj, preprocessor)


if __name__ == "__main__":
    main()