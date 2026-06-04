import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.charts import *
from utils.insights import *
from utils.ml_models import *

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Industry Analysis",
    page_icon="🏭",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("🏭 Industry Intelligence Dashboard")

st.markdown("""
Deep industry analysis for:

- Venture Capital Firms
- Startup Accelerators
- Corporate Innovation Teams
- Investors
- Founders
- Market Analysts
""")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Industry Filters")

selected_industries = st.sidebar.multiselect(
    "Select Industry",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

df = df[
    df["Industry"].isin(selected_industries)
]

# =====================================================
# INDUSTRY KPIs
# =====================================================

st.subheader("📊 Industry Overview")

total_industries = df["Industry"].nunique()

total_funding = df["Funding Amount (M USD)"].sum()

total_revenue = df["Revenue (M USD)"].sum()

avg_valuation = df["Valuation (M USD)"].mean()

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Industries",
    total_industries
)

col2.metric(
    "Funding",
    f"${total_funding:,.2f} M"
)

col3.metric(
    "Revenue",
    f"${total_revenue:,.2f} M"
)

col4.metric(
    "Avg Valuation",
    f"${avg_valuation:,.2f} M"
)

st.divider()

# =====================================================
# INDUSTRY SUMMARY TABLE
# =====================================================

industry_summary = (

    df.groupby("Industry")
    .agg({

        "Funding Amount (M USD)": "sum",
        "Revenue (M USD)": "sum",
        "Valuation (M USD)": "mean",
        "Employees": "sum",
        "Market Share (%)": "mean"

    })

    .reset_index()

)

# =====================================================
# FUNDING ANALYSIS
# =====================================================

st.subheader("💰 Industry Funding Analysis")

fig = px.bar(
    industry_summary,
    x="Industry",
    y="Funding Amount (M USD)",
    color="Industry",
    title="Funding by Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# REVENUE ANALYSIS
# =====================================================

st.subheader("📈 Industry Revenue Analysis")

fig = px.bar(
    industry_summary,
    x="Industry",
    y="Revenue (M USD)",
    color="Industry",
    title="Revenue by Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# VALUATION ANALYSIS
# =====================================================

st.subheader("💎 Industry Valuation Analysis")

fig = px.bar(
    industry_summary,
    x="Industry",
    y="Valuation (M USD)",
    color="Industry",
    title="Average Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# INDUSTRY PERFORMANCE MATRIX
# =====================================================

st.subheader("🚀 Industry Growth Matrix")

fig = px.scatter(
    industry_summary,
    x="Funding Amount (M USD)",
    y="Revenue (M USD)",
    size="Valuation (M USD)",
    color="Industry",
    hover_name="Industry",
    title="Funding vs Revenue vs Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# MARKET SHARE ANALYSIS
# =====================================================

st.subheader("🌍 Market Share Analysis")

fig = px.treemap(
    df,
    path=[
        "Industry",
        "Startup Name"
    ],
    values="Market Share (%)",
    color="Market Share (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# FUNDING EFFICIENCY
# =====================================================

st.subheader("⚡ Industry Funding Efficiency")

df["Funding Efficiency"] = (

    df["Revenue (M USD)"]
    /
    df["Funding Amount (M USD)"]

)

efficiency = (

    df.groupby("Industry")
    ["Funding Efficiency"]
    .mean()
    .reset_index()

)

fig = px.bar(
    efficiency,
    x="Industry",
    y="Funding Efficiency",
    color="Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# REVENUE PER EMPLOYEE
# =====================================================

st.subheader("👨‍💼 Revenue Per Employee")

df["Revenue Per Employee"] = (

    df["Revenue (M USD)"]
    /
    df["Employees"]

)

employee_perf = (

    df.groupby("Industry")
    ["Revenue Per Employee"]
    .mean()
    .reset_index()

)

fig = px.bar(
    employee_perf,
    x="Industry",
    y="Revenue Per Employee",
    color="Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# PROFITABILITY ANALYSIS
# =====================================================

st.subheader("📊 Industry Profitability")

if "Profitable" in df.columns:

    profitability = (

        df.groupby("Industry")
        ["Profitable"]
        .mean()
        .reset_index()

    )

    fig = px.bar(
        profitability,
        x="Industry",
        y="Profitable",
        color="Industry"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# INDUSTRY HEALTH SCORE
# =====================================================

st.subheader("❤️ Industry Health Score")

health_df = calculate_health_score(df)

industry_health = (

    health_df.groupby("Industry")
    ["Health Score"]
    .mean()
    .reset_index()

)

fig = px.bar(
    industry_health,
    x="Industry",
    y="Health Score",
    color="Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# INDUSTRY RANKING
# =====================================================

st.subheader("🏆 Industry Ranking")

ranking = industry_ranking(df)

st.dataframe(
    ranking,
    use_container_width=True
)

# =====================================================
# UNICORN ANALYSIS
# =====================================================

st.subheader("🦄 Industry Unicorn Analysis")

df["Unicorn"] = (

    df["Valuation (M USD)"]
    >=
    1000

)

unicorns = (

    df.groupby("Industry")
    ["Unicorn"]
    .sum()
    .reset_index()

)

fig = px.bar(
    unicorns,
    x="Industry",
    y="Unicorn",
    color="Industry",
    title="Unicorn Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RISK ANALYSIS
# =====================================================

st.subheader("🚨 Industry Risk Analysis")

risk_df = calculate_risk_score(df)

industry_risk = (

    risk_df.groupby("Industry")
    ["Risk Score"]
    .mean()
    .reset_index()

)

fig = px.bar(
    industry_risk,
    x="Industry",
    y="Risk Score",
    color="Industry"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

st.subheader("🔥 Industry Correlation Heatmap")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

# =====================================================
# TOP INDUSTRIES TABLE
# =====================================================

st.subheader("📋 Industry Benchmarking")

benchmark = (

    df.groupby("Industry")
    .agg({

        "Funding Amount (M USD)": "sum",
        "Revenue (M USD)": "sum",
        "Valuation (M USD)": "mean",
        "Employees": "sum"

    })

    .sort_values(
        "Revenue (M USD)",
        ascending=False
    )

)

st.dataframe(
    benchmark,
    use_container_width=True
)

# =====================================================
# ML FEATURE IMPORTANCE
# =====================================================

st.subheader("🤖 Valuation Drivers")

importance = valuation_feature_importance(df)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Factors Affecting Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# AI INDUSTRY INSIGHTS
# =====================================================

st.subheader("🧠 AI Industry Insights")

top_revenue_industry = (
    df.groupby("Industry")
    ["Revenue (M USD)"]
    .sum()
    .idxmax()
)

top_funding_industry = (
    df.groupby("Industry")
    ["Funding Amount (M USD)"]
    .sum()
    .idxmax()
)

top_valuation_industry = (
    df.groupby("Industry")
    ["Valuation (M USD)"]
    .mean()
    .idxmax()
)

st.success(f"""
INDUSTRY INTELLIGENCE REPORT

• Top Revenue Industry:
{top_revenue_industry}

• Top Funding Industry:
{top_funding_industry}

• Highest Valuation Industry:
{top_valuation_industry}

• Industries Analyzed:
{df['Industry'].nunique()}

• Total Startups:
{len(df)}

Recommendation:

Focus investment opportunities
in industries demonstrating strong
revenue growth, healthy valuation
multiples, and superior funding
efficiency.
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

report_text = f"""
Industry Intelligence Report

Top Revenue Industry:
{top_revenue_industry}

Top Funding Industry:
{top_funding_industry}

Highest Valuation Industry:
{top_valuation_industry}

Industries Analyzed:
{df['Industry'].nunique()}
"""

st.download_button(
    label="📥 Download Industry Report",
    data=report_text,
    file_name="Industry_Analysis_Report.txt",
    mime="text/plain"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Industry Intelligence Dashboard | Powered by Streamlit + AI + Machine Learning"
)
