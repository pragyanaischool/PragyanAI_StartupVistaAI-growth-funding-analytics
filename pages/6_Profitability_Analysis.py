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
    page_title="Profitability Analysis",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# =====================================================
# HANDLE PROFITABILITY COLUMN
# =====================================================

if "Profitable" not in df.columns:

    median_revenue = df["Revenue (M USD)"].median()

    df["Profitable"] = (
        df["Revenue (M USD)"] > median_revenue
    ).astype(int)

# =====================================================
# HEADER
# =====================================================

st.title("📈 Startup Profitability Intelligence Dashboard")

st.markdown("""
Analyze profitability performance across startups,
industries, and regions to identify sustainable
business growth opportunities.
""")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Profitability Filters")

industry_filter = st.sidebar.multiselect(
    "Industry",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

region_filter = st.sidebar.multiselect(
    "Region",
    sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

df = df[
    (df["Industry"].isin(industry_filter))
    &
    (df["Region"].isin(region_filter))
]

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("📊 Profitability Overview")

total_revenue = df["Revenue (M USD)"].sum()

avg_revenue = df["Revenue (M USD)"].mean()

profitable_count = df["Profitable"].sum()

profitability_rate = (
    profitable_count / len(df)
) * 100

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"${total_revenue:,.0f} M"
)

col2.metric(
    "Average Revenue",
    f"${avg_revenue:,.2f} M"
)

col3.metric(
    "Profitable Startups",
    int(profitable_count)
)

col4.metric(
    "Profitability %",
    f"{profitability_rate:.1f}%"
)

st.divider()

# =====================================================
# PROFITABILITY DISTRIBUTION
# =====================================================

st.subheader("📈 Profitability Distribution")

col1,col2 = st.columns(2)

with col1:

    profitability_counts = (
        df["Profitable"]
        .value_counts()
        .reset_index()
    )

    profitability_counts.columns = [
        "Status",
        "Count"
    ]

    profitability_counts["Status"] = (
        profitability_counts["Status"]
        .map({
            1:"Profitable",
            0:"Non-Profitable"
        })
    )

    fig = px.pie(
        profitability_counts,
        names="Status",
        values="Count",
        title="Profitability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="Profitable",
        y="Revenue (M USD)",
        color="Profitable",
        title="Revenue by Profitability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# INDUSTRY PROFITABILITY
# =====================================================

st.subheader("🏭 Industry Profitability")

industry_profit = (

    df.groupby("Industry")
    ["Profitable"]
    .mean()
    .reset_index()

)

industry_profit["Profitable"] *= 100

fig = px.bar(
    industry_profit,
    x="Industry",
    y="Profitable",
    color="Industry",
    title="Industry Profitability Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# REGIONAL PROFITABILITY
# =====================================================

st.subheader("🌎 Regional Profitability")

regional_profit = (

    df.groupby("Region")
    ["Profitable"]
    .mean()
    .reset_index()

)

regional_profit["Profitable"] *= 100

fig = px.bar(
    regional_profit,
    x="Region",
    y="Profitable",
    color="Region",
    title="Regional Profitability Rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# REVENUE ANALYSIS
# =====================================================

st.subheader("💰 Revenue Analysis")

st.plotly_chart(
    revenue_by_industry(df),
    use_container_width=True
)

# =====================================================
# FUNDING EFFICIENCY
# =====================================================

st.subheader("⚡ Funding Efficiency")

df["Funding Efficiency"] = (
    df["Revenue (M USD)"]
    /
    df["Funding Amount (M USD)"]
)

st.plotly_chart(
    funding_efficiency_chart(df),
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

st.plotly_chart(
    revenue_per_employee_chart(df),
    use_container_width=True
)

# =====================================================
# HEALTH SCORE
# =====================================================

st.subheader("❤️ Startup Health Score")

health_df = calculate_health_score(df)

st.plotly_chart(
    health_score_chart(health_df),
    use_container_width=True
)

# =====================================================
# TOP PROFITABLE STARTUPS
# =====================================================

st.subheader("🏆 Top Revenue Generating Startups")

top_profit = (

    df.sort_values(
        "Revenue (M USD)",
        ascending=False
    )
    .head(20)

)

st.dataframe(
    top_profit[
        [
            "Startup Name",
            "Industry",
            "Region",
            "Revenue (M USD)",
            "Valuation (M USD)"
        ]
    ],
    use_container_width=True
)

# =====================================================
# PROFITABILITY MATRIX
# =====================================================

st.subheader("🚀 Profitability Matrix")

fig = px.scatter(
    df,
    x="Funding Amount (M USD)",
    y="Revenue (M USD)",
    size="Valuation (M USD)",
    color="Industry",
    hover_name="Startup Name",
    title="Funding vs Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RISK ANALYSIS
# =====================================================

st.subheader("🚨 Profitability Risk Analysis")

risk_df = calculate_risk_score(df)

st.dataframe(
    risk_df[
        [
            "Startup Name",
            "Industry",
            "Risk Score"
        ]
    ]
    .sort_values(
        "Risk Score",
        ascending=False
    )
    .head(20),
    use_container_width=True
)

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

st.subheader("🔥 Profitability Correlation Analysis")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

# =====================================================
# ML INSIGHTS
# =====================================================

st.subheader("🤖 Machine Learning Insights")

model = train_valuation_model(df)

col1,col2 = st.columns(2)

with col1:

    st.metric(
        "Valuation Prediction R²",
        round(
            model["score"],
            3
        )
    )

with col2:

    importance = valuation_feature_importance(df)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🧠 AI Profitability Insights")

top_industry = (
    df.groupby("Industry")
    ["Revenue (M USD)"]
    .sum()
    .idxmax()
)

top_region = (
    df.groupby("Region")
    ["Revenue (M USD)"]
    .sum()
    .idxmax()
)

st.success(f"""
PROFITABILITY INTELLIGENCE REPORT

• Total Revenue:
${df['Revenue (M USD)'].sum():,.2f} M

• Average Revenue:
${df['Revenue (M USD)'].mean():,.2f} M

• Profitability Rate:
{profitability_rate:.2f}%

• Top Revenue Industry:
{top_industry}

• Top Revenue Region:
{top_region}

Recommendation:

Focus on startups with strong revenue,
high funding efficiency, sustainable
market share growth and healthy
valuation multiples.
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

report = f"""
Profitability Analysis Report

Revenue:
${df['Revenue (M USD)'].sum():,.2f} M

Profitability Rate:
{profitability_rate:.2f}%

Top Industry:
{top_industry}

Top Region:
{top_region}
"""

st.download_button(
    "📥 Download Profitability Report",
    data=report,
    file_name="Profitability_Report.txt",
    mime="text/plain"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Profitability Dashboard | Powered by AI + ML + Streamlit"
)
