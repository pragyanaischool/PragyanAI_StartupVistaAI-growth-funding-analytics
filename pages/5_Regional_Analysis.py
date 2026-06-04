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
    page_title="Regional Analysis",
    page_icon="🌎",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("🌎 Regional Intelligence Dashboard")

st.markdown("""
Analyze startup ecosystems across regions,
identify emerging hubs, discover investment
opportunities, and benchmark regional performance.
""")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Regional Filters")

selected_regions = st.sidebar.multiselect(
    "Select Regions",
    sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

selected_industries = st.sidebar.multiselect(
    "Industry",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

df = df[
    (df["Region"].isin(selected_regions))
    &
    (df["Industry"].isin(selected_industries))
]

# =====================================================
# REGIONAL KPIs
# =====================================================

st.subheader("📊 Regional Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Regions",
    df["Region"].nunique()
)

col2.metric(
    "Startups",
    len(df)
)

col3.metric(
    "Funding",
    f"${df['Funding Amount (M USD)'].sum():,.0f}M"
)

col4.metric(
    "Revenue",
    f"${df['Revenue (M USD)'].sum():,.0f}M"
)

col5.metric(
    "Avg Valuation",
    f"${df['Valuation (M USD)'].mean():,.0f}M"
)

st.divider()

# =====================================================
# REGIONAL SUMMARY
# =====================================================

regional_summary = (

    df.groupby("Region")
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

st.subheader("💰 Regional Funding Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        regional_summary,
        x="Region",
        y="Funding Amount (M USD)",
        color="Region",
        title="Funding by Region"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        regional_summary,
        names="Region",
        values="Funding Amount (M USD)",
        title="Funding Share"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# REVENUE ANALYSIS
# =====================================================

st.subheader("📈 Regional Revenue Analysis")

fig = px.bar(
    regional_summary,
    x="Region",
    y="Revenue (M USD)",
    color="Region",
    title="Revenue by Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# VALUATION ANALYSIS
# =====================================================

st.subheader("💎 Regional Valuation Analysis")

fig = px.bar(
    regional_summary,
    x="Region",
    y="Valuation (M USD)",
    color="Region",
    title="Average Valuation by Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# STARTUP DENSITY
# =====================================================

st.subheader("🏢 Startup Density")

startup_count = (

    df.groupby("Region")
    .size()
    .reset_index(name="Startup Count")

)

fig = px.bar(
    startup_count,
    x="Region",
    y="Startup Count",
    color="Region",
    title="Startup Count by Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# GROWTH MATRIX
# =====================================================

st.subheader("🚀 Regional Growth Matrix")

fig = px.scatter(
    regional_summary,
    x="Funding Amount (M USD)",
    y="Revenue (M USD)",
    size="Valuation (M USD)",
    color="Region",
    hover_name="Region",
    title="Funding vs Revenue vs Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# MARKET SHARE ANALYSIS
# =====================================================

st.subheader("🌍 Regional Market Share")

market_share = (

    df.groupby("Region")
    ["Market Share (%)"]
    .mean()
    .reset_index()

)

fig = px.treemap(
    market_share,
    path=["Region"],
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

st.subheader("⚡ Regional Funding Efficiency")

df["Funding Efficiency"] = (

    df["Revenue (M USD)"]
    /
    df["Funding Amount (M USD)"]

)

efficiency = (

    df.groupby("Region")
    ["Funding Efficiency"]
    .mean()
    .reset_index()

)

fig = px.bar(
    efficiency,
    x="Region",
    y="Funding Efficiency",
    color="Region"
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

employee_efficiency = (

    df.groupby("Region")
    ["Revenue Per Employee"]
    .mean()
    .reset_index()

)

fig = px.bar(
    employee_efficiency,
    x="Region",
    y="Revenue Per Employee",
    color="Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# PROFITABILITY ANALYSIS
# =====================================================

if "Profitable" in df.columns:

    st.subheader("📊 Profitability by Region")

    profitability = (

        df.groupby("Region")
        ["Profitable"]
        .mean()
        .reset_index()

    )

    fig = px.bar(
        profitability,
        x="Region",
        y="Profitable",
        color="Region"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# UNICORN ANALYSIS
# =====================================================

st.subheader("🦄 Regional Unicorn Analysis")

df["Unicorn"] = (
    df["Valuation (M USD)"] >= 1000
)

unicorns = (

    df.groupby("Region")
    ["Unicorn"]
    .sum()
    .reset_index()

)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        unicorns,
        x="Region",
        y="Unicorn",
        color="Region",
        title="Unicorn Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        unicorns,
        names="Region",
        values="Unicorn",
        title="Regional Unicorn Share"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# HEALTH SCORE
# =====================================================

st.subheader("❤️ Regional Health Score")

health_df = calculate_health_score(df)

regional_health = (

    health_df.groupby("Region")
    ["Health Score"]
    .mean()
    .reset_index()

)

fig = px.bar(
    regional_health,
    x="Region",
    y="Health Score",
    color="Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RISK ANALYSIS
# =====================================================

st.subheader("🚨 Regional Risk Analysis")

risk_df = calculate_risk_score(df)

regional_risk = (

    risk_df.groupby("Region")
    ["Risk Score"]
    .mean()
    .reset_index()

)

fig = px.bar(
    regional_risk,
    x="Region",
    y="Risk Score",
    color="Region"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# REGIONAL BENCHMARKING
# =====================================================

st.subheader("📋 Regional Benchmarking")

benchmark = (

    df.groupby("Region")
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
# TOP STARTUPS BY REGION
# =====================================================

st.subheader("🏆 Regional Startup Leaderboard")

top_startups = (

    df.sort_values(
        "Valuation (M USD)",
        ascending=False
    )

    .head(25)

)

st.dataframe(
    top_startups[
        [
            "Startup Name",
            "Region",
            "Industry",
            "Valuation (M USD)",
            "Revenue (M USD)"
        ]
    ],
    use_container_width=True
)

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

st.subheader("🔥 Regional Correlation Analysis")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

# =====================================================
# ML INSIGHTS
# =====================================================

st.subheader("🤖 Machine Learning Insights")

model_result = train_valuation_model(df)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Valuation Prediction R²",
        round(model_result["score"], 3)
    )

with col2:

    importance = valuation_feature_importance(df)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Valuation Drivers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# AI REGIONAL INSIGHTS
# =====================================================

st.subheader("🧠 AI Regional Intelligence")

top_region_revenue = (
    df.groupby("Region")
    ["Revenue (M USD)"]
    .sum()
    .idxmax()
)

top_region_funding = (
    df.groupby("Region")
    ["Funding Amount (M USD)"]
    .sum()
    .idxmax()
)

top_region_valuation = (
    df.groupby("Region")
    ["Valuation (M USD)"]
    .mean()
    .idxmax()
)

st.success(f"""
REGIONAL INTELLIGENCE REPORT

• Leading Revenue Region:
{top_region_revenue}

• Leading Funding Region:
{top_region_funding}

• Highest Valuation Region:
{top_region_valuation}

• Regions Analyzed:
{df['Region'].nunique()}

• Total Startups:
{len(df)}

• Total Funding:
${df['Funding Amount (M USD)'].sum():,.2f} M

Recommendation:

Prioritize regions demonstrating
high funding efficiency, strong
valuation growth, and sustainable
startup ecosystem development.
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

report = f"""
Regional Intelligence Report

Top Revenue Region:
{top_region_revenue}

Top Funding Region:
{top_region_funding}

Top Valuation Region:
{top_region_valuation}

Regions Analyzed:
{df['Region'].nunique()}

Total Startups:
{len(df)}
"""

st.download_button(
    label="📥 Download Regional Report",
    data=report,
    file_name="Regional_Analysis_Report.txt",
    mime="text/plain"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Regional Intelligence Dashboard | Powered by AI + ML + Streamlit"
)
