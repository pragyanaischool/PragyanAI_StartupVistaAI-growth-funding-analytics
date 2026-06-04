import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.charts import *
from utils.insights import *
from utils.ml_models import *

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Valuation Insights",
    page_icon="💎",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("💎 Startup Valuation Intelligence")
st.markdown("""
Analyze startup valuations, identify unicorns,
discover investment opportunities and understand
the drivers behind startup growth.
""")

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Valuation Filters")

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

# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 Valuation Overview")

total_valuation = df["Valuation (M USD)"].sum()

avg_valuation = df["Valuation (M USD)"].mean()

max_valuation = df["Valuation (M USD)"].max()

median_valuation = df["Valuation (M USD)"].median()

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Valuation",
    f"${total_valuation:,.2f} M"
)

col2.metric(
    "Average Valuation",
    f"${avg_valuation:,.2f} M"
)

col3.metric(
    "Highest Valuation",
    f"${max_valuation:,.2f} M"
)

col4.metric(
    "Median Valuation",
    f"${median_valuation:,.2f} M"
)

st.divider()

# ==========================================================
# VALUATION DISTRIBUTION
# ==========================================================

st.subheader("📈 Valuation Distribution")

col1,col2 = st.columns(2)

with col1:

    st.plotly_chart(
        valuation_distribution(df),
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        y="Valuation (M USD)",
        title="Valuation Spread"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# INDUSTRY ANALYSIS
# ==========================================================

st.subheader("🏭 Industry Valuation Analysis")

industry_val = (
    df.groupby("Industry")
    ["Valuation (M USD)"]
    .agg(["sum","mean","max"])
    .reset_index()
)

fig = px.bar(
    industry_val,
    x="Industry",
    y="mean",
    color="Industry",
    title="Average Industry Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    industry_val,
    use_container_width=True
)

# ==========================================================
# REGION ANALYSIS
# ==========================================================

st.subheader("🌎 Regional Valuation Analysis")

region_val = (
    df.groupby("Region")
    ["Valuation (M USD)"]
    .sum()
    .reset_index()
)

col1,col2 = st.columns(2)

with col1:

    fig = px.pie(
        region_val,
        values="Valuation (M USD)",
        names="Region",
        title="Valuation Share by Region"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.bar(
        region_val,
        x="Region",
        y="Valuation (M USD)",
        color="Region",
        title="Regional Valuation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# TOP VALUED STARTUPS
# ==========================================================

st.subheader("🏆 Top Valued Startups")

top_valued = (
    df.sort_values(
        "Valuation (M USD)",
        ascending=False
    )
    .head(20)
)

fig = px.bar(
    top_valued,
    x="Startup Name",
    y="Valuation (M USD)",
    color="Industry",
    title="Top Valued Startups"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    top_valued[
        [
            "Startup Name",
            "Industry",
            "Valuation (M USD)",
            "Revenue (M USD)"
        ]
    ],
    use_container_width=True
)

# ==========================================================
# VALUATION VS REVENUE
# ==========================================================

st.subheader("📊 Valuation vs Revenue")

fig = px.scatter(
    df,
    x="Revenue (M USD)",
    y="Valuation (M USD)",
    size="Employees",
    color="Industry",
    hover_name="Startup Name",
    title="Revenue vs Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# VALUATION VS FUNDING
# ==========================================================

st.subheader("💰 Funding vs Valuation")

fig = px.scatter(
    df,
    x="Funding Amount (M USD)",
    y="Valuation (M USD)",
    color="Industry",
    size="Revenue (M USD)",
    hover_name="Startup Name",
    title="Funding vs Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# VALUATION MULTIPLE
# ==========================================================

st.subheader("⚡ Valuation Multiple")

df["Valuation Multiple"] = (
    df["Valuation (M USD)"]
    /
    df["Revenue (M USD)"]
)

fig = px.scatter(
    df,
    x="Revenue (M USD)",
    y="Valuation Multiple",
    color="Industry",
    size="Valuation (M USD)",
    hover_name="Startup Name"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# UNICORN ANALYSIS
# ==========================================================

st.subheader("🦄 Unicorn Intelligence")

df["Unicorn"] = (
    df["Valuation (M USD)"] >= 1000
)

col1,col2 = st.columns(2)

with col1:

    st.plotly_chart(
        unicorn_chart(df),
        use_container_width=True
    )

with col2:

    unicorn_data = unicorn_analysis(df)

    st.metric(
        "Total Unicorns",
        unicorn_data["count"]
    )

    st.metric(
        "Unicorn %",
        f"{unicorn_data['percentage']}%"
    )

# ==========================================================
# POTENTIAL UNICORNS
# ==========================================================

st.subheader("🚀 Potential Unicorns")

potential = potential_unicorns(df)

st.dataframe(
    potential[
        [
            "Startup Name",
            "Industry",
            "Valuation (M USD)",
            "Revenue (M USD)"
        ]
    ],
    use_container_width=True
)

# ==========================================================
# HEALTH SCORE
# ==========================================================

st.subheader("❤️ Startup Health Score")

health_df = calculate_health_score(df)

st.plotly_chart(
    health_score_chart(health_df),
    use_container_width=True
)

# ==========================================================
# VALUATION PREDICTION MODEL
# ==========================================================

st.subheader("🤖 Valuation Prediction Model")

model_result = train_valuation_model(df)

col1,col2 = st.columns(2)

with col1:

    st.metric(
        "Model R² Score",
        round(model_result["score"],3)
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

# ==========================================================
# CLUSTER ANALYSIS
# ==========================================================

st.subheader("🎯 Startup Segmentation")

cluster_df = startup_segmentation(df)

fig = px.scatter(
    cluster_df,
    x="Funding Amount (M USD)",
    y="Valuation (M USD)",
    color="Cluster",
    size="Revenue (M USD)",
    hover_name="Startup Name",
    title="Valuation-Based Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RISK ANALYSIS
# ==========================================================

st.subheader("🚨 Valuation Risk Analysis")

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

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

st.subheader("🔥 Correlation Analysis")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

# ==========================================================
# INVESTMENT OPPORTUNITIES
# ==========================================================

st.subheader("💸 Top Investment Opportunities")

recommendations = investment_recommendations(df)

st.dataframe(
    recommendations[
        [
            "Startup Name",
            "Industry",
            "Success Score",
            "Valuation (M USD)"
        ]
    ].head(20),
    use_container_width=True
)

# ==========================================================
# AI VALUATION INSIGHTS
# ==========================================================

st.subheader("🧠 AI Valuation Insights")

top_industry = (
    df.groupby("Industry")
    ["Valuation (M USD)"]
    .mean()
    .idxmax()
)

top_region = (
    df.groupby("Region")
    ["Valuation (M USD)"]
    .mean()
    .idxmax()
)

highest = highest_valued_startup(df)

st.success(f"""
VALUATION INTELLIGENCE REPORT

• Total Valuation:
${df['Valuation (M USD)'].sum():,.2f} M

• Average Valuation:
${df['Valuation (M USD)'].mean():,.2f} M

• Highest Valued Startup:
{highest['Startup Name']}

• Valuation:
${highest['Valuation (M USD)']:,.2f} M

• Leading Industry:
{top_industry}

• Leading Region:
{top_region}

• Unicorn Count:
{len(df[df['Valuation (M USD)'] >= 1000])}

Recommendation:

Prioritize startups with strong revenue,
healthy valuation multiples and scalable
market-share growth potential.
""")

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

report = f"""
Valuation Intelligence Report

Total Valuation:
${df['Valuation (M USD)'].sum():,.2f} M

Average Valuation:
${df['Valuation (M USD)'].mean():,.2f} M

Top Industry:
{top_industry}

Top Region:
{top_region}
"""

st.download_button(
    label="📥 Download Valuation Report",
    data=report,
    file_name="Valuation_Report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Valuation Intelligence Dashboard | Powered by AI + ML + Streamlit"
)
