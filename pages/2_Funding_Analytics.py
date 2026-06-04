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
    page_title="Funding Analytics",
    page_icon="💰",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("💰 Startup Funding Analytics")
st.markdown("""
Comprehensive funding intelligence platform for:

- Venture Capital Firms
- Angel Investors
- Startup Accelerators
- Incubators
- Founders
- Corporate Investors
""")

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Funding Filters")

industry = st.sidebar.multiselect(
    "Industry",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

region = st.sidebar.multiselect(
    "Region",
    sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

df = df[
    (df["Industry"].isin(industry))
    &
    (df["Region"].isin(region))
]

# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 Funding Overview")

total_funding = df["Funding Amount (M USD)"].sum()

avg_funding = df["Funding Amount (M USD)"].mean()

max_funding = df["Funding Amount (M USD)"].max()

total_rounds = df["Funding Rounds"].sum()

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Funding",
    f"${total_funding:,.2f} M"
)

col2.metric(
    "Average Funding",
    f"${avg_funding:,.2f} M"
)

col3.metric(
    "Highest Funding",
    f"${max_funding:,.2f} M"
)

col4.metric(
    "Total Funding Rounds",
    f"{total_rounds:,}"
)

st.divider()

# ==========================================================
# FUNDING DISTRIBUTION
# ==========================================================

st.subheader("📈 Funding Distribution")

col1,col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="Funding Amount (M USD)",
        nbins=30,
        title="Funding Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        y="Funding Amount (M USD)",
        title="Funding Spread Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# INDUSTRY FUNDING
# ==========================================================

st.subheader("🏭 Industry Funding Analysis")

st.plotly_chart(
    funding_by_industry(df),
    use_container_width=True
)

industry_funding = (
    df.groupby("Industry")
    ["Funding Amount (M USD)"]
    .agg(["sum","mean","max"])
    .reset_index()
)

st.dataframe(
    industry_funding,
    use_container_width=True
)

# ==========================================================
# REGIONAL FUNDING
# ==========================================================

st.subheader("🌎 Regional Funding Analysis")

col1,col2 = st.columns(2)

with col1:

    st.plotly_chart(
        regional_funding(df),
        use_container_width=True
    )

with col2:

    region_funding = (
        df.groupby("Region")
        ["Funding Amount (M USD)"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        region_funding,
        x="Region",
        y="Funding Amount (M USD)",
        color="Region",
        title="Regional Funding"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FUNDING ROUNDS ANALYSIS
# ==========================================================

st.subheader("🔄 Funding Rounds Analysis")

fig = px.scatter(
    df,
    x="Funding Rounds",
    y="Funding Amount (M USD)",
    color="Industry",
    size="Revenue (M USD)",
    hover_name="Startup Name",
    title="Funding Rounds vs Funding Amount"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FUNDING EFFICIENCY
# ==========================================================

st.subheader("⚡ Funding Efficiency")

df["Funding Efficiency"] = (
    df["Revenue (M USD)"]
    /
    df["Funding Amount (M USD)"]
)

col1,col2 = st.columns(2)

with col1:

    st.plotly_chart(
        funding_efficiency_chart(df),
        use_container_width=True
    )

with col2:

    top_efficiency = (
        df.sort_values(
            "Funding Efficiency",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        top_efficiency[
            [
                "Startup Name",
                "Industry",
                "Funding Efficiency"
            ]
        ],
        use_container_width=True
    )

# ==========================================================
# TOP FUNDED STARTUPS
# ==========================================================

st.subheader("🏆 Top Funded Startups")

top_funded = (
    df.sort_values(
        "Funding Amount (M USD)",
        ascending=False
    )
    .head(20)
)

fig = px.bar(
    top_funded,
    x="Startup Name",
    y="Funding Amount (M USD)",
    color="Industry",
    title="Top Funded Startups"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# STARTUP AGE VS FUNDING
# ==========================================================

st.subheader("📅 Startup Age vs Funding")

fig = px.scatter(
    df,
    x="Startup Age",
    y="Funding Amount (M USD)",
    size="Revenue (M USD)",
    color="Industry",
    hover_name="Startup Name"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# INVESTMENT MATRIX
# ==========================================================

st.subheader("💸 Investor Opportunity Matrix")

fig = px.scatter(
    df,
    x="Funding Amount (M USD)",
    y="Valuation (M USD)",
    size="Revenue (M USD)",
    color="Industry",
    hover_name="Startup Name",
    title="Funding vs Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FUNDING RISK ANALYSIS
# ==========================================================

st.subheader("🚨 Funding Risk Analysis")

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

st.subheader("🔥 Funding Correlation Analysis")

numeric_cols = [

    "Funding Amount (M USD)",
    "Funding Rounds",
    "Revenue (M USD)",
    "Valuation (M USD)",
    "Employees",
    "Market Share (%)"

]

corr = df[numeric_cols].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Funding Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# ML INSIGHTS
# ==========================================================

st.subheader("🤖 Machine Learning Insights")

valuation_model = train_valuation_model(df)

col1,col2 = st.columns(2)

with col1:

    st.metric(
        "Valuation Prediction R²",
        round(
            valuation_model["score"],
            3
        )
    )

with col2:

    importance = valuation_feature_importance(df)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Funding Drivers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FUNDING ANOMALIES
# ==========================================================

st.subheader("🔍 Funding Anomaly Detection")

anomalies = detect_anomalous_startups(df)

if len(anomalies) > 0:

    st.warning(
        f"{len(anomalies)} anomalies detected."
    )

    st.dataframe(
        anomalies,
        use_container_width=True
    )

else:

    st.success(
        "No significant anomalies detected."
    )

# ==========================================================
# AI GENERATED INSIGHTS
# ==========================================================

st.subheader("🧠 AI Funding Insights")

top_industry = (
    df.groupby("Industry")
    ["Funding Amount (M USD)"]
    .sum()
    .idxmax()
)

top_region = (
    df.groupby("Region")
    ["Funding Amount (M USD)"]
    .sum()
    .idxmax()
)

highest_funded = highest_funded_startup(df)

st.success(f"""

Funding Intelligence Summary

• Total Funding Raised:
${df['Funding Amount (M USD)'].sum():,.2f} M

• Most Funded Industry:
{top_industry}

• Most Funded Region:
{top_region}

• Highest Funded Startup:
{highest_funded['Startup Name']}

• Funding Received:
${highest_funded['Funding Amount (M USD)']:,.2f} M

• Average Funding:
${df['Funding Amount (M USD)'].mean():,.2f} M

Strategic Recommendation:

Focus on industries demonstrating
strong valuation growth and superior
funding efficiency ratios.
""")

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

report_text = f"""
Funding Analytics Report

Total Funding:
${df['Funding Amount (M USD)'].sum():,.2f} M

Average Funding:
${df['Funding Amount (M USD)'].mean():,.2f} M

Top Industry:
{top_industry}

Top Region:
{top_region}
"""

st.download_button(
    label="📥 Download Funding Report",
    data=report_text,
    file_name="Funding_Analytics_Report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Funding Intelligence Dashboard | Powered by Streamlit + AI + Machine Learning"
)
