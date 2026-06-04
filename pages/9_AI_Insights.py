import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

from utils.data_loader import load_data
from utils.insights import *
from utils.ml_models import *
from utils.charts import *

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Insights",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("🧠 StartupVista AI Boardroom Copilot")

st.markdown("""
AI-powered executive intelligence for:

- Investors
- Venture Capital Firms
- Startup Accelerators
- Founders
- Board Members
- Corporate Innovation Teams
""")

# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 Executive Snapshot")

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric(
    "Startups",
    len(df)
)

col2.metric(
    "Industries",
    df["Industry"].nunique()
)

col3.metric(
    "Regions",
    df["Region"].nunique()
)

col4.metric(
    "Funding",
    f"${df['Funding Amount (M USD)'].sum():,.0f}M"
)

col5.metric(
    "Revenue",
    f"${df['Revenue (M USD)'].sum():,.0f}M"
)

st.divider()

# ==========================================================
# EXECUTIVE AI SUMMARY
# ==========================================================

st.subheader("🤖 AI Executive Summary")

summary = generate_ai_summary(df)

st.success(summary)

# ==========================================================
# TOP BUSINESS DRIVERS
# ==========================================================

st.subheader("🔥 Key Business Drivers")

importance = valuation_feature_importance(df)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Factors Driving Startup Valuation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# INDUSTRY INTELLIGENCE
# ==========================================================

st.subheader("🏭 AI Industry Intelligence")

industry_revenue = (

    df.groupby("Industry")
    ["Revenue (M USD)"]
    .sum()
    .sort_values(
        ascending=False
    )

)

industry_funding = (

    df.groupby("Industry")
    ["Funding Amount (M USD)"]
    .sum()
    .sort_values(
        ascending=False
    )

)

top_industry = industry_revenue.index[0]

st.info(f"""
Industry Insights

• Highest Revenue Industry:
{top_industry}

• Revenue:
${industry_revenue.iloc[0]:,.2f} M

• Highest Funded Industry:
{industry_funding.index[0]}

Recommendation:

Focus future investments in
industries demonstrating both
strong revenue generation and
sustainable valuation growth.
""")

# ==========================================================
# REGIONAL INTELLIGENCE
# ==========================================================

st.subheader("🌎 AI Regional Intelligence")

regional = (

    df.groupby("Region")
    .agg({

        "Revenue (M USD)": "sum",
        "Funding Amount (M USD)": "sum",
        "Valuation (M USD)": "mean"

    })

)

top_region = (
    regional["Revenue (M USD)"]
    .idxmax()
)

st.success(f"""
Regional Intelligence

Leading Region:
{top_region}

Revenue:
${regional.loc[top_region,'Revenue (M USD)']:,.2f} M

Average Valuation:
${regional.loc[top_region,'Valuation (M USD)']:,.2f} M

Recommendation:

Expand startup scouting and
investment activity in leading
startup ecosystems.
""")

# ==========================================================
# STARTUP INTELLIGENCE
# ==========================================================

st.subheader("🚀 Startup Intelligence")

top_startup = highest_valued_startup(df)

st.metric(
    "Most Valuable Startup",
    top_startup["Startup Name"]
)

st.metric(
    "Valuation",
    f"${top_startup['Valuation (M USD)']:,.2f} M"
)

# ==========================================================
# UNICORN INTELLIGENCE
# ==========================================================

st.subheader("🦄 Unicorn Intelligence")

df["Unicorn"] = (
    df["Valuation (M USD)"] >= 1000
)

unicorn_count = (
    df["Unicorn"].sum()
)

col1,col2 = st.columns(2)

with col1:

    st.metric(
        "Total Unicorns",
        int(unicorn_count)
    )

with col2:

    st.metric(
        "Unicorn %",
        round(
            unicorn_count /
            len(df) * 100,
            2
        )
    )

st.plotly_chart(
    unicorn_chart(df),
    use_container_width=True
)

# ==========================================================
# INVESTOR RECOMMENDATIONS
# ==========================================================

st.subheader("💸 AI Investor Recommendations")

recommendations = (
    investment_recommendations(df)
)

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
# RISK INTELLIGENCE
# ==========================================================

st.subheader("🚨 Risk Intelligence")

risk_df = calculate_risk_score(df)

high_risk = (

    risk_df.sort_values(
        "Risk Score",
        ascending=False
    )

)

st.dataframe(
    high_risk[
        [
            "Startup Name",
            "Industry",
            "Risk Score"
        ]
    ].head(20),
    use_container_width=True
)

# ==========================================================
# SWOT ANALYSIS
# ==========================================================

st.subheader("📋 Automated SWOT Analysis")

strengths = [
    "Strong funding ecosystem",
    "Growing startup valuations",
    "High innovation capacity",
    "Expanding market opportunities"
]

weaknesses = [
    "Funding concentration",
    "Regional imbalance",
    "Valuation volatility"
]

opportunities = [
    "Emerging industries",
    "AI-driven businesses",
    "Global expansion",
    "Investor interest"
]

threats = [
    "Economic uncertainty",
    "Market saturation",
    "Competitive pressure"
]

col1,col2 = st.columns(2)

with col1:

    st.success("### Strengths")

    for item in strengths:
        st.write("✅", item)

    st.warning("### Weaknesses")

    for item in weaknesses:
        st.write("⚠️", item)

with col2:

    st.info("### Opportunities")

    for item in opportunities:
        st.write("🚀", item)

    st.error("### Threats")

    for item in threats:
        st.write("🔴", item)

# ==========================================================
# BOARDROOM ASSISTANT
# ==========================================================

st.subheader("🏛 Boardroom Assistant")

board_report = boardroom_report(df)

st.json(board_report)

# ==========================================================
# NATURAL LANGUAGE QUERY
# ==========================================================

st.subheader("💬 Ask the AI")

query = st.text_input(
    "Ask a business question"
)

if query:

    q = query.lower()

    if "top industry" in q:

        st.success(
            f"Top Industry: {top_industry}"
        )

    elif "top region" in q:

        st.success(
            f"Top Region: {top_region}"
        )

    elif "unicorn" in q:

        st.success(
            f"Total Unicorns: {unicorn_count}"
        )

    elif "funding" in q:

        st.success(
            f"Total Funding: "
            f"${df['Funding Amount (M USD)'].sum():,.2f} M"
        )

    else:

        st.info(
            "AI assistant could not find a direct answer."
        )

# ==========================================================
# ANOMALY DETECTION
# ==========================================================

st.subheader("🔍 AI Anomaly Detection")

anomalies = detect_anomalous_startups(df)

if len(anomalies):

    st.dataframe(
        anomalies,
        use_container_width=True
    )

else:

    st.success(
        "No major anomalies detected."
    )

# ==========================================================
# CLUSTER INSIGHTS
# ==========================================================

st.subheader("🎯 Startup Segment Insights")

cluster_df = startup_segmentation(df)

fig = px.scatter(
    cluster_df,
    x="Funding Amount (M USD)",
    y="Valuation (M USD)",
    color="Cluster",
    size="Revenue (M USD)",
    hover_name="Startup Name"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# STRATEGIC RECOMMENDATIONS
# ==========================================================

st.subheader("🎯 AI Strategic Recommendations")

st.success(f"""
STRATEGIC RECOMMENDATIONS

1. Increase investment allocation to
high-growth industries.

2. Prioritize startups with
high Success Scores.

3. Expand regional investment
coverage.

4. Focus on potential unicorns.

5. Monitor high-risk startups
for portfolio optimization.

6. Improve funding efficiency
across investment portfolios.

7. Encourage sustainable revenue
growth over valuation inflation.

8. Strengthen startup ecosystem
diversification.
""")

# ==========================================================
# EXECUTIVE REPORT
# ==========================================================

st.subheader("📄 Executive AI Report")

executive_report = f"""
STARTUPVISTA AI EXECUTIVE REPORT

Date:
{datetime.now()}

Total Startups:
{len(df)}

Total Funding:
${df['Funding Amount (M USD)'].sum():,.2f} M

Total Revenue:
${df['Revenue (M USD)'].sum():,.2f} M

Top Industry:
{top_industry}

Top Region:
{top_region}

Total Unicorns:
{unicorn_count}
"""

st.text_area(
    "Executive Report",
    executive_report,
    height=250
)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.download_button(
    label="📥 Download AI Report",
    data=executive_report,
    file_name="AI_Executive_Report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "StartupVista AI Boardroom Copilot | Powered by AI + ML + Streamlit"
)
