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
    page_title="Executive Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color:#f7f9fc;
}

.metric-card {
    background-color:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

.big-font {
    font-size:28px !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("🚀 StartupVista AI")
st.subheader("Executive Intelligence Dashboard")

st.markdown("""
This dashboard provides deep insights into:

✔ Startup Ecosystem

✔ Funding Trends

✔ Valuation Analytics

✔ Revenue Intelligence

✔ Unicorn Detection

✔ Investment Opportunities

✔ Startup Health Assessment

✔ AI Generated Strategic Insights
""")

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Dashboard Filters")

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

summary = generate_overall_summary(df)

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    st.metric(
        "Total Startups",
        f"{summary['total_startups']:,}"
    )

with col2:
    st.metric(
        "Funding Raised",
        f"${summary['total_funding']:,.0f}M"
    )

with col3:
    st.metric(
        "Revenue",
        f"${summary['total_revenue']:,.0f}M"
    )

with col4:
    st.metric(
        "Avg Valuation",
        f"${summary['avg_valuation']:,.0f}M"
    )

with col5:
    st.metric(
        "Avg Market Share",
        f"{summary['avg_market_share']:.2f}%"
    )

st.divider()

# ==========================================================
# AI SUMMARY
# ==========================================================

st.subheader("🤖 AI Executive Summary")

summary_text = generate_ai_summary(df)

st.success(summary_text)

# ==========================================================
# TOP KPIs
# ==========================================================

col1,col2,col3 = st.columns(3)

highest_funding = highest_funded_startup(df)

highest_value = highest_valued_startup(df)

top_profit = most_profitable_startup(df)

with col1:
    st.info(
        f"""
        💰 Highest Funded Startup

        {highest_funding['Startup Name']}

        Funding:
        ${highest_funding['Funding Amount (M USD)']:,.2f} M
        """
    )

with col2:
    st.info(
        f"""
        🏆 Highest Valued Startup

        {highest_value['Startup Name']}

        Valuation:
        ${highest_value['Valuation (M USD)']:,.2f} M
        """
    )

with col3:
    st.info(
        f"""
        📈 Highest Revenue Startup

        {top_profit['Startup Name']}

        Revenue:
        ${top_profit['Revenue (M USD)']:,.2f} M
        """
    )

st.divider()

# ==========================================================
# FUNDING & REVENUE
# ==========================================================

st.subheader("💰 Funding Analytics")

col1,col2 = st.columns(2)

with col1:
    st.plotly_chart(
        funding_by_industry(df),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        regional_funding(df),
        use_container_width=True
    )

# ==========================================================
# REVENUE ANALYSIS
# ==========================================================

st.subheader("📊 Revenue Analytics")

col1,col2 = st.columns(2)

with col1:
    st.plotly_chart(
        revenue_by_industry(df),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        profitability_analysis(df),
        use_container_width=True
    )

# ==========================================================
# VALUATION ANALYTICS
# ==========================================================

st.subheader("💎 Valuation Analytics")

col1,col2 = st.columns(2)

with col1:
    st.plotly_chart(
        valuation_distribution(df),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        industry_performance(df),
        use_container_width=True
    )

# ==========================================================
# FUNDING EFFICIENCY
# ==========================================================

st.subheader("⚡ Funding Efficiency")

if "Funding Efficiency" not in df.columns:

    df["Funding Efficiency"] = (
        df["Revenue (M USD)"]
        /
        df["Funding Amount (M USD)"]
    )

st.plotly_chart(
    funding_efficiency_chart(df),
    use_container_width=True
)

# ==========================================================
# HEALTH SCORE
# ==========================================================

st.subheader("❤️ Startup Health Analysis")

df_health = calculate_health_score(df)

st.plotly_chart(
    health_score_chart(df_health),
    use_container_width=True
)

# ==========================================================
# CORRELATION
# ==========================================================

st.subheader("🔥 Correlation Heatmap")

st.plotly_chart(
    correlation_heatmap(df),
    use_container_width=True
)

# ==========================================================
# TOP STARTUPS
# ==========================================================

st.subheader("🏆 Top 20 Startup Rankings")

top20 = top_healthy_startups(df)

st.dataframe(
    top20[
        [
            "Startup Name",
            "Industry",
            "Region",
            "Health Score",
            "Valuation (M USD)",
            "Revenue (M USD)"
        ]
    ],
    use_container_width=True
)

# ==========================================================
# UNICORN ANALYSIS
# ==========================================================

st.subheader("🦄 Unicorn Analysis")

if "Unicorn" not in df.columns:
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

    unicorn_info = unicorn_analysis(df)

    st.metric(
        "Total Unicorns",
        unicorn_info["count"]
    )

    st.metric(
        "Unicorn %",
        f"{unicorn_info['percentage']}%"
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
            "Revenue (M USD)",
            "Valuation (M USD)"
        ]
    ].head(20),
    use_container_width=True
)

# ==========================================================
# ANOMALIES
# ==========================================================

st.subheader("🚨 Startup Anomaly Detection")

anomalies = detect_anomalous_startups(df)

st.dataframe(
    anomalies,
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
    title="Startup Segmentation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# REGIONAL PERFORMANCE
# ==========================================================

st.subheader("🌎 Regional Performance")

region_perf = regional_performance(df)

st.dataframe(
    region_perf,
    use_container_width=True
)

# ==========================================================
# EXECUTIVE REPORT
# ==========================================================

st.subheader("📋 Executive Report")

board_report = boardroom_report(df)

st.json(board_report["summary"])

# ==========================================================
# DOWNLOAD SUMMARY
# ==========================================================

report_text = generate_ai_summary(df)

st.download_button(
    label="📥 Download Executive Summary",
    data=report_text,
    file_name="Executive_Report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Executive Dashboard | Powered by Streamlit + AI + ML"
)
