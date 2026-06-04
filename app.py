import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="StartupVista AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.hero-box {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    padding: 2rem;
    border-radius: 15px;
    color: white;
}

.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
}

.footer {
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class='hero-box'>

<h1>🚀 StartupVista AI</h1>

<h3>
Startup Funding, Valuation, Growth &
Investment Intelligence Platform
</h3>

<p>
AI-Powered Analytics Platform for
Investors, Accelerators, Incubators,
Founders, Venture Capital Firms,
Universities and Innovation Hubs.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# KPI OVERVIEW
# =====================================================

st.subheader("📊 Ecosystem Snapshot")

col1, col2, col3, col4, col5 = st.columns(5)

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

# =====================================================
# PLATFORM OVERVIEW
# =====================================================

st.subheader("🌍 Platform Overview")

col1, col2 = st.columns(2)

with col1:

    industry_funding = (
        df.groupby("Industry")
        ["Funding Amount (M USD)"]
        .sum()
        .reset_index()
        .sort_values(
            "Funding Amount (M USD)",
            ascending=False
        )
    )

    fig = px.bar(
        industry_funding,
        x="Industry",
        y="Funding Amount (M USD)",
        color="Industry",
        title="Funding by Industry"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    region_funding = (
        df.groupby("Region")
        ["Funding Amount (M USD)"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        region_funding,
        names="Region",
        values="Funding Amount (M USD)",
        title="Funding Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# STARTUP ECOSYSTEM
# =====================================================

st.subheader("🚀 Startup Ecosystem Overview")

col1, col2 = st.columns(2)

with col1:

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

with col2:

    fig = px.histogram(
        df,
        x="Valuation (M USD)",
        nbins=30,
        title="Valuation Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PLATFORM MODULES
# =====================================================

st.subheader("🧩 Analytics Modules")

modules = pd.DataFrame({

    "Module": [

        "Executive Dashboard",
        "Funding Analytics",
        "Valuation Insights",
        "Industry Analysis",
        "Regional Analysis",
        "Profitability Analysis",
        "Startup Segmentation",
        "Predictive Analytics",
        "AI Insights",
        "Report Center"

    ],

    "Purpose": [

        "Executive KPI Monitoring",
        "Funding Intelligence",
        "Valuation Intelligence",
        "Industry Benchmarking",
        "Regional Benchmarking",
        "Profitability Insights",
        "Startup Clustering",
        "ML Predictions",
        "AI Boardroom Assistant",
        "Reporting & Exports"

    ]

})

st.dataframe(
    modules,
    use_container_width=True
)

# =====================================================
# DATASET PREVIEW
# =====================================================

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =====================================================
# TOP STARTUPS
# =====================================================

st.subheader("🏆 Top Valued Startups")

top_startups = (

    df.sort_values(
        "Valuation (M USD)",
        ascending=False
    )

    .head(15)

)

st.dataframe(

    top_startups[
        [
            "Startup Name",
            "Industry",
            "Region",
            "Funding Amount (M USD)",
            "Revenue (M USD)",
            "Valuation (M USD)"
        ]
    ],

    use_container_width=True

)

# =====================================================
# QUICK INSIGHTS
# =====================================================

st.subheader("🧠 Quick AI Insights")

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

highest_valuation = (
    df["Valuation (M USD)"].max()
)

st.success(f"""

StartupVista AI Summary

• Total Startups:
{len(df)}

• Total Funding:
${df['Funding Amount (M USD)'].sum():,.2f} M

• Total Revenue:
${df['Revenue (M USD)'].sum():,.2f} M

• Leading Industry:
{top_industry}

• Leading Region:
{top_region}

• Highest Startup Valuation:
${highest_valuation:,.2f} M

Recommendation:

Focus investments on industries
with strong revenue generation,
healthy valuation growth and
high funding efficiency.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🚀 StartupVista AI")

st.sidebar.success(
    "Select a page from the sidebar."
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Available Modules

📊 Executive Dashboard

💰 Funding Analytics

💎 Valuation Insights

🏭 Industry Analysis

🌎 Regional Analysis

📈 Profitability Analysis

🎯 Startup Segmentation

🤖 Predictive Analytics

🧠 AI Insights

📑 Report Center
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "Powered by Streamlit, Plotly, AI & Machine Learning"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>

© 2026 StartupVista AI

Startup Funding, Valuation &
Investment Intelligence Platform

Built using:

✅ Streamlit

✅ Plotly

✅ Machine Learning

✅ AI Analytics

</div>
""",
unsafe_allow_html=True
)
