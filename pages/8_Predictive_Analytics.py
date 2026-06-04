import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.ml_models import *
from utils.insights import *
from utils.charts import *

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Predictive Analytics",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("🤖 AI Predictive Analytics Center")

st.markdown("""
Machine Learning powered startup forecasting,
valuation prediction, unicorn prediction and
investment intelligence.
""")

# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 Model Performance Overview")

valuation_model = train_valuation_model(df)

unicorn_model = train_unicorn_model(df)

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Startups",
    len(df)
)

col2.metric(
    "Valuation Model R²",
    round(
        valuation_model["score"],
        3
    )
)

col3.metric(
    "Unicorn Accuracy",
    round(
        unicorn_model["accuracy"],
        3
    )
)

col4.metric(
    "Industries",
    df["Industry"].nunique()
)

st.divider()

# ==========================================================
# VALUATION PREDICTOR
# ==========================================================

st.subheader("💎 Startup Valuation Predictor")

col1,col2,col3 = st.columns(3)

with col1:

    funding = st.number_input(
        "Funding Amount (M USD)",
        min_value=0.0,
        value=100.0
    )

    funding_rounds = st.number_input(
        "Funding Rounds",
        min_value=1,
        value=3
    )

with col2:

    revenue = st.number_input(
        "Revenue (M USD)",
        min_value=0.0,
        value=50.0
    )

    employees = st.number_input(
        "Employees",
        min_value=1,
        value=500
    )

with col3:

    market_share = st.number_input(
        "Market Share (%)",
        min_value=0.0,
        value=10.0
    )

    industry = st.selectbox(
        "Industry",
        sorted(df["Industry"].unique())
    )

region = st.selectbox(
    "Region",
    sorted(df["Region"].unique())
)

if st.button("Predict Valuation"):

    try:

        temp_df = df.copy()

        industry_encoded = (
            valuation_model["encoders"]
            ["Industry"]
            .transform([industry])[0]
        )

        region_encoded = (
            valuation_model["encoders"]
            ["Region"]
            .transform([region])[0]
        )

        features = [[

            funding,
            funding_rounds,
            revenue,
            employees,
            market_share,
            industry_encoded,
            region_encoded

        ]]

        predicted = (
            valuation_model["model"]
            .predict(features)[0]
        )

        st.success(
            f"Predicted Valuation: "
            f"${predicted:,.2f} M"
        )

    except Exception as e:

        st.error(str(e))

# ==========================================================
# UNICORN PREDICTION
# ==========================================================

st.subheader("🦄 Unicorn Prediction Engine")

if st.button("Predict Unicorn Potential"):

    try:

        industry_encoded = (
            valuation_model["encoders"]
            ["Industry"]
            .transform([industry])[0]
        )

        region_encoded = (
            valuation_model["encoders"]
            ["Region"]
            .transform([region])[0]
        )

        input_data = [[

            funding,
            funding_rounds,
            revenue,
            employees,
            market_share,
            industry_encoded,
            region_encoded

        ]]

        result = (
            unicorn_model["model"]
            .predict(input_data)[0]
        )

        probability = (
            unicorn_model["model"]
            .predict_proba(input_data)[0][1]
        )

        if result == 1:

            st.success(
                f"🦄 Potential Unicorn "
                f"({probability:.1%})"
            )

        else:

            st.warning(
                f"Currently not predicted "
                f"as Unicorn ({probability:.1%})"
            )

    except Exception as e:

        st.error(str(e))

# ==========================================================
# WHAT IF SIMULATOR
# ==========================================================

st.subheader("🎯 What-If Funding Simulator")

sim_funding = st.slider(
    "Adjust Funding",
    0,
    1000,
    100
)

sim_revenue = revenue

industry_encoded = (
    valuation_model["encoders"]
    ["Industry"]
    .transform([industry])[0]
)

region_encoded = (
    valuation_model["encoders"]
    ["Region"]
    .transform([region])[0]
)

scenario_input = [[

    sim_funding,
    funding_rounds,
    sim_revenue,
    employees,
    market_share,
    industry_encoded,
    region_encoded

]]

scenario_value = (
    valuation_model["model"]
    .predict(scenario_input)[0]
)

st.metric(
    "Projected Valuation",
    f"${scenario_value:,.2f} M"
)

# ==========================================================
# FUNDING REQUIREMENT ESTIMATOR
# ==========================================================

st.subheader("💰 Funding Requirement Estimator")

target_revenue = st.number_input(
    "Target Revenue (M USD)",
    min_value=1.0,
    value=500.0
)

required_funding = (
    estimate_required_funding(
        target_revenue
    )
)

st.info(
    f"Estimated Funding Required: "
    f"${required_funding:,.2f} M"
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.subheader("🔥 Valuation Drivers")

importance = valuation_feature_importance(df)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# SUCCESS SCORE ANALYSIS
# ==========================================================

st.subheader("🏆 Success Score Prediction")

success_df = calculate_success_score(df)

top_success = (

    success_df.sort_values(
        "Success Score",
        ascending=False
    )

)

st.dataframe(
    top_success[
        [
            "Startup Name",
            "Industry",
            "Success Score"
        ]
    ].head(20),
    use_container_width=True
)

# ==========================================================
# INVESTMENT RECOMMENDATIONS
# ==========================================================

st.subheader("💸 Investor Recommendation Engine")

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
# ANOMALY DETECTION
# ==========================================================

st.subheader("🚨 Startup Anomaly Detection")

anomalies = (
    detect_anomalous_startups(df)
)

if len(anomalies):

    st.dataframe(
        anomalies,
        use_container_width=True
    )

else:

    st.success(
        "No anomalies detected."
    )

# ==========================================================
# CLUSTER PREDICTIONS
# ==========================================================

st.subheader("🎯 Startup Clustering")

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
# EXECUTIVE ML REPORT
# ==========================================================

st.subheader("📋 Executive AI Report")

report = generate_ml_report(df)

st.metric(
    "Valuation Model Score",
    report["valuation_model_score"]
)

st.metric(
    "Unicorn Accuracy",
    report["unicorn_model_accuracy"]
)

st.dataframe(
    report["top_investment_opportunities"],
    use_container_width=True
)

# ==========================================================
# AI INSIGHTS
# ==========================================================

st.subheader("🧠 AI Prediction Insights")

top_company = (
    recommendations.iloc[0]["Startup Name"]
)

top_score = (
    recommendations.iloc[0]["Success Score"]
)

st.success(f"""
PREDICTIVE ANALYTICS REPORT

• Valuation Model R²:
{valuation_model['score']:.3f}

• Unicorn Prediction Accuracy:
{unicorn_model['accuracy']:.3f}

• Best Investment Opportunity:
{top_company}

• Success Score:
{top_score:.2f}

• Total Startups Analyzed:
{len(df)}

Recommendation:

Prioritize startups with
high success scores,
strong revenue growth,
healthy market share and
sustainable funding efficiency.
""")

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

download_report = f"""
StartupVista AI Predictive Analytics Report

Valuation Model Score:
{valuation_model['score']:.3f}

Unicorn Accuracy:
{unicorn_model['accuracy']:.3f}

Best Investment Opportunity:
{top_company}

Success Score:
{top_score:.2f}
"""

st.download_button(
    label="📥 Download Prediction Report",
    data=download_report,
    file_name="Predictive_Analytics_Report.txt",
    mime="text/plain"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Predictive Analytics Center | Powered by AI + Machine Learning + Streamlit"
)
