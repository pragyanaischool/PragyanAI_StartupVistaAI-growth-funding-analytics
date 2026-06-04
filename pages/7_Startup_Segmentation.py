import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data
from utils.insights import *
from utils.ml_models import *
from utils.charts import *

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Startup Segmentation",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("🎯 Startup Segmentation Intelligence")

st.markdown("""
Discover startup segments, growth patterns,
investment opportunities and strategic clusters.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Segmentation Settings")

num_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    8,
    4
)

industry_filter = st.sidebar.multiselect(
    "Industry",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

df = df[
    df["Industry"].isin(industry_filter)
]

# =====================================================
# PREPARE DATA
# =====================================================

features = [
    "Funding Amount (M USD)",
    "Revenue (M USD)",
    "Valuation (M USD)",
    "Employees",
    "Market Share (%)"
]

seg_df = df.copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    seg_df[features]
)

# =====================================================
# KMEANS
# =====================================================

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42,
    n_init=10
)

seg_df["Cluster"] = (
    kmeans.fit_predict(X_scaled)
)

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("📊 Segmentation Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Startups",
    len(seg_df)
)

col2.metric(
    "Clusters",
    num_clusters
)

col3.metric(
    "Industries",
    seg_df["Industry"].nunique()
)

col4.metric(
    "Regions",
    seg_df["Region"].nunique()
)

st.divider()

# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================

st.subheader("📈 Cluster Distribution")

cluster_counts = (

    seg_df.groupby("Cluster")
    .size()
    .reset_index(name="Count")

)

fig = px.pie(
    cluster_counts,
    names="Cluster",
    values="Count",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# 2D CLUSTER VISUALIZATION
# =====================================================

st.subheader("🎯 Cluster Visualization")

fig = px.scatter(
    seg_df,
    x="Funding Amount (M USD)",
    y="Valuation (M USD)",
    color="Cluster",
    size="Revenue (M USD)",
    hover_name="Startup Name",
    title="Funding vs Valuation Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# 3D CLUSTER VISUALIZATION
# =====================================================

st.subheader("🚀 3D Startup Segmentation")

fig = px.scatter_3d(
    seg_df,
    x="Funding Amount (M USD)",
    y="Revenue (M USD)",
    z="Valuation (M USD)",
    color="Cluster",
    size="Employees",
    hover_name="Startup Name"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CLUSTER PROFILES
# =====================================================

st.subheader("📋 Cluster Profiles")

cluster_profile = (

    seg_df.groupby("Cluster")
    .agg({

        "Funding Amount (M USD)": "mean",
        "Revenue (M USD)": "mean",
        "Valuation (M USD)": "mean",
        "Employees": "mean",
        "Market Share (%)": "mean"

    })

)

st.dataframe(
    cluster_profile,
    use_container_width=True
)

# =====================================================
# CLUSTER HEATMAP
# =====================================================

st.subheader("🔥 Cluster Heatmap")

heatmap_data = cluster_profile.copy()

fig = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    title="Cluster Benchmarking"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CLUSTER LABELS
# =====================================================

cluster_labels = {}

sorted_clusters = (

    cluster_profile
    .sort_values(
        "Valuation (M USD)"
    )

)

cluster_ids = list(
    sorted_clusters.index
)

if len(cluster_ids) >= 4:

    cluster_labels[cluster_ids[0]] = "Emerging"
    cluster_labels[cluster_ids[1]] = "Growing"
    cluster_labels[cluster_ids[2]] = "Scale-Up"
    cluster_labels[cluster_ids[3]] = "Market Leader"

for cluster in seg_df["Cluster"].unique():

    if cluster not in cluster_labels:
        cluster_labels[cluster] = "Growth Segment"

seg_df["Segment"] = (
    seg_df["Cluster"]
    .map(cluster_labels)
)

# =====================================================
# SEGMENT DISTRIBUTION
# =====================================================

st.subheader("🏆 Startup Segments")

segment_count = (

    seg_df.groupby("Segment")
    .size()
    .reset_index(name="Count")

)

fig = px.bar(
    segment_count,
    x="Segment",
    y="Count",
    color="Segment"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# SEGMENT TABLE
# =====================================================

st.subheader("📋 Startup Segment Listing")

st.dataframe(
    seg_df[
        [
            "Startup Name",
            "Industry",
            "Region",
            "Cluster",
            "Segment"
        ]
    ],
    use_container_width=True
)

# =====================================================
# UNICORN SEGMENTATION
# =====================================================

st.subheader("🦄 Unicorn Segmentation")

seg_df["Unicorn"] = (

    seg_df["Valuation (M USD)"]
    >=
    1000

)

unicorn_clusters = (

    seg_df.groupby("Cluster")
    ["Unicorn"]
    .sum()
    .reset_index()

)

fig = px.bar(
    unicorn_clusters,
    x="Cluster",
    y="Unicorn",
    color="Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# INVESTMENT OPPORTUNITIES
# =====================================================

st.subheader("💸 Investment Opportunity Segments")

investment_df = (
    investment_recommendations(seg_df)
)

st.dataframe(
    investment_df[
        [
            "Startup Name",
            "Industry",
            "Success Score",
            "Valuation (M USD)"
        ]
    ].head(20),
    use_container_width=True
)

# =====================================================
# RISK ANALYSIS
# =====================================================

st.subheader("🚨 Cluster Risk Analysis")

risk_df = calculate_risk_score(seg_df)

cluster_risk = (

    risk_df.groupby("Cluster")
    ["Risk Score"]
    .mean()
    .reset_index()

)

fig = px.bar(
    cluster_risk,
    x="Cluster",
    y="Risk Score",
    color="Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TOP STARTUPS BY CLUSTER
# =====================================================

st.subheader("🏅 Top Startups Per Cluster")

cluster_select = st.selectbox(
    "Select Cluster",
    sorted(seg_df["Cluster"].unique())
)

top_cluster = (

    seg_df[
        seg_df["Cluster"]
        ==
        cluster_select
    ]

    .sort_values(
        "Valuation (M USD)",
        ascending=False
    )

)

st.dataframe(
    top_cluster[
        [
            "Startup Name",
            "Industry",
            "Valuation (M USD)",
            "Revenue (M USD)"
        ]
    ].head(20),
    use_container_width=True
)

# =====================================================
# SIMILAR STARTUP FINDER
# =====================================================

st.subheader("🔍 Similar Startup Finder")

startup_name = st.selectbox(
    "Select Startup",
    sorted(seg_df["Startup Name"].unique())
)

if st.button("Find Similar Startups"):

    try:

        similar = find_similar_startups(
            seg_df,
            startup_name
        )

        st.dataframe(
            similar[
                [
                    "Startup Name",
                    "Industry",
                    "Valuation (M USD)",
                    "Revenue (M USD)"
                ]
            ],
            use_container_width=True
        )

    except:
        st.warning(
            "Unable to generate recommendations."
        )

# =====================================================
# SEGMENT BENCHMARKING
# =====================================================

st.subheader("📊 Segment Benchmarking")

benchmark = (

    seg_df.groupby("Segment")
    .agg({

        "Funding Amount (M USD)": "mean",
        "Revenue (M USD)": "mean",
        "Valuation (M USD)": "mean",
        "Employees": "mean"

    })

)

st.dataframe(
    benchmark,
    use_container_width=True
)

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🧠 AI Segment Intelligence")

largest_segment = (

    segment_count.sort_values(
        "Count",
        ascending=False
    )

    .iloc[0]["Segment"]

)

top_cluster = (

    cluster_profile[
        "Valuation (M USD)"
    ]

    .idxmax()

)

st.success(f"""
SEGMENTATION INTELLIGENCE REPORT

• Total Startups:
{len(seg_df)}

• Clusters Identified:
{num_clusters}

• Largest Segment:
{largest_segment}

• Highest Value Cluster:
Cluster {top_cluster}

• Unicorn Startups:
{seg_df['Unicorn'].sum()}

Recommendation:

Focus on startups in Growth,
Scale-Up and Market Leader
segments with high revenue
efficiency and market share.
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

report = f"""
Startup Segmentation Report

Clusters:
{num_clusters}

Largest Segment:
{largest_segment}

Highest Value Cluster:
{top_cluster}

Total Startups:
{len(seg_df)}
"""

st.download_button(
    "📥 Download Segmentation Report",
    data=report,
    file_name="Startup_Segmentation_Report.txt",
    mime="text/plain"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "StartupVista AI | Startup Segmentation Dashboard | Powered by AI + ML + Streamlit"
)
