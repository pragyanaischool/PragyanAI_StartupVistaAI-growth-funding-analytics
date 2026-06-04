import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# FUNDING ANALYTICS
# =====================================================

def funding_by_industry(df):

    funding = (
        df.groupby("Industry")["Funding Amount (M USD)"]
        .sum()
        .reset_index()
        .sort_values(
            "Funding Amount (M USD)",
            ascending=False
        )
    )

    fig = px.bar(
        funding,
        x="Industry",
        y="Funding Amount (M USD)",
        color="Funding Amount (M USD)",
        title="Funding by Industry"
    )

    return fig


# =====================================================
# VALUATION DISTRIBUTION
# =====================================================

def valuation_distribution(df):

    fig = px.histogram(
        df,
        x="Valuation (M USD)",
        nbins=30,
        title="Startup Valuation Distribution"
    )

    return fig


# =====================================================
# INDUSTRY PERFORMANCE
# =====================================================

def industry_performance(df):

    industry = (
        df.groupby("Industry")
        .agg({
            "Revenue (M USD)": "mean",
            "Valuation (M USD)": "mean",
            "Employees": "mean"
        })
        .reset_index()
    )

    fig = px.scatter(
        industry,
        x="Revenue (M USD)",
        y="Valuation (M USD)",
        size="Employees",
        color="Industry",
        title="Industry Performance Matrix"
    )

    return fig


# =====================================================
# REGIONAL FUNDING
# =====================================================

def regional_funding(df):

    region = (
        df.groupby("Region")
        ["Funding Amount (M USD)"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        region,
        names="Region",
        values="Funding Amount (M USD)",
        title="Regional Funding Share"
    )

    return fig


# =====================================================
# PROFITABILITY ANALYSIS
# =====================================================

def profitability_analysis(df):

    fig = px.box(
        df,
        x="Profitable",
        y="Revenue (M USD)",
        color="Profitable",
        title="Revenue vs Profitability"
    )

    return fig


# =====================================================
# FUNDING VS REVENUE
# =====================================================

def funding_vs_revenue(df):

    fig = px.scatter(
        df,
        x="Funding Amount (M USD)",
        y="Revenue (M USD)",
        size="Employees",
        color="Industry",
        hover_name="Startup Name",
        title="Funding vs Revenue"
    )

    return fig


# =====================================================
# REVENUE BY INDUSTRY
# =====================================================

def revenue_by_industry(df):

    revenue = (
        df.groupby("Industry")
        ["Revenue (M USD)"]
        .sum()
        .reset_index()
        .sort_values(
            "Revenue (M USD)",
            ascending=False
        )
    )

    fig = px.bar(
        revenue,
        x="Industry",
        y="Revenue (M USD)",
        color="Revenue (M USD)",
        title="Revenue by Industry"
    )

    return fig


# =====================================================
# MARKET SHARE ANALYSIS
# =====================================================

def market_share_chart(df):

    fig = px.treemap(
        df,
        path=[
            "Industry",
            "Region",
            "Startup Name"
        ],
        values="Market Share (%)",
        color="Market Share (%)",
        title="Market Share Distribution"
    )

    return fig


# =====================================================
# FUNDING EFFICIENCY
# =====================================================

def funding_efficiency_chart(df):

    fig = px.scatter(
        df,
        x="Funding Amount (M USD)",
        y="Funding Efficiency",
        size="Revenue (M USD)",
        color="Industry",
        hover_name="Startup Name",
        title="Funding Efficiency Analysis"
    )

    return fig


# =====================================================
# STARTUP AGE ANALYSIS
# =====================================================

def startup_age_chart(df):

    fig = px.histogram(
        df,
        x="Startup Age",
        color="Industry",
        nbins=20,
        title="Startup Age Distribution"
    )

    return fig


# =====================================================
# TOP 10 STARTUPS
# =====================================================

def top_startups(df):

    top = (
        df.sort_values(
            "Valuation (M USD)",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top,
        x="Startup Name",
        y="Valuation (M USD)",
        color="Industry",
        title="Top 10 Most Valuable Startups"
    )

    return fig


# =====================================================
# UNICORN ANALYSIS
# =====================================================

def unicorn_chart(df):

    unicorns = (
        df.groupby("Unicorn")
        .size()
        .reset_index(name="Count")
    )

    fig = px.pie(
        unicorns,
        names="Unicorn",
        values="Count",
        title="Unicorn Distribution"
    )

    return fig


# =====================================================
# EXIT STATUS ANALYSIS
# =====================================================

def exit_status_chart(df):

    exit_data = (
        df.groupby("Exit Status")
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        exit_data,
        x="Exit Status",
        y="Count",
        color="Exit Status",
        title="Exit Status Distribution"
    )

    return fig


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def correlation_heatmap(df):

    numeric_cols = [
        "Funding Amount (M USD)",
        "Funding Rounds",
        "Valuation (M USD)",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Startup Age"
    ]

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    return fig


# =====================================================
# RADAR CHART
# =====================================================

def industry_radar_chart(df):

    radar = (
        df.groupby("Industry")
        .agg({
            "Revenue (M USD)": "mean",
            "Valuation (M USD)": "mean",
            "Market Share (%)": "mean"
        })
        .reset_index()
    )

    first = radar.iloc[0]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                first["Revenue (M USD)"],
                first["Valuation (M USD)"],
                first["Market Share (%)"]
            ],
            theta=[
                "Revenue",
                "Valuation",
                "Market Share"
            ],
            fill='toself',
            name=first["Industry"]
        )
    )

    fig.update_layout(
        title="Industry Radar Analysis"
    )

    return fig


# =====================================================
# WATERFALL CHART
# =====================================================

def waterfall_funding(df):

    industry = (
        df.groupby("Industry")
        ["Funding Amount (M USD)"]
        .sum()
        .reset_index()
    )

    fig = go.Figure(
        go.Waterfall(
            x=industry["Industry"],
            y=industry["Funding Amount (M USD)"]
        )
    )

    fig.update_layout(
        title="Funding Contribution by Industry"
    )

    return fig


# =====================================================
# SANKEY DIAGRAM
# =====================================================

def sankey_funding_exit(df):

    labels = [
        "Funding",
        "Valuation",
        "IPO",
        "Acquired",
        "Private"
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels
                ),
                link=dict(
                    source=[0, 1, 1],
                    target=[1, 2, 3],
                    value=[100, 50, 30]
                )
            )
        ]
    )

    fig.update_layout(
        title="Funding → Valuation → Exit Flow"
    )

    return fig


# =====================================================
# FUNNEL ANALYSIS
# =====================================================

def startup_funnel(df):

    stages = [
        "Founded",
        "Funded",
        "Growing",
        "Profitable",
        "Exited"
    ]

    values = [
        len(df),
        len(df),
        int(len(df) * 0.8),
        int(len(df) * 0.5),
        int(len(df) * 0.2)
    ]

    fig = px.funnel(
        x=values,
        y=stages,
        title="Startup Lifecycle Funnel"
    )

    return fig


# =====================================================
# HEALTH SCORE
# =====================================================

def health_score_chart(df):

    top = (
        df.sort_values(
            "Health Score",
            ascending=False
        )
        .head(20)
    )

    fig = px.bar(
        top,
        x="Startup Name",
        y="Health Score",
        color="Industry",
        title="Top Startup Health Scores"
    )

    return fig


# =====================================================
# REVENUE PER EMPLOYEE
# =====================================================

def revenue_per_employee_chart(df):

    fig = px.scatter(
        df,
        x="Employees",
        y="Revenue Per Employee",
        color="Industry",
        size="Revenue (M USD)",
        hover_name="Startup Name",
        title="Revenue Per Employee"
    )

    return fig
