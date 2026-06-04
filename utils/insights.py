import pandas as pd
import numpy as np


# ==========================================================
# OVERALL BUSINESS SUMMARY
# ==========================================================

def generate_overall_summary(df):

    summary = {}

    summary["total_startups"] = len(df)

    summary["total_funding"] = round(
        df["Funding Amount (M USD)"].sum(), 2
    )

    summary["total_revenue"] = round(
        df["Revenue (M USD)"].sum(), 2
    )

    summary["avg_valuation"] = round(
        df["Valuation (M USD)"].mean(), 2
    )

    summary["avg_market_share"] = round(
        df["Market Share (%)"].mean(), 2
    )

    summary["avg_employees"] = round(
        df["Employees"].mean(), 0
    )

    return summary


# ==========================================================
# TOP PERFORMING INDUSTRY
# ==========================================================

def top_industry_by_valuation(df):

    result = (
        df.groupby("Industry")
        ["Valuation (M USD)"]
        .mean()
        .sort_values(ascending=False)
    )

    return result.head(1)


# ==========================================================
# TOP REVENUE INDUSTRY
# ==========================================================

def top_industry_by_revenue(df):

    result = (
        df.groupby("Industry")
        ["Revenue (M USD)"]
        .sum()
        .sort_values(ascending=False)
    )

    return result.head(1)


# ==========================================================
# BEST REGION
# ==========================================================

def best_region(df):

    result = (
        df.groupby("Region")
        ["Revenue (M USD)"]
        .sum()
        .sort_values(ascending=False)
    )

    return result.head(1)


# ==========================================================
# HIGHEST FUNDED STARTUP
# ==========================================================

def highest_funded_startup(df):

    row = df.loc[
        df["Funding Amount (M USD)"].idxmax()
    ]

    return row


# ==========================================================
# HIGHEST VALUED STARTUP
# ==========================================================

def highest_valued_startup(df):

    row = df.loc[
        df["Valuation (M USD)"].idxmax()
    ]

    return row


# ==========================================================
# MOST PROFITABLE STARTUP
# ==========================================================

def most_profitable_startup(df):

    row = df.loc[
        df["Revenue (M USD)"].idxmax()
    ]

    return row


# ==========================================================
# UNICORN ANALYSIS
# ==========================================================

def unicorn_analysis(df):

    unicorns = df[
        df["Valuation (M USD)"] >= 1000
    ]

    return {
        "count": len(unicorns),
        "percentage":
        round(
            len(unicorns) /
            len(df) * 100,
            2
        )
    }


# ==========================================================
# INDUSTRY RANKING
# ==========================================================

def industry_ranking(df):

    ranking = (
        df.groupby("Industry")
        .agg({
            "Valuation (M USD)": "mean",
            "Revenue (M USD)": "mean",
            "Funding Amount (M USD)": "mean"
        })
        .sort_values(
            "Valuation (M USD)",
            ascending=False
        )
    )

    return ranking


# ==========================================================
# FUNDING EFFICIENCY
# ==========================================================

def funding_efficiency(df):

    df["Funding Efficiency"] = (
        df["Revenue (M USD)"]
        /
        df["Funding Amount (M USD)"]
    )

    return (
        df.sort_values(
            "Funding Efficiency",
            ascending=False
        )
    )


# ==========================================================
# REVENUE PER EMPLOYEE
# ==========================================================

def revenue_per_employee(df):

    df["Revenue Per Employee"] = (
        df["Revenue (M USD)"]
        /
        df["Employees"]
    )

    return (
        df.sort_values(
            "Revenue Per Employee",
            ascending=False
        )
    )


# ==========================================================
# STARTUP HEALTH SCORE
# ==========================================================

def calculate_health_score(df):

    df["Health Score"] = (

        0.35 *
        (
            df["Revenue (M USD)"] /
            df["Revenue (M USD)"].max()
        )

        +

        0.35 *
        (
            df["Market Share (%)"] /
            df["Market Share (%)"].max()
        )

        +

        0.30 *
        (
            df["Valuation (M USD)"] /
            df["Valuation (M USD)"].max()
        )

    ) * 100

    return df


# ==========================================================
# TOP HEALTHY STARTUPS
# ==========================================================

def top_healthy_startups(df):

    df = calculate_health_score(df)

    return (
        df.sort_values(
            "Health Score",
            ascending=False
        )
        .head(20)
    )


# ==========================================================
# STARTUP RISK SCORE
# ==========================================================

def calculate_risk_score(df):

    df["Risk Score"] = (

        100

        -

        (
            (
                df["Revenue (M USD)"]
                /
                df["Revenue (M USD)"].max()
            ) * 40
        )

        -

        (
            (
                df["Valuation (M USD)"]
                /
                df["Valuation (M USD)"].max()
            ) * 30
        )

        -

        (
            (
                df["Market Share (%)"]
                /
                df["Market Share (%)"].max()
            ) * 30
        )

    )

    return df


# ==========================================================
# HIGH RISK STARTUPS
# ==========================================================

def high_risk_startups(df):

    df = calculate_risk_score(df)

    return (
        df.sort_values(
            "Risk Score",
            ascending=False
        )
        .head(20)
    )


# ==========================================================
# POTENTIAL UNICORNS
# ==========================================================

def potential_unicorns(df):

    future = df[
        (df["Valuation (M USD)"] > 500)
        &
        (df["Revenue (M USD)"] > 50)
    ]

    return future.sort_values(
        "Valuation (M USD)",
        ascending=False
    )


# ==========================================================
# INDUSTRY GROWTH ANALYSIS
# ==========================================================

def industry_growth_analysis(df):

    growth = (
        df.groupby("Industry")
        .agg({
            "Revenue (M USD)": "mean",
            "Funding Amount (M USD)": "mean",
            "Valuation (M USD)": "mean"
        })
    )

    return growth


# ==========================================================
# REGION PERFORMANCE
# ==========================================================

def regional_performance(df):

    region = (
        df.groupby("Region")
        .agg({
            "Revenue (M USD)": "sum",
            "Valuation (M USD)": "sum",
            "Funding Amount (M USD)": "sum"
        })
        .sort_values(
            "Revenue (M USD)",
            ascending=False
        )
    )

    return region


# ==========================================================
# EXIT STATUS ANALYSIS
# ==========================================================

def exit_status_analysis(df):

    result = (
        df.groupby("Exit Status")
        .agg({
            "Valuation (M USD)": "mean",
            "Revenue (M USD)": "mean"
        })
    )

    return result


# ==========================================================
# ANOMALY DETECTION
# ==========================================================

def detect_anomalies(df):

    valuation_mean = (
        df["Valuation (M USD)"].mean()
    )

    valuation_std = (
        df["Valuation (M USD)"].std()
    )

    anomalies = df[
        abs(
            df["Valuation (M USD)"]
            - valuation_mean
        )
        >
        3 * valuation_std
    ]

    return anomalies


# ==========================================================
# INVESTMENT ATTRACTIVENESS
# ==========================================================

def investment_score(df):

    df["Investment Score"] = (

        (
            df["Revenue (M USD)"]
            /
            df["Revenue (M USD)"].max()
        ) * 40

        +

        (
            df["Market Share (%)"]
            /
            df["Market Share (%)"].max()
        ) * 30

        +

        (
            df["Valuation (M USD)"]
            /
            df["Valuation (M USD)"].max()
        ) * 30

    )

    return (
        df.sort_values(
            "Investment Score",
            ascending=False
        )
    )


# ==========================================================
# INVESTOR RECOMMENDATION
# ==========================================================

def investor_recommendations(df):

    ranked = investment_score(df)

    return ranked.head(20)


# ==========================================================
# AI EXECUTIVE SUMMARY
# ==========================================================

def generate_ai_summary(df):

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

    unicorn_count = len(
        df[df["Valuation (M USD)"] >= 1000]
    )

    summary = f"""
    Executive Insights

    • Total Startups Analyzed : {len(df)}

    • Total Funding Raised :
      ${df['Funding Amount (M USD)'].sum():,.2f} M

    • Total Revenue :
      ${df['Revenue (M USD)'].sum():,.2f} M

    • Average Valuation :
      ${df['Valuation (M USD)'].mean():,.2f} M

    • Leading Industry :
      {top_industry}

    • Best Performing Region :
      {top_region}

    • Unicorn Startups :
      {unicorn_count}

    • Funding Efficiency :
      {round((df['Revenue (M USD)'].sum()/df['Funding Amount (M USD)'].sum()),2)}

    Recommendation:
    Focus investments on high-growth
    industries with strong revenue,
    market share, and funding efficiency.
    """

    return summary


# ==========================================================
# BOARD ROOM REPORT
# ==========================================================

def boardroom_report(df):

    report = {
        "summary":
        generate_overall_summary(df),

        "top_industry":
        top_industry_by_valuation(df),

        "best_region":
        best_region(df),

        "unicorns":
        unicorn_analysis(df),

        "highest_valued":
        highest_valued_startup(df),

        "highest_funded":
        highest_funded_startup(df)
    }

    return report
