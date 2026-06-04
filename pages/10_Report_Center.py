import streamlit as st
import pandas as pd
import json
from datetime import datetime
from io import BytesIO

from utils.data_loader import load_data
from utils.insights import *
from utils.ml_models import *

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Report Center",
    page_icon="📑",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("📑 StartupVista AI Report Center")

st.markdown("""
Generate Executive Reports, Industry Reports,
Regional Reports, Startup Reports and AI
Boardroom Reports.
""")

# =====================================================
# REPORT TYPE
# =====================================================

report_type = st.selectbox(

    "Select Report Type",

    [
        "Executive Report",
        "Industry Report",
        "Regional Report",
        "Startup Report",
        "AI Boardroom Report"
    ]

)

# =====================================================
# EXECUTIVE REPORT
# =====================================================

def executive_report():

    summary = generate_overall_summary(df)

    report = f"""
STARTUPVISTA EXECUTIVE REPORT

Generated:
{datetime.now()}

----------------------------------

Total Startups:
{summary['total_startups']}

Total Funding:
${summary['total_funding']:,.2f} M

Total Revenue:
${summary['total_revenue']:,.2f} M

Average Valuation:
${summary['avg_valuation']:,.2f} M

Average Market Share:
{summary['avg_market_share']:.2f}%

Average Employees:
{summary['avg_employees']}

----------------------------------
"""

    return report


# =====================================================
# INDUSTRY REPORT
# =====================================================

def industry_report():

    report = []

    industry_data = (

        df.groupby("Industry")
        .agg({

            "Funding Amount (M USD)": "sum",
            "Revenue (M USD)": "sum",
            "Valuation (M USD)": "mean"

        })

    )

    report.append(
        "INDUSTRY REPORT\n"
    )

    report.append(
        str(industry_data)
    )

    return "\n".join(report)


# =====================================================
# REGIONAL REPORT
# =====================================================

def regional_report():

    region_data = (

        df.groupby("Region")
        .agg({

            "Funding Amount (M USD)": "sum",
            "Revenue (M USD)": "sum",
            "Valuation (M USD)": "mean"

        })

    )

    return str(region_data)


# =====================================================
# STARTUP REPORT
# =====================================================

def startup_report():

    top = (

        df.sort_values(
            "Valuation (M USD)",
            ascending=False
        )

        .head(50)

    )

    return top.to_string()


# =====================================================
# AI BOARDROOM REPORT
# =====================================================

def ai_boardroom_report():

    summary = generate_ai_summary(df)

    report = f"""

AI BOARDROOM REPORT

Generated:
{datetime.now()}

----------------------------------

{summary}

----------------------------------

Top Industry:
{top_industry_by_revenue(df).index[0]}

Top Region:
{best_region(df).index[0]}

Highest Valued Startup:
{highest_valued_startup(df)['Startup Name']}

Highest Funded Startup:
{highest_funded_startup(df)['Startup Name']}

----------------------------------

Recommendation:

Prioritize investments in
high-growth startups with
strong revenue, market share,
and funding efficiency.
"""

    return report


# =====================================================
# GENERATE REPORT
# =====================================================

if report_type == "Executive Report":

    report_content = executive_report()

elif report_type == "Industry Report":

    report_content = industry_report()

elif report_type == "Regional Report":

    report_content = regional_report()

elif report_type == "Startup Report":

    report_content = startup_report()

else:

    report_content = ai_boardroom_report()

# =====================================================
# REPORT VIEWER
# =====================================================

st.subheader("📋 Report Preview")

st.text_area(
    "Report",
    report_content,
    height=400
)

# =====================================================
# KPI SNAPSHOT
# =====================================================

st.subheader("📊 KPI Snapshot")

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Funding",
    f"${df['Funding Amount (M USD)'].sum():,.0f}M"
)

col2.metric(
    "Revenue",
    f"${df['Revenue (M USD)'].sum():,.0f}M"
)

col3.metric(
    "Avg Valuation",
    f"${df['Valuation (M USD)'].mean():,.0f}M"
)

col4.metric(
    "Startups",
    len(df)
)

# =====================================================
# CSV EXPORT
# =====================================================

st.subheader("📁 Data Export")

csv_data = (
    df.to_csv(index=False)
)

st.download_button(
    label="📥 Download CSV",
    data=csv_data,
    file_name="startup_data.csv",
    mime="text/csv"
)

# =====================================================
# EXCEL EXPORT
# =====================================================

def to_excel(dataframe):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False
        )

    return output.getvalue()


excel_file = to_excel(df)

st.download_button(
    label="📥 Download Excel",
    data=excel_file,
    file_name="startup_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =====================================================
# JSON EXPORT
# =====================================================

json_data = (
    df.to_json(
        orient="records"
    )
)

st.download_button(
    label="📥 Download JSON",
    data=json_data,
    file_name="startup_report.json",
    mime="application/json"
)

# =====================================================
# REPORT EXPORT
# =====================================================

st.subheader("📄 Report Export")

st.download_button(
    label="📥 Download Text Report",
    data=report_content,
    file_name="executive_report.txt",
    mime="text/plain"
)

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🧠 AI Recommendations")

recommendations = investment_recommendations(df)

st.dataframe(

    recommendations[
        [
            "Startup Name",
            "Industry",
            "Success Score",
            "Valuation (M USD)"
        ]
    ]

    .head(20),

    use_container_width=True

)

# =====================================================
# BOARDROOM SNAPSHOT
# =====================================================

st.subheader("🏛 Boardroom Snapshot")

board_report = boardroom_report(df)

st.json(board_report)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

st.subheader("🤖 Executive AI Summary")

st.success(
    generate_ai_summary(df)
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "StartupVista AI Report Center | Executive Reporting Suite"
)
