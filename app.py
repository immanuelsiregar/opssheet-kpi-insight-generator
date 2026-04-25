import os
import pandas as pd
import streamlit as st
from openai import OpenAI

from src.insight_engine import (
    build_metric_summary,
    generate_rule_based_insights,
    build_prompt,
)

st.set_page_config(page_title="OpsSheet", layout="wide")

st.title("OpsSheet")
st.caption("KPI explanation generator for business reporting")

uploaded_file = st.file_uploader("Upload KPI CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    st.subheader("KPI Preview")
    st.dataframe(df.tail(10), use_container_width=True)

    summary = build_metric_summary(df)
    rule_insights = generate_rule_based_insights(summary)

    st.subheader("Metric Summary")
    st.json(summary)

    st.subheader("Rule-Based Findings")
    for item in rule_insights:
        st.write(f"- {item}")

    if st.button("Generate AI Explanation"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error("OPENAI_API_KEY not set.")
        else:
            client = OpenAI(api_key=api_key)
            prompt = build_prompt(summary, rule_insights)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You write practical business KPI insights."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            st.subheader("AI Business Explanation")
            st.write(response.choices[0].message.content.strip())
else:
    st.info("Upload a KPI CSV to begin.")