# OpsSheet

OpsSheet is a lightweight tool that converts KPI datasets into business-facing explanations and recommended actions.

It combines simple rule-based analysis with LLM-generated summaries to help interpret changes in metrics like revenue, orders, and customers.

## Features

- Week-over-week KPI comparison
- Rule-based change detection
- AI-generated business explanations
- Optional Streamlit interface

## Example Use Case

Given a KPI dataset, OpsSheet can generate insights like:

"Revenue decreased 22% week-over-week, driven by a drop in returning customers. This may indicate retention issues. Recommended action: review recent campaigns or pricing changes."

## Run

```bash
python src/generate_insights.py