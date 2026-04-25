import pandas as pd


def load_kpis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start")


def build_metric_summary(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    metrics = ["total_revenue", "total_orders", "unique_customers", "aov"]

    summary = {
        "latest_week": latest["week_start"].strftime("%Y-%m-%d"),
        "previous_week": previous["week_start"].strftime("%Y-%m-%d"),
        "metrics": {}
    }

    for metric in metrics:
        current = latest[metric]
        prev = previous[metric]

        if prev == 0:
            change_pct = None
        else:
            change_pct = ((current - prev) / prev) * 100

        summary["metrics"][metric] = {
            "current": round(float(current), 2),
            "previous": round(float(prev), 2),
            "change_pct": None if change_pct is None else round(float(change_pct), 2),
        }

    return summary


def generate_rule_based_insights(summary: dict) -> list[str]:
    insights = []

    for metric, values in summary["metrics"].items():
        change = values["change_pct"]

        if change is None:
            continue

        if change <= -20:
            insights.append(
                f"{metric} dropped sharply by {change}% week-over-week."
            )
        elif change <= -10:
            insights.append(
                f"{metric} declined by {change}% week-over-week."
            )
        elif change >= 20:
            insights.append(
                f"{metric} increased sharply by {change}% week-over-week."
            )
        elif change >= 10:
            insights.append(
                f"{metric} improved by {change}% week-over-week."
            )

    if not insights:
        insights.append("No major week-over-week movement detected.")

    return insights


def build_prompt(summary: dict, rule_insights: list[str]) -> str:
    return f"""
You are a business analyst writing a concise KPI explanation.

Write:
1. A short executive summary
2. Key changes
3. Possible business interpretation
4. Recommended next actions

Keep it practical and business-facing. Do not overclaim causes. Use cautious language like "may indicate" or "could suggest."

KPI summary:
{summary}

Rule-based findings:
{rule_insights}
""".strip()