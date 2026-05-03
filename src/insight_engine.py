import pandas as pd


DEFAULT_COLUMN_MAP = {
    "transaction_id": "transaction_id",
    "date": "date",
    "amount": "amount",
    # optional:
    "customer_id": "customer_id",
    "product": "product",
    "category": "category",
}


REQUIRED_FIELDS = ["transaction_id", "date", "amount"]


def standardize_transactions(df: pd.DataFrame, column_map: dict) -> tuple[pd.DataFrame, dict]:
    standardized = df.copy()

    for logical_col, actual_col in column_map.items():
        if actual_col in standardized.columns:
            standardized = standardized.rename(columns={actual_col: logical_col})

    missing_required = [col for col in REQUIRED_FIELDS if col not in standardized.columns]
    if missing_required:
        raise ValueError(
            f"Missing required fields after mapping: {missing_required}. "
            f"Available columns: {list(df.columns)}"
        )

    original_rows = len(standardized)

    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    standardized["amount"] = pd.to_numeric(standardized["amount"], errors="coerce")

    invalid_date_rows = int(standardized["date"].isna().sum())
    invalid_amount_rows = int(standardized["amount"].isna().sum())
    negative_amounts_exist = bool((standardized["amount"] < 0).any())

    standardized = standardized.dropna(subset=["date", "amount"])

    data_quality = {
        "original_rows": original_rows,
        "usable_rows": len(standardized),
        "dropped_rows": original_rows - len(standardized),
        "invalid_date_rows": invalid_date_rows,
        "invalid_amount_rows": invalid_amount_rows,
        "negative_amounts_exist": negative_amounts_exist,
    }

    return standardized, data_quality


def load_kpis(path: str, column_map: dict = None):
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP

    raw_df = pd.read_csv(path)
    df, data_quality = standardize_transactions(raw_df, column_map)

    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    agg_dict = {
        "total_revenue": ("amount", "sum"),
        "total_orders": ("transaction_id", "count"),
    }

    if "customer_id" in df.columns:
        agg_dict["unique_customers"] = ("customer_id", "nunique")

    weekly = (
        df.groupby("week_start")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("week_start")
    )

    if "unique_customers" not in weekly.columns:
        weekly["unique_customers"] = 0

    weekly["aov"] = weekly["total_revenue"] / weekly["total_orders"]

    return weekly, data_quality


def build_metric_summary(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        raise ValueError("Need at least 2 weeks of KPI data to compare week-over-week.")

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    metrics = ["total_revenue", "total_orders", "unique_customers", "aov"]

    summary = {
        "latest_week": latest["week_start"].strftime("%Y-%m-%d"),
        "previous_week": previous["week_start"].strftime("%Y-%m-%d"),
        "metrics": {},
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


def generate_rule_based_insights(summary: dict, data_quality: dict) -> list[str]:
    insights = []

    for metric, values in summary["metrics"].items():
        change = values["change_pct"]

        if change is None:
            continue

        if change <= -20:
            insights.append(f"{metric} dropped sharply by {change}% week-over-week.")
        elif change <= -10:
            insights.append(f"{metric} declined by {change}% week-over-week.")
        elif change >= 20:
            insights.append(f"{metric} increased sharply by {change}% week-over-week.")
        elif change >= 10:
            insights.append(f"{metric} improved by {change}% week-over-week.")

    rev = summary["metrics"]["total_revenue"]["change_pct"]
    orders = summary["metrics"]["total_orders"]["change_pct"]
    aov = summary["metrics"]["aov"]["change_pct"]
    customers = summary["metrics"]["unique_customers"]["change_pct"]

    if rev is not None and aov is not None and rev > 20 and aov > 20 and (orders is None or abs(orders) < 5):
        insights.append(
            "Revenue growth appears to be driven primarily by higher average order value rather than higher order volume."
        )

    if customers is not None and customers <= 0:
        insights.append(
            "Customer growth appears flat or negative, which may limit sustainability of revenue growth."
        )

    if data_quality.get("negative_amounts_exist"):
        insights.append(
            "Negative transactions detected, which may indicate refunds or adjustments impacting revenue."
        )

    if data_quality.get("invalid_date_rows", 0) > 0 or data_quality.get("invalid_amount_rows", 0) > 0:
        insights.append(
            "Data quality issues detected, including invalid dates or amounts, which may affect KPI reliability."
        )

    if data_quality.get("dropped_rows", 0) > 0:
        insights.append(
            f"{data_quality['dropped_rows']} rows were excluded because they could not be parsed into usable dates or amounts."
        )

    if not insights:
        insights.append("No major week-over-week movement detected.")

    return insights


def build_prompt(summary: dict, rule_insights: list[str], data_quality: dict) -> str:
    return f"""
You are a business analyst writing a concise KPI explanation.

Write:
1. A short executive summary
2. Key changes
3. Possible business interpretation
4. Recommended next actions

Keep it practical and business-facing.
Do not overclaim causes.
Use cautious language like "may indicate" or "could suggest."
Distinguish between growth driven by volume and growth driven by average order value.
If the dataset is small, frame findings as directional signals rather than firm conclusions.
If data quality issues exist, mention that the findings should be interpreted with caution.

KPI summary:
{summary}

Data quality context:
{data_quality}

Rule-based findings:
{rule_insights}
""".strip()