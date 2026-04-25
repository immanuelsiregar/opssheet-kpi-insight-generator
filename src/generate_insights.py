import os
from openai import OpenAI

from insight_engine import (
    load_kpis,
    build_metric_summary,
    generate_rule_based_insights,
    build_prompt,
)

INPUT_PATH = "data/input/kpi_weekly_sales.csv"
OUTPUT_PATH = "output/insights.txt"


def main():
    os.makedirs("output", exist_ok=True)

    df = load_kpis(INPUT_PATH)
    summary = build_metric_summary(df)
    rule_insights = generate_rule_based_insights(summary)

    prompt = build_prompt(summary, rule_insights)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        insight_text = (
            "AI summary unavailable because OPENAI_API_KEY is not set.\n\n"
            "Rule-based insights:\n- " + "\n- ".join(rule_insights)
        )
    else:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write practical business KPI insights."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        insight_text = response.choices[0].message.content.strip()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(insight_text)

    print("Generated insights:")
    print(OUTPUT_PATH)
    print()
    print(insight_text)


if __name__ == "__main__":
    main()