import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# ─────────────────────────────────────────────
# STEP 5 — Text Analytics: Sentiment Analysis
# ─────────────────────────────────────────────
# WHAT IS VADER?
# VADER (Valence Aware Dictionary and sEntiment Reasoner)
# is a rule-based NLP tool built specifically for news
# and social media text. It reads a sentence and returns
# 4 scores:
#   neg  → how negative (0.0 to 1.0)
#   neu  → how neutral  (0.0 to 1.0)
#   pos  → how positive (0.0 to 1.0)
#   compound → overall score (-1.0 = most negative,
#                              +1.0 = most positive)
#
# We use the compound score to label each headline as:
#   Positive  → compound >= 0.05
#   Negative  → compound <= -0.05
#   Neutral   → everything in between

CHARTS_DIR = "data/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

PALETTE = {
    "Positive": "#55A868",
    "Neutral":  "#4C72B0",
    "Negative": "#C44E52"
}


def load_clean_data():
    path = "data/processed/headlines_clean.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Clean data not found. Run preprocess.py first.")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} articles\n")
    return df


# ─────────────────────────────────────────────
# CORE — Score every headline
# ─────────────────────────────────────────────
def run_sentiment_analysis(df):
    """
    Applies VADER to every headline title and adds
    4 new columns: neg, neu, pos, compound, sentiment_label
    """
    print("  Running VADER on all headlines...")
    analyser = SentimentIntensityAnalyzer()

    # Score each title — returns dict like:
    # {'neg': 0.0, 'neu': 0.6, 'pos': 0.4, 'compound': 0.52}
    scores = df["title"].apply(analyser.polarity_scores)

    df["neg"]      = scores.apply(lambda x: x["neg"])
    df["neu"]      = scores.apply(lambda x: x["neu"])
    df["pos"]      = scores.apply(lambda x: x["pos"])
    df["compound"] = scores.apply(lambda x: x["compound"])

    # Label based on compound threshold (VADER standard)
    def label(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_label"] = df["compound"].apply(label)

    # Summary
    counts = df["sentiment_label"].value_counts()
    print(f"\n  Sentiment breakdown:")
    for label_name, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {label_name:<10}: {count:>4} articles ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────
# CHART 6 — Overall Sentiment Distribution (Pie)
# ─────────────────────────────────────────────
def chart_sentiment_pie(df):
    print("\n  [6] Sentiment distribution pie chart...")
    counts = df["sentiment_label"].value_counts()
    colors = [PALETTE[l] for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 13}
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Overall Headline Sentiment Distribution",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/06_sentiment_pie.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/06_sentiment_pie.png")


# ─────────────────────────────────────────────
# CHART 7 — Sentiment by Topic (Stacked Bar)
# ─────────────────────────────────────────────
def chart_sentiment_by_topic(df):
    print("  [7] Sentiment by topic stacked bar...")

    pivot = (
        df.groupby(["topic", "sentiment_label"])
          .size()
          .unstack(fill_value=0)
    )

    # Make sure all 3 columns exist even if one sentiment is 0
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Positive", "Neutral", "Negative"]]

    # Normalise to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    pivot_pct.plot(
        kind="bar", stacked=True, ax=ax,
        color=[PALETTE["Positive"], PALETTE["Neutral"], PALETTE["Negative"]],
        edgecolor="white", linewidth=0.5
    )
    ax.set_title("Sentiment Distribution by Topic (%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Percentage of Articles")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/07_sentiment_by_topic.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/07_sentiment_by_topic.png")


# ─────────────────────────────────────────────
# CHART 8 — Average Compound Score by Topic
# ─────────────────────────────────────────────
def chart_avg_compound_by_topic(df):
    print("  [8] Average compound score by topic...")

    avg = (df.groupby("topic")["compound"]
             .mean()
             .sort_values())

    colors = [PALETTE["Negative"] if v < 0 else PALETTE["Positive"]
              for v in avg.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(avg.index, avg.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, avg.values):
        ax.text(val + (0.003 if val >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10)

    ax.set_title("Average Sentiment Score by Topic",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Compound Score (−1 = most negative, +1 = most positive)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/08_avg_compound_by_topic.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/08_avg_compound_by_topic.png")


# ─────────────────────────────────────────────
# CHART 9 — Sentiment Over Time (Line)
# ─────────────────────────────────────────────
def chart_sentiment_over_time(df):
    print("  [9] Sentiment over time...")

    df["date"] = pd.to_datetime(df["date"])
    daily = (df.groupby("date")["compound"]
               .mean()
               .reset_index())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily["compound"],
            color="#4C72B0", linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.fill_between(daily["date"], daily["compound"],
                    where=daily["compound"] >= 0,
                    alpha=0.15, color=PALETTE["Positive"], label="Positive days")
    ax.fill_between(daily["date"], daily["compound"],
                    where=daily["compound"] < 0,
                    alpha=0.15, color=PALETTE["Negative"], label="Negative days")

    ax.set_title("Average News Sentiment Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Compound Score")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/09_sentiment_over_time.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/09_sentiment_over_time.png")


# ─────────────────────────────────────────────
# SAVE final dataset (used in Power BI)
# ─────────────────────────────────────────────
def save_final_dataset(df):
    path = "data/processed/headlines_with_sentiment.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved final dataset → {path}")
    print("  This file is what you import into Power BI / Tableau.")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Step 5 — Text Analytics (Sentiment Analysis)")
    print("=" * 50)

    df = load_clean_data()
    df = run_sentiment_analysis(df)

    chart_sentiment_pie(df)
    chart_sentiment_by_topic(df)
    chart_avg_compound_by_topic(df)
    chart_sentiment_over_time(df)

    save_final_dataset(df)

    print("\n  Sample headlines with scores:")
    cols = ["title", "topic", "compound", "sentiment_label"]
    print(df[cols].head(10).to_string(index=False))

    print("\n  Done! Move on to Step 6 — Power BI Dashboard.")
