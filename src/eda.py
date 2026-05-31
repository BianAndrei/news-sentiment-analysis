import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
import os

# ─────────────────────────────────────────────
# STEP 3 — Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────
# Goal: produce 5 charts that uncover patterns
# in the headlines. These go into your dashboard
# and your presentation as "insights".

# Output folder for all chart images
CHARTS_DIR = "data/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Consistent colour palette across all charts
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

def load_clean_data():
    path = "data/processed/headlines_clean.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Clean data not found. Run preprocess.py first.")
    df = pd.read_csv(path, parse_dates=["publishedAt", "date"])
    print(f"  Loaded {len(df)} clean articles\n")
    return df


# ─────────────────────────────────────────────
# CHART 1 — Articles per Topic (Bar Chart)
# ─────────────────────────────────────────────
def chart_articles_per_topic(df):
    print("  [1] Articles per topic...")
    counts = df["topic"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE)

    # Add value labels on top of each bar
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=11)

    ax.set_title("Number of Articles per Topic", fontsize=14, fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Article Count")
    ax.set_ylim(0, counts.max() + 30)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/01_articles_per_topic.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/01_articles_per_topic.png")


# ─────────────────────────────────────────────
# CHART 2 — Top 10 News Sources (Horizontal Bar)
# ─────────────────────────────────────────────
def chart_top_sources(df):
    print("  [2] Top 10 sources...")
    top = df["source_name"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0")

    ax.set_title("Top 10 News Sources by Article Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Articles")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/02_top_sources.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/02_top_sources.png")


# ─────────────────────────────────────────────
# CHART 3 — Articles Over Time (Line Chart)
# ─────────────────────────────────────────────
def chart_articles_over_time(df):
    print("  [3] Articles over time...")

    # Count articles published per day
    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["count"],
            color="#4C72B0", linewidth=2, marker="o", markersize=4)
    ax.fill_between(daily["date"], daily["count"], alpha=0.15, color="#4C72B0")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    ax.set_title("Articles Published Per Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Articles")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/03_articles_over_time.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/03_articles_over_time.png")


# ─────────────────────────────────────────────
# CHART 4 — Publishing Hour Heatmap
# ─────────────────────────────────────────────
def chart_publishing_hours(df):
    print("  [4] Publishing hours heatmap...")

    # Count articles by topic and hour
    pivot = df.groupby(["topic", "hour"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.4,
                annot=True, fmt="d", ax=ax)

    ax.set_title("Articles by Topic and Publishing Hour (UTC)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/04_publishing_heatmap.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/04_publishing_heatmap.png")


# ─────────────────────────────────────────────
# CHART 5 — Word Cloud of All Headlines
# ─────────────────────────────────────────────
def chart_wordcloud(df):
    print("  [5] Word cloud...")

    # Join all titles into one big string
    text = " ".join(df["title"].dropna().tolist())

    # Words to exclude — too common, add no meaning
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "it", "its", "as", "by",
        "this", "that", "was", "are", "be", "has", "have", "been",
        "from", "will", "not", "new", "says", "after", "over",
        "what", "how", "why", "who", "can", "about", "up", "more"
    }

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        stopwords=stopwords,
        colormap="Blues",
        max_words=100,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in Headlines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/05_wordcloud.png", dpi=150)
    plt.close()
    print(f"     Saved → {CHARTS_DIR}/05_wordcloud.png")


# ─────────────────────────────────────────────
# RUN ALL CHARTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Step 3 — Exploratory Data Analysis")
    print("=" * 50)

    df = load_clean_data()

    chart_articles_per_topic(df)
    chart_top_sources(df)
    chart_articles_over_time(df)
    chart_publishing_hours(df)
    chart_wordcloud(df)

    print(f"\n  All 5 charts saved to '{CHARTS_DIR}/'")
    print("\n  Key stats:")
    print(f"    Total articles   : {len(df)}")
    print(f"    Unique sources   : {df['source_name'].nunique()}")
    print(f"    Date range       : {df['date'].min()} → {df['date'].max()}")
    print(f"    Most active topic: {df['topic'].value_counts().idxmax()}")
    print("\n  Done! Move on to Step 4 — PySpark.")
