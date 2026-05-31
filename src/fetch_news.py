import requests
import pandas as pd
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION — paste your API key below
# ─────────────────────────────────────────────
API_KEY = "b2a49e9f060546bf940b22dbc28e4f47"
BASE_URL = "https://newsapi.org/v2/everything"

# We fetch 5 topics × up to 100 articles = ~500 rows
# That's enough to justify PySpark in Step 4
TOPICS = ["technology", "business", "health", "science", "sports"]


# ─────────────────────────────────────────────
# STEP 1A — Fetch articles for one topic
# ─────────────────────────────────────────────
def fetch_headlines(topic, api_key, page_size=100):
    """
    Calls the NewsAPI /everything endpoint and returns
    a list of article dicts for the given topic.
    """
    params = {
        "q": topic,           # search keyword
        "language": "en",     # English only
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(BASE_URL, params=params)

    # Raise an error if something went wrong (e.g. bad API key)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"NewsAPI error: {data.get('message')}")

    articles = data.get("articles", [])

    # Tag each article with the topic we searched for
    for article in articles:
        article["topic"] = topic

    return articles


# ─────────────────────────────────────────────
# STEP 1B — Loop over all topics
# ─────────────────────────────────────────────
def fetch_all_topics(topics, api_key):
    """
    Fetches headlines for every topic in the list
    and returns one combined list of articles.
    """
    all_articles = []

    for topic in topics:
        print(f"  Fetching: '{topic}' ...", end=" ")
        articles = fetch_headlines(topic, api_key)
        all_articles.extend(articles)
        print(f"{len(articles)} articles collected")

    print(f"\n  Total collected: {len(all_articles)} articles")
    return all_articles


# ─────────────────────────────────────────────
# STEP 1C — Save to CSV
# ─────────────────────────────────────────────
def save_to_csv(articles, filepath):
    """
    Converts the list of article dicts into a clean
    DataFrame and saves it to a CSV file.
    """
    df = pd.DataFrame(articles)

    # Keep only useful columns
    desired_columns = [
        "title", "description", "content",
        "author", "source", "publishedAt", "url", "topic"
    ]
    df = df[[col for col in desired_columns if col in df.columns]]

    # The 'source' column is a dict like {"id": "bbc-news", "name": "BBC News"}
    # We flatten it to just the source name
    if "source" in df.columns:
        df["source_name"] = df["source"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )
        df.drop(columns=["source"], inplace=True)

    # Save to file
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"\n  Saved {len(df)} rows → {filepath}")
    return df


# ─────────────────────────────────────────────
# RUN THE SCRIPT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Step 1 — Data Acquisition")
    print("=" * 50)

    # Create the output folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    # Fetch all topics
    articles = fetch_all_topics(TOPICS, API_KEY)

    # Build a timestamped filename so each run is saved separately
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = f"data/raw/headlines_{timestamp}.csv"

    # Save and preview
    df = save_to_csv(articles, filepath)

    print("\n  Preview (first 5 rows):")
    print(df[["title", "topic", "source_name"]].head())

    print("\n  Articles per topic:")
    print(df["topic"].value_counts().to_string())

    print("\n  Dataset shape:", df.shape)
    print("\n  Done! Move on to Step 2 — Preprocessing.")
