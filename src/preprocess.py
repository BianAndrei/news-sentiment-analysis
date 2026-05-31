import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# STEP 2 — Preprocessing & Cleaning
# ─────────────────────────────────────────────
# Goal: take the raw CSV from Step 1 and produce
# a clean, analysis-ready version of it.


def load_latest_csv(folder="data/raw"):
    """
    Automatically finds and loads the most recently
    created CSV file from the data/raw folder.
    """
    files = glob.glob(f"{folder}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{folder}'. Run fetch_news.py first.")

    latest = max(files, key=os.path.getctime)
    print(f"  Loading: {latest}")
    df = pd.read_csv(latest)
    return df


def inspect_data(df):
    """
    Prints a summary of the raw data so we understand
    what we're working with before cleaning.
    """
    print("\n── Raw Data Overview ──────────────────────")
    print(f"  Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    print(f"\n  Missing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\n  Duplicate rows : {df.duplicated().sum()}")
    print(f"\n  Articles per topic:")
    if "topic" in df.columns:
        print(df["topic"].value_counts().to_string())


def clean_data(df):
    """
    Applies all cleaning steps and returns the clean df.
    """
    print("\n── Cleaning Steps ─────────────────────────")
    original_len = len(df)

    # ── 1. Remove fully duplicate rows ──────────
    df = df.drop_duplicates()
    print(f"  [1] Removed duplicates     : {original_len - len(df)} rows dropped")

    # ── 2. Drop rows with no title ───────────────
    # Title is the most important column — useless without it
    before = len(df)
    df = df.dropna(subset=["title"])
    print(f"  [2] Dropped missing titles : {before - len(df)} rows dropped")

    # ── 3. Fill missing descriptions ────────────
    # Replace NaN descriptions with an empty string
    # so NLP tools don't crash later
    df["description"] = df["description"].fillna("")
    df["content"]     = df["content"].fillna("")
    df["author"]      = df["author"].fillna("Unknown")
    print(f"  [3] Filled missing description/content/author with defaults")

    # ── 4. Fix the publishedAt column ───────────
    # Convert the string "2024-01-15T10:30:00Z" into
    # a proper datetime so we can sort and filter by date
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt"])  # drop rows with bad dates
    print(f"  [4] Converted publishedAt to datetime")

    # ── 5. Extract useful date parts ────────────
    df["date"]  = df["publishedAt"].dt.date        # 2024-01-15
    df["hour"]  = df["publishedAt"].dt.hour        # 10
    df["month"] = df["publishedAt"].dt.month       # 1
    df["day_of_week"] = df["publishedAt"].dt.day_name()  # "Monday"
    print(f"  [5] Extracted date, hour, month, day_of_week columns")

    # ── 6. Clean the title text ──────────────────
    # Strip leading/trailing whitespace
    # Remove articles that are "[Removed]" (NewsAPI placeholder)
    df["title"] = df["title"].str.strip()
    before = len(df)
    df = df[df["title"] != "[Removed]"]
    print(f"  [6] Removed '[Removed]' placeholders : {before - len(df)} rows dropped")

    # ── 7. Normalise topic to lowercase ─────────
    if "topic" in df.columns:
        df["topic"] = df["topic"].str.lower().str.strip()
    print(f"  [7] Normalised topic column to lowercase")

    # ── 8. Reset index ───────────────────────────
    df = df.reset_index(drop=True)
    print(f"\n  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def save_clean_data(df, folder="data/processed"):
    """
    Saves the cleaned DataFrame to data/processed/
    """
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/headlines_clean.csv"
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"\n  Saved clean data → {filepath}")
    return filepath


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Step 2 — Preprocessing & Cleaning")
    print("=" * 50)

    # Load
    df = load_latest_csv()

    # Inspect raw
    inspect_data(df)

    # Clean
    df_clean = clean_data(df)

    # Save
    save_clean_data(df_clean)

    # Preview
    print("\n  Preview of clean data (first 5 rows):")
    print(df_clean[["title", "topic", "source_name", "date"]].head().to_string())

    print("\n  Done! Move on to Step 3 — EDA.")
