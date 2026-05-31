from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import os

# ─────────────────────────────────────────────
# STEP 4 — Big Data Processing with PySpark
# ─────────────────────────────────────────────
# WHY PySpark?
# Pandas runs on ONE core of your machine.
# PySpark can distribute work across many cores
# (or many machines in the cloud). For a dataset
# like ours — 500 rows — Pandas is fine. But in
# real industry projects, news datasets have
# MILLIONS of rows. PySpark is the standard tool
# for that scale. This step shows you know how
# to work at that scale.


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("NewsSentimentAnalysis")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED "
                "-Djava.security.manager=allow")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("  Spark session started ✓")
    return spark


def load_data(spark):
    path = "data/processed/headlines_clean.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Clean data not found. Run preprocess.py first.")

    # PySpark reads CSV differently from Pandas:
    # - header=True  → first row is column names
    # - inferSchema  → auto-detect int/string/date types
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )
    print(f"  Loaded {df.count()} rows into Spark DataFrame")
    print(f"  Columns: {df.columns}\n")
    return df


# ─────────────────────────────────────────────
# ANALYSIS 1 — Article count per topic
# ─────────────────────────────────────────────
def analysis_topic_counts(df):
    print("── Analysis 1: Article count per topic ────")
    result = (
        df.groupBy("topic")
          .agg(F.count("*").alias("article_count"))
          .orderBy(F.desc("article_count"))
    )
    result.show()
    return result


# ─────────────────────────────────────────────
# ANALYSIS 2 — Average title length per topic
# ─────────────────────────────────────────────
def analysis_title_length(df):
    print("── Analysis 2: Average title length per topic ─")
    # F.length() counts characters in a string — a PySpark built-in
    result = (
        df.withColumn("title_length", F.length("title"))
          .groupBy("topic")
          .agg(
              F.round(F.avg("title_length"), 1).alias("avg_title_chars"),
              F.min("title_length").alias("shortest"),
              F.max("title_length").alias("longest")
          )
          .orderBy(F.desc("avg_title_chars"))
    )
    result.show()
    return result


# ─────────────────────────────────────────────
# ANALYSIS 3 — Top 10 most active sources
# ─────────────────────────────────────────────
def analysis_top_sources(df):
    print("── Analysis 3: Top 10 most active sources ──")
    result = (
        df.groupBy("source_name")
          .agg(F.count("*").alias("article_count"))
          .orderBy(F.desc("article_count"))
          .limit(10)
    )
    result.show()
    return result


# ─────────────────────────────────────────────
# ANALYSIS 4 — Articles per topic per day
# ─────────────────────────────────────────────
def analysis_topic_per_day(df):
    print("── Analysis 4: Articles per topic per day ──")
    result = (
        df.groupBy("date", "topic")
          .agg(F.count("*").alias("article_count"))
          .orderBy("date", "topic")
    )
    result.show(20)
    return result


# ─────────────────────────────────────────────
# ANALYSIS 5 — Word frequency in titles
# (This is the "Big Data" showcase moment)
# ─────────────────────────────────────────────
def analysis_word_frequency(df):
    print("── Analysis 5: Top 30 words in headlines ───")

    # Stopwords to filter out
    stopwords = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "it", "its", "as", "by",
        "this", "that", "was", "are", "be", "has", "have", "been",
        "from", "will", "not", "new", "says", "after", "over",
        "what", "how", "why", "who", "can", "about", "up", "more",
        "s", "u", "removed"
    ]

    result = (
        df.select(F.lower(F.col("title")).alias("title"))   # lowercase
          .select(F.split("title", r"\W+").alias("words"))  # split on non-word chars
          .select(F.explode("words").alias("word"))          # one row per word
          .filter(F.length("word") > 2)                     # skip 1-2 letter words
          .filter(~F.col("word").isin(stopwords))            # remove stopwords
          .groupBy("word")
          .agg(F.count("*").alias("frequency"))
          .orderBy(F.desc("frequency"))
          .limit(30)
    )
    result.show(30)
    return result


# ─────────────────────────────────────────────
# SAVE results to CSV for dashboard
# ─────────────────────────────────────────────
def save_spark_results(topic_counts, title_lengths, top_sources, word_freq):
    """
    Convert Spark DataFrames back to Pandas to save as CSV.
    In real Big Data pipelines you'd write directly to HDFS or S3,
    but for this project we save locally for Power BI.
    """
    out = "data/processed"
    os.makedirs(out, exist_ok=True)

    topic_counts.toPandas().to_csv(f"{out}/spark_topic_counts.csv", index=False)
    title_lengths.toPandas().to_csv(f"{out}/spark_title_lengths.csv", index=False)
    top_sources.toPandas().to_csv(f"{out}/spark_top_sources.csv", index=False)
    word_freq.toPandas().to_csv(f"{out}/spark_word_freq.csv", index=False)

    print(f"\n  Saved 4 result CSVs to '{out}/'")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Step 4 — Big Data Processing (PySpark)")
    print("=" * 50)

    spark = create_spark_session()
    df    = load_data(spark)

    topic_counts  = analysis_topic_counts(df)
    title_lengths = analysis_title_length(df)
    top_sources   = analysis_top_sources(df)
    topic_per_day = analysis_topic_per_day(df)
    word_freq     = analysis_word_frequency(df)

    save_spark_results(topic_counts, title_lengths, top_sources, word_freq)

    spark.stop()
    print("\n  Spark session closed.")
    print("  Done! Move on to Step 5 — Sentiment Analysis.")
