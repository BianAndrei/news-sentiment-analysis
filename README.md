# News Headline Sentiment Analyzer

A Big Data end-to-end pipeline that collects live news headlines,
cleans the data, runs sentiment analysis, and presents findings
in an interactive dashboard.

## Problem Statement

Can we detect whether the news is getting more positive or negative over time?
Which topics (technology, health, business…) carry the most negative sentiment?
This project answers those questions using real-time data from NewsAPI.

## Architecture

```
NewsAPI (REST) → Python (Pandas) → PySpark → VADER NLP → Power BI Dashboard
```

## Tools Used

| Stage               | Tool                           |
| ------------------- | ------------------------------ |
| Data Acquisition    | Python `requests`, NewsAPI     |
| Preprocessing       | Python, Pandas                 |
| EDA                 | Matplotlib, Seaborn, WordCloud |
| Big Data Processing | PySpark                        |
| Text Analytics      | VADER Sentiment (NLP)          |
| Visualization       | Power BI / Tableau Public      |
| Version Control     | GitHub                         |

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/BianAndrei/news-sentiment-analysis
cd news-sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your NewsAPI key to src/fetch_news.py
#    (get a free key at https://newsapi.org)
```

## How to Run

```bash
# Step 1 — Fetch headlines
python src/fetch_news.py

# Then open each notebook in order:
# notebooks/01_data_acquisition.ipynb
# notebooks/02_preprocessing.ipynb
# ...
```

## Key Findings

_(To be filled in after analysis is complete)_

## Author

Bian Andrei — Big Data Fundamentals, Final Project
