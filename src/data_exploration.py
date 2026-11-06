"""Exploratory data analysis of the processed tweet dataset.

Run:  ``python -m src.data_exploration``  (writes a JSON summary to reports/)
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from config import settings
from src.data_loader import load_data_report, load_dataset

HASHTAG_RE = re.compile(r"#\w+")


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_length"] = df["text_original"].str.len()
    df["word_count"] = df["text_original"].str.split().str.len()
    df["processed_token_count"] = df["text_processed"].str.split().str.len()
    df["hashtag_count"] = df["text_original"].str.count(r"#\w+")
    return df


def summary_statistics(df: pd.DataFrame, report: dict | None = None) -> dict:
    report = report if report is not None else load_data_report()
    df = add_length_features(df)
    counts = df["label"].value_counts()
    cleaning = report.get("composite", {}).get("cleaning", {})
    shortest = df.loc[df["char_length"].idxmin()]
    longest = df.loc[df["char_length"].idxmax()]
    return {
        "total_tweets": int(len(df)),
        "n_categories": int(df["label"].nunique()),
        "tweets_per_category": counts.to_dict(),
        "missing_text_removed": int(cleaning.get("missing_text", 0)),
        "duplicates_removed": int(cleaning.get("exact_duplicates", 0) + cleaning.get("near_duplicates_retweets", 0)
                                  + cleaning.get("conflicting_label_duplicates_removed", 0)),
        "ambiguous_removed": int(report.get("composite", {}).get("tweettopic_label_mapping", {}).get("too_ambiguous", 0)),
        "remaining_missing_values": int(df[["text_original", "label"]].isna().sum().sum()),
        "avg_char_length": float(df["char_length"].mean()),
        "avg_word_count": float(df["word_count"].mean()),
        "median_word_count": float(df["word_count"].median()),
        "shortest_tweet": {"text": shortest["text_original"], "chars": int(shortest["char_length"]), "label": shortest["label"]},
        "longest_tweet": {"text": longest["text_original"], "chars": int(longest["char_length"]), "label": longest["label"]},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "majority_class": counts.idxmax(),
        "minority_class": counts.idxmin(),
        "split_sizes": df["split"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "tweets_with_hashtags_pct": float((df["hashtag_count"] > 0).mean() * 100),
    }


def top_words(df: pd.DataFrame, n: int = 25, category: str | None = None) -> pd.DataFrame:
    sub = df if category is None else df[df["label"] == category]
    counter = Counter(tok for text in sub["text_processed"] for tok in text.split())
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def top_words_by_category(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    frames = []
    for cat in sorted(df["label"].unique()):
        t = top_words(df, n, cat)
        t["category"] = cat
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def top_hashtags(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    counter = Counter(h.lower() for text in df["text_original"] for h in HASHTAG_RE.findall(text))
    return pd.DataFrame(counter.most_common(n), columns=["hashtag", "count"])


def category_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = add_length_features(df)
    return (
        df.groupby("label")
        .agg(tweets=("uid", "count"), avg_words=("word_count", "mean"), avg_chars=("char_length", "mean"),
             avg_hashtags=("hashtag_count", "mean"), avg_processed_tokens=("processed_token_count", "mean"))
        .round(2)
        .sort_values("tweets", ascending=False)
        .reset_index()
        .rename(columns={"label": "category"})
    )


def main() -> None:
    df = load_dataset()
    stats = summary_statistics(df)
    stats["top_words"] = top_words(df, 30).to_dict("records")
    stats["top_hashtags"] = top_hashtags(df, 20).to_dict("records")
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.REPORTS_DIR / "data_exploration.json"
    out.write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("top_words", "top_hashtags")}, indent=2, default=str))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
