"""Dataset collection and preparation.

Sources (both public and human-annotated):

1. **TweetTopic** – ``cardiffnlp/tweet_topic_multi`` (Antypas et al., COLING 2022).
   11,267 English tweets (Sep 2019 – Aug 2021) annotated with 19 topics.
2. **HumAID** – ``QCRI/HumAID-events`` (Alam et al., ICWSM 2021).
   ~77k English tweets collected during 19 real natural disasters (2016-2019),
   annotated with humanitarian categories. Every tweet belongs to a named
   event (e.g. ``kaikoura_earthquake_2016``), which gives the disaster type.

Composite label design (documented in README):

* TweetTopic's 19 topics are mapped to 8 coarse event categories
  (``settings.TWEETTOPIC_TO_CATEGORY``). Tweets whose labels span more than two
  coarse categories are dropped as ambiguous; for tweets spanning two, the
  generic "Lifestyle & Daily Life" category is only used as a fallback, and
  otherwise the more specific (less frequent) category is the primary label.
* A stratified sample of HumAID humanitarian tweets (``not_humanitarian``
  excluded) forms the ``Natural Disaster`` category.
* The disaster event type (Earthquake / Hurricane / Flood / Wildfire) comes
  from the HumAID event name and trains a separate event-type model.

Run:  ``python -m src.data_loader``
"""
from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import settings
from src.preprocessing import (
    clean_for_transformer,
    dedup_key,
    ensure_nltk_resources,
    preprocess_for_classical,
)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tweet-event-classifier/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.rename(dest)
    return dest


def download_raw_data(force: bool = False) -> None:
    """Download the raw parquet files from the Hugging Face Hub."""
    for split in settings.TWEETTOPIC_SPLITS:
        url = settings.HF_PARQUET_API.format(
            repo=settings.TWEETTOPIC_REPO, config=settings.TWEETTOPIC_CONFIG, split=split
        )
        _download(url, settings.TWEETTOPIC_RAW_DIR / f"{split}.parquet", force)
    for event in settings.HUMAID_EVENTS:
        for split in settings.HUMAID_SPLITS:
            url = settings.HF_PARQUET_API.format(repo=settings.HUMAID_REPO, config=event, split=split)
            _download(url, settings.HUMAID_RAW_DIR / f"{event}__{split}.parquet", force)
    print(f"Raw data available in {settings.RAW_DIR}")


# --------------------------------------------------------------------------- #
# Loading raw sources
# --------------------------------------------------------------------------- #
def load_tweettopic_raw() -> pd.DataFrame:
    frames = [pd.read_parquet(settings.TWEETTOPIC_RAW_DIR / f"{s}.parquet") for s in settings.TWEETTOPIC_SPLITS]
    df = pd.concat(frames, ignore_index=True)
    df["label_name"] = df["label_name"].apply(list)
    return df


def load_humaid_raw() -> pd.DataFrame:
    frames = []
    for event, event_type in settings.HUMAID_EVENTS.items():
        for split in settings.HUMAID_SPLITS:
            part = pd.read_parquet(settings.HUMAID_RAW_DIR / f"{event}__{split}.parquet")
            part["event_name"] = event
            part["event_type"] = event_type
            part["humaid_split"] = split
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Label construction
# --------------------------------------------------------------------------- #
def assign_tweettopic_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map fine topics to coarse categories and pick one primary label."""
    mapping = settings.TWEETTOPIC_TO_CATEGORY
    df = df.copy()
    df["coarse_set"] = df["label_name"].apply(lambda ls: sorted({mapping[label] for label in ls}))
    df["n_coarse"] = df["coarse_set"].apply(len)
    stats = {
        "no_label": int((df["n_coarse"] == 0).sum()),
        "too_ambiguous": int((df["n_coarse"] > settings.MAX_COARSE_CATEGORIES).sum()),
    }
    df = df[(df["n_coarse"] >= 1) & (df["n_coarse"] <= settings.MAX_COARSE_CATEGORIES)].copy()
    freq = Counter(c for cs in df["coarse_set"] for c in cs)
    # 1) Lifestyle is a generic fallback (daily_life + news -> News).
    # 2) Otherwise the more specific (less frequent) category wins: Business+News -> Business.
    def primary(cs: list[str]) -> str:
        specific = [c for c in cs if c != settings.FALLBACK_CATEGORY] or cs
        return min(specific, key=lambda c: (freq[c], c))

    df["label"] = df["coarse_set"].apply(primary)
    stats["category_frequency_used_for_priority"] = dict(freq)
    return df, stats


def _sample_disaster_tweets(humaid: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Equal number of tweets per disaster type, spread over all events."""
    types = sorted(humaid["event_type"].unique())
    per_type = n_total // len(types)
    parts = []
    for i, etype in enumerate(types):
        pool = humaid[humaid["event_type"] == etype]
        parts.append(pool.sample(n=min(per_type, len(pool)), random_state=seed + i))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Main preparation
# --------------------------------------------------------------------------- #
def _drop_missing_and_duplicates(df: pd.DataFrame, text_col: str, label_col: str) -> tuple[pd.DataFrame, dict]:
    stats = {"input_rows": int(len(df))}
    missing = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    stats["missing_text"] = int(missing.sum())
    df = df[~missing].copy()

    exact_dup = df.duplicated(subset=[text_col])
    stats["exact_duplicates"] = int(exact_dup.sum())
    df = df[~exact_dup].copy()

    df["_key"] = df[text_col].map(dedup_key)
    empty = df["_key"] == ""
    stats["empty_after_cleaning"] = int(empty.sum())
    df = df[~empty]
    # Near-duplicates (retweets, same text with different URL/handle).
    labels_per_key = df.groupby("_key")[label_col].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    stats["conflicting_label_duplicates_removed"] = int(df["_key"].isin(conflicting).sum())
    df = df[~df["_key"].isin(conflicting)]
    near_dup = df.duplicated(subset=["_key"])
    stats["near_duplicates_retweets"] = int(near_dup.sum())
    df = df[~near_dup].drop(columns="_key")
    stats["output_rows"] = int(len(df))
    return df, stats


def build_event_type_dataset(humaid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dataset for the disaster event-type model (uses HumAID's own splits)."""
    df = humaid[~humaid["class_label"].isin(settings.HUMAID_EXCLUDED_LABELS)]
    df = df.rename(columns={"tweet_text": "text_original"})
    df, stats = _drop_missing_and_duplicates(df, "text_original", "event_type")
    capped = []
    for i, (etype, grp) in enumerate(df.groupby("event_type")):
        n = min(len(grp), settings.EVENT_TYPE_MAX_PER_TYPE)
        capped.append(grp.sample(n=n, random_state=settings.RANDOM_STATE + i))
    df = pd.concat(capped, ignore_index=True)
    df["text_processed"] = df["text_original"].map(preprocess_for_classical)
    df = df[df["text_processed"] != ""]
    df["split"] = df["humaid_split"].replace({"dev": "val"})
    stats["final_rows"] = int(len(df))
    stats["per_type"] = df["event_type"].value_counts().to_dict()
    cols = ["tweet_id", "text_original", "text_processed", "event_type", "event_name", "class_label", "split"]
    return df[cols].reset_index(drop=True), stats


def build_composite_dataset(tweettopic: pd.DataFrame, humaid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    tt, tt_stats = assign_tweettopic_categories(tweettopic)
    report["tweettopic_label_mapping"] = tt_stats
    tt_df = pd.DataFrame(
        {
            "tweet_id": tt["id"].astype(str),
            "source": "TweetTopic",
            "date": tt["date"].astype(str),
            "text_original": tt["text"],
            "label": tt["label"],
            "original_labels": tt["label_name"].apply(lambda ls: ", ".join(ls)),
            "n_topics": tt["label_name"].apply(len),
            "event_type": "",
            "event_name": "",
        }
    )

    humaid_rel = humaid[~humaid["class_label"].isin(settings.HUMAID_EXCLUDED_LABELS)]
    report["humaid_excluded_not_humanitarian"] = int(len(humaid) - len(humaid_rel))
    # Over-sample before deduplication, then trim to the target size.
    hd = _sample_disaster_tweets(humaid_rel, int(settings.HUMAID_SAMPLE_SIZE * 1.3), settings.RANDOM_STATE)
    hd_df = pd.DataFrame(
        {
            "tweet_id": hd["tweet_id"].astype(str),
            "source": "HumAID",
            "date": hd["event_name"].str.extract(r"(\d{4})$")[0].fillna(""),
            "text_original": hd["tweet_text"],
            "label": settings.NATURAL_DISASTER,
            "original_labels": hd["class_label"],
            "n_topics": 1,
            "event_type": hd["event_type"],
            "event_name": hd["event_name"],
        }
    )

    df = pd.concat([tt_df, hd_df], ignore_index=True)
    df, dedup_stats = _drop_missing_and_duplicates(df, "text_original", "label")
    report["cleaning"] = dedup_stats

    # Trim disaster tweets back to the target size, balanced by event type.
    disaster = df[df["label"] == settings.NATURAL_DISASTER]
    if len(disaster) > settings.HUMAID_SAMPLE_SIZE:
        keep = _sample_disaster_tweets(disaster, settings.HUMAID_SAMPLE_SIZE, settings.RANDOM_STATE)
        df = pd.concat([df[df["label"] != settings.NATURAL_DISASTER], keep], ignore_index=True)

    df["text_clean"] = df["text_original"].map(clean_for_transformer)
    df["text_processed"] = df["text_original"].map(preprocess_for_classical)
    empty = df["text_processed"] == ""
    report["empty_after_classical_preprocessing"] = int(empty.sum())
    df = df[~empty].reset_index(drop=True)

    # Stratified train / validation / test split (test set is never used for tuning).
    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=settings.TEST_SIZE, stratify=df["label"], random_state=settings.RANDOM_STATE
    )
    val_fraction = settings.VAL_SIZE / (1 - settings.TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        stratify=df["label"].iloc[train_val_idx],
        random_state=settings.RANDOM_STATE,
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df.insert(0, "uid", [f"T{i:05d}" for i in range(len(df))])

    report["final_rows"] = int(len(df))
    report["class_distribution"] = df["label"].value_counts().to_dict()
    report["split_sizes"] = df["split"].value_counts().to_dict()
    report["source_distribution"] = df["source"].value_counts().to_dict()
    return df, report


def write_sample_csv(df: pd.DataFrame) -> Path:
    """Real held-out test tweets for trying the bulk-upload page."""
    settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    test = df[df["split"] == "test"]
    sample = test.groupby("label").sample(n=25, random_state=settings.RANDOM_STATE)
    sample = sample.sample(frac=1.0, random_state=settings.RANDOM_STATE)
    out = pd.DataFrame({"tweet": sample["text_original"], "true_category": sample["label"], "source": sample["source"]})
    path = settings.SAMPLES_DIR / "sample_test_tweets.csv"
    out.to_csv(path, index=False)
    return path


def prepare_datasets(force_download: bool = False) -> pd.DataFrame:
    ensure_nltk_resources()
    download_raw_data(force=force_download)
    tweettopic = load_tweettopic_raw()
    humaid = load_humaid_raw()

    report = {
        "sources": {
            "TweetTopic": {
                "hub_id": settings.TWEETTOPIC_REPO,
                "url": f"https://huggingface.co/datasets/{settings.TWEETTOPIC_REPO}",
                "raw_rows": int(len(tweettopic)),
                "unique_ids": int(tweettopic["id"].nunique()),
            },
            "HumAID": {
                "hub_id": settings.HUMAID_REPO,
                "url": f"https://huggingface.co/datasets/{settings.HUMAID_REPO}",
                "raw_rows": int(len(humaid)),
                "events": len(settings.HUMAID_EVENTS),
                "raw_class_labels": humaid["class_label"].value_counts().to_dict(),
            },
        }
    }
    tweettopic = tweettopic.drop_duplicates(subset="id")

    df, composite_report = build_composite_dataset(tweettopic, humaid)
    report["composite"] = composite_report
    event_df, event_report = build_event_type_dataset(humaid)
    report["event_type_dataset"] = event_report

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.DATASET_FILE, index=False)
    df.to_csv(settings.DATASET_CSV_FILE, index=False)
    event_df.to_parquet(settings.EVENT_TYPE_DATASET_FILE, index=False)
    report["sample_csv"] = str(write_sample_csv(df).relative_to(settings.ROOT_DIR))
    settings.DATA_REPORT_FILE.write_text(json.dumps(report, indent=2, default=str))

    print(f"Composite dataset: {len(df):,} tweets -> {settings.DATASET_FILE}")
    print(df["label"].value_counts().to_string())
    print(f"Event-type dataset: {len(event_df):,} tweets -> {settings.EVENT_TYPE_DATASET_FILE}")
    return df


# --------------------------------------------------------------------------- #
# Loaders used by the app
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    if not settings.DATASET_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python -m src.data_loader` (or `python run_pipeline.py`)."
        )
    return pd.read_parquet(settings.DATASET_FILE)


def load_data_report() -> dict:
    if settings.DATA_REPORT_FILE.exists():
        return json.loads(settings.DATA_REPORT_FILE.read_text())
    return {}


if __name__ == "__main__":
    prepare_datasets()
