"""Error analysis on the held-out test predictions.

Each misclassified tweet is annotated with *possible* reasons using transparent
heuristics (they are indicators, not proofs):

* short tweet           – very few informative tokens after preprocessing
* multi-topic tweet     – human annotators gave the tweet several topics
* similar categories    – the confused pair is semantically close
* hashtag-heavy         – meaning is mostly carried by hashtags
* out-of-vocabulary     – many tokens unseen in training (slang, names, typos)
* sarcasm / humour cues – "lol", "lmao", 😂, "/s", "yeah right" ...
* non-English / code-mixed text
* low confidence        – the model itself was unsure
"""
from __future__ import annotations

import re

import pandas as pd

from config import settings

SIMILAR_PAIRS = {
    frozenset({"News & Social Concern", "Natural Disaster"}): "disaster coverage is also news",
    frozenset({"News & Social Concern", "Business"}): "economic news overlaps with business",
    frozenset({"News & Social Concern", "Health & Fitness"}): "public-health news overlaps with health",
    frozenset({"News & Social Concern", "Education"}): "school/policy news overlaps with education",
    frozenset({"News & Social Concern", "Science & Technology"}): "tech policy news overlaps with technology",
    frozenset({"Entertainment", "Lifestyle & Daily Life"}): "personal posts about music/TV/hobbies",
    frozenset({"Entertainment", "Sports & Gaming"}): "gaming/streaming content is also entertainment",
    frozenset({"Business", "Science & Technology"}): "tech companies and products",
    frozenset({"Health & Fitness", "Lifestyle & Daily Life"}): "wellbeing and routines",
    frozenset({"Education", "Lifestyle & Daily Life"}): "student life posts",
    frozenset({"Business", "Lifestyle & Daily Life"}): "shopping, products and personal finance",
    frozenset({"Entertainment", "News & Social Concern"}): "celebrity news and activism",
}
SARCASM_RE = re.compile(r"\b(lol|lmao|lmfao|rofl|smh|jk|yeah right|sure+ly|/s|haha+|hehe)\b|😂|🤣|🙄|😏", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")

MODEL_KEYS = ("svm", "logreg", "distilbert")


def available_models(preds: pd.DataFrame) -> list[str]:
    return [k for k in MODEL_KEYS if f"pred_{k}" in preds.columns]


def possible_reasons(row: pd.Series, model_key: str, vocabulary: dict | None = None) -> list[str]:
    reasons = []
    true, pred = row["label"], row[f"pred_{model_key}"]
    tokens = row["text_processed"].split()
    original = row["text_original"]

    if len(tokens) <= 4:
        reasons.append("Short tweet: very little content to classify")
    if row.get("n_topics", 1) and int(row.get("n_topics", 1)) > 1:
        reasons.append(f"Multi-topic tweet: annotators labelled it [{row['original_labels']}]")
    pair = SIMILAR_PAIRS.get(frozenset({true, pred}))
    if pair:
        reasons.append(f"Similar categories: {pair}")
    n_hashtags = len(HASHTAG_RE.findall(original))
    if n_hashtags >= 3 or (tokens and n_hashtags / max(len(original.split()), 1) > 0.3):
        reasons.append("Hashtag-heavy: meaning is carried mainly by hashtags")
    if vocabulary and tokens:
        oov = sum(t not in vocabulary for t in tokens) / len(tokens)
        if oov >= 0.5:
            reasons.append(f"Out-of-vocabulary words ({oov:.0%}): slang, names or misspellings")
    if SARCASM_RE.search(original):
        reasons.append("Humour / sarcasm cues (lol, 😂, /s ...)")
    non_ascii = sum(ord(ch) > 127 for ch in original) / max(len(original), 1)
    if non_ascii > 0.3:
        reasons.append("Non-English or code-mixed text")
    if row[f"conf_{model_key}"] < 0.5:
        reasons.append("Low model confidence: genuinely ambiguous to the model")
    if not reasons:
        reasons.append("Implicit context: topic requires world knowledge not present in the text")
    return reasons


def _vocabulary() -> dict | None:
    try:
        import joblib

        return joblib.load(settings.TFIDF_FILE).vocabulary_
    except Exception:  # noqa: BLE001
        return None


def misclassified(preds: pd.DataFrame, model_key: str, with_reasons: bool = True) -> pd.DataFrame:
    wrong = preds[preds["label"] != preds[f"pred_{model_key}"]].copy()
    if with_reasons and not wrong.empty:
        vocab = _vocabulary()
        wrong["possible_reasons"] = wrong.apply(lambda r: possible_reasons(r, model_key, vocab), axis=1)
    return wrong.sort_values(f"conf_{model_key}", ascending=False)


def correctly_classified(preds: pd.DataFrame, model_key: str) -> pd.DataFrame:
    return preds[preds["label"] == preds[f"pred_{model_key}"]].sort_values(f"conf_{model_key}", ascending=False)


def error_rate_by_category(preds: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = preds.assign(error=preds["label"] != preds[f"pred_{model_key}"])
    out = df.groupby("label").agg(tweets=("uid", "count"), errors=("error", "sum")).reset_index()
    out["error_rate"] = out["errors"] / out["tweets"]
    return out.rename(columns={"label": "category"}).sort_values("error_rate", ascending=False)


def top_confusions(preds: pd.DataFrame, model_key: str, n: int = 10) -> pd.DataFrame:
    wrong = preds[preds["label"] != preds[f"pred_{model_key}"]]
    pairs = wrong.groupby(["label", f"pred_{model_key}"]).size().reset_index(name="count")
    pairs = pairs.rename(columns={"label": "actual", f"pred_{model_key}": "predicted"})
    totals = preds["label"].value_counts()
    pairs["share_of_actual_class"] = pairs.apply(lambda r: r["count"] / totals[r["actual"]], axis=1)
    pairs["related"] = pairs.apply(lambda r: SIMILAR_PAIRS.get(frozenset({r["actual"], r["predicted"]}), ""), axis=1)
    return pairs.sort_values("count", ascending=False).head(n).reset_index(drop=True)


def reason_summary(wrong: pd.DataFrame) -> pd.DataFrame:
    if wrong.empty or "possible_reasons" not in wrong:
        return pd.DataFrame(columns=["reason", "count"])
    reasons = wrong["possible_reasons"].explode().str.split(":").str[0].str.replace(r"\s*\(.*\)", "", regex=True)
    out = reasons.value_counts().reset_index()
    out.columns = ["reason", "count"]
    return out


def model_disagreements(preds: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    d = preds[preds[f"pred_{a}"] != preds[f"pred_{b}"]].copy()
    d["winner"] = "neither"
    d.loc[d[f"pred_{a}"] == d["label"], "winner"] = a
    d.loc[d[f"pred_{b}"] == d["label"], "winner"] = b
    return d


def error_by_length(preds: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = preds.assign(tokens=preds["text_processed"].str.split().str.len(), error=preds["label"] != preds[f"pred_{model_key}"])
    df["length_bucket"] = pd.cut(df["tokens"], bins=[0, 4, 8, 12, 16, 25, 1000], labels=["1-4", "5-8", "9-12", "13-16", "17-25", "26+"])
    out = df.groupby("length_bucket", observed=True).agg(tweets=("uid", "count"), error_rate=("error", "mean")).reset_index()
    return out


def error_by_topic_count(preds: pd.DataFrame, model_key: str) -> pd.DataFrame:
    df = preds[preds["source"] == "TweetTopic"].assign(error=lambda x: x["label"] != x[f"pred_{model_key}"])
    df = df.assign(topics=df["n_topics"].clip(upper=3).map({1: "1 topic", 2: "2 topics", 3: "3+ topics"}))
    return df.groupby("topics").agg(tweets=("uid", "count"), error_rate=("error", "mean")).reset_index()
