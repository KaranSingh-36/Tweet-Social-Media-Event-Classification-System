"""Evaluate every trained model on the SAME held-out test split.

Outputs
-------
* ``reports/evaluation_results/metrics.json`` – accuracy, precision, recall,
  macro/weighted F1, per-class report, confusion matrix, timings.
* ``data/evaluation/test_predictions.parquet`` – per-tweet predictions and
  probabilities of every model (used by the error-analysis page).
* ``reports/evaluation_results/*_classification_report.csv``

Run:  ``python -m src.evaluation``
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import settings
from src.data_loader import load_dataset
from src.prediction import MODEL_DISPLAY_NAMES, EventClassifierService


def compute_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "classification_report": classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def _training_seconds() -> dict:
    out = {}
    classical = settings.REPORTS_DIR / "classical_training.json"
    if classical.exists():
        rep = json.loads(classical.read_text())
        out["svm"] = rep.get("svm", {}).get("train_seconds")
        out["logreg"] = rep.get("logreg", {}).get("train_seconds")
    if settings.TRANSFORMER_HISTORY_FILE.exists():
        out["distilbert"] = json.loads(settings.TRANSFORMER_HISTORY_FILE.read_text()).get("train_seconds")
    return out


def evaluate_all(models: list[str] | None = None) -> dict:
    df = load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    service = EventClassifierService()
    models = models or service.available_models()
    train_secs = _training_seconds()

    preds = test[["uid", "source", "text_original", "text_clean", "text_processed", "label", "original_labels", "n_topics", "event_type"]].copy()
    results = {"test_size": int(len(test)), "classes": None, "models": {}}

    for key in models:
        print(f"Evaluating {MODEL_DISPLAY_NAMES[key]} on {len(test):,} test tweets ...")
        texts = test["text_original"].tolist()
        service.predict_proba(texts[:64], key)  # warm-up: exclude one-off initialisation from the timing
        start = time.perf_counter()
        proba, classes = service.predict_proba(texts, key)
        elapsed = time.perf_counter() - start
        results["classes"] = classes
        y_pred = [classes[i] for i in proba.argmax(1)]
        metrics = compute_metrics(test["label"].tolist(), y_pred, classes)
        metrics.update({
            "display_name": MODEL_DISPLAY_NAMES[key],
            "inference_seconds_total": elapsed,
            "inference_ms_per_tweet": elapsed / len(test) * 1000,
            "train_seconds": train_secs.get(key),
            "mean_confidence": float(proba.max(1).mean()),
        })
        results["models"][key] = metrics
        preds[f"pred_{key}"] = y_pred
        preds[f"conf_{key}"] = proba.max(1)
        preds[f"true_prob_{key}"] = proba[np.arange(len(test)), [classes.index(c) for c in test["label"]]]
        preds[f"proba_{key}"] = list(proba.astype(np.float32))
        pd.DataFrame(metrics["classification_report"]).T.to_csv(settings.REPORTS_DIR / f"{key}_classification_report.csv")
        print(f"  accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['f1_macro']:.4f} | weighted-F1 {metrics['f1_weighted']:.4f} "
              f"| {metrics['inference_ms_per_tweet']:.2f} ms/tweet")

    classical = settings.REPORTS_DIR / "classical_training.json"
    if classical.exists():
        results["event_type_model"] = json.loads(classical.read_text()).get("event_type")

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    settings.METRICS_FILE.write_text(json.dumps(results, indent=2))
    preds.to_parquet(settings.TEST_PREDICTIONS_FILE, index=False)
    summary = pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                                                    "train_seconds", "inference_ms_per_tweet")}
                            for k, v in results["models"].items()}).T
    summary.to_csv(settings.REPORTS_DIR / "model_comparison.csv")
    print(summary.round(4).to_string())
    return results


def load_metrics() -> dict:
    if not settings.METRICS_FILE.exists():
        return {}
    return json.loads(settings.METRICS_FILE.read_text())


def load_test_predictions() -> pd.DataFrame:
    if not settings.TEST_PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(settings.TEST_PREDICTIONS_FILE)


def comparison_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, m in metrics.get("models", {}).items():
        rows.append({
            "key": key,
            "Model": m["display_name"],
            "Accuracy": m["accuracy"],
            "Precision (macro)": m["precision_macro"],
            "Recall (macro)": m["recall_macro"],
            "Macro F1": m["f1_macro"],
            "Weighted F1": m["f1_weighted"],
            "Training time (s)": m.get("train_seconds"),
            "Inference (ms/tweet)": m["inference_ms_per_tweet"],
        })
    return pd.DataFrame(rows)


def per_class_table(metrics: dict, key: str) -> pd.DataFrame:
    report = metrics["models"][key]["classification_report"]
    rows = [{"Category": c, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1-score"], "Support": int(v["support"])}
            for c, v in report.items() if c in metrics["classes"]]
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    evaluate_all()
