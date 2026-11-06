"""Single-tweet prediction service shared by the dashboard and the tests.

All outputs come from the saved models – nothing is hard-coded:

* ``category`` / ``confidence`` – arg-max class and its probability
  (calibrated probabilities for the SVM, softmax for DistilBERT).
* ``top_categories`` – the full probability distribution, sorted.
* ``keywords`` – linear term contributions (TF-IDF models) or occlusion
  importance (DistilBERT).
* ``event_type`` – only for tweets predicted as *Natural Disaster*, from the
  separate HumAID event-type model.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from config import settings
from src.classical_model import ClassicalClassifier, EventTypeClassifier

MODEL_KEYS = ("svm", "logreg", "distilbert")
MODEL_DISPLAY_NAMES = {
    "svm": "TF-IDF + SVM",
    "logreg": "TF-IDF + Logistic Regression",
    "distilbert": "DistilBERT (fine-tuned)",
}


@dataclass
class Prediction:
    text: str
    model_key: str
    model_name: str
    category: str
    confidence: float
    top_categories: list[tuple[str, float]]
    keywords: list[dict] = field(default_factory=list)
    event_type: str | None = None
    event_type_confidence: float | None = None
    event_type_probabilities: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0


class EventClassifierService:
    """Lazily loads every available model once and serves predictions."""

    def __init__(self, load_transformer: bool = True):
        self._models: dict[str, object] = {}
        self._load_transformer = load_transformer
        self.event_type_model = EventTypeClassifier.load()
        self.load_errors: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def available_models(self) -> list[str]:
        keys = []
        if settings.SVM_FILE.exists():
            keys.append("svm")
        if settings.LOGREG_FILE.exists():
            keys.append("logreg")
        if self._load_transformer:
            from src.transformer_model import TransformerClassifier

            if TransformerClassifier.is_available():
                keys.append("distilbert")
        return keys

    def get_model(self, key: str):
        if key not in self._models:
            if key in ("svm", "logreg"):
                self._models[key] = ClassicalClassifier.load(key)
            elif key == "distilbert":
                from src.transformer_model import TransformerClassifier

                self._models[key] = TransformerClassifier.load()
            else:
                raise ValueError(f"Unknown model '{key}'. Choose from {MODEL_KEYS}.")
        return self._models[key]

    def warm_up(self) -> None:
        """Load every available model and run one dummy prediction so that the
        first real request does not pay for lazy initialisation (WordNet,
        tokenizer, PyTorch kernels)."""
        for key in self.available_models():
            try:
                self.predict("warm up the event classifier", key, explain=False)
            except Exception as exc:  # noqa: BLE001 - reported in the UI instead
                self.load_errors[key] = str(exc)

    # ------------------------------------------------------------------ #
    def predict_proba(self, texts: list[str], model_key: str, batch_size: int = settings.BULK_BATCH_SIZE) -> tuple[np.ndarray, list[str]]:
        model = self.get_model(model_key)
        if model_key == "distilbert":
            return model.predict_proba(texts, batch_size=batch_size), model.classes
        return model.predict_proba(texts), model.classes

    def predict_event_types(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        if self.event_type_model is None or not texts:
            return [""] * len(texts), np.zeros(len(texts))
        proba = self.event_type_model.predict_proba(texts)
        idx = proba.argmax(1)
        return [self.event_type_model.classes[i] for i in idx], proba[np.arange(len(texts)), idx]

    def predict(self, text: str, model_key: str = "svm", explain: bool = True, top_k_keywords: int = 8) -> Prediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter a non-empty tweet.")
        model = self.get_model(model_key)
        start = time.perf_counter()
        proba, classes = self.predict_proba([text], model_key)
        latency = (time.perf_counter() - start) * 1000
        probs = proba[0]
        best = int(np.argmax(probs))
        ranked = sorted(zip(classes, probs.tolist()), key=lambda kv: -kv[1])
        pred = Prediction(
            text=text,
            model_key=model_key,
            model_name=MODEL_DISPLAY_NAMES[model_key],
            category=classes[best],
            confidence=float(probs[best]),
            top_categories=[(c, float(p)) for c, p in ranked],
            latency_ms=latency,
        )
        if explain:
            pred.keywords = model.explain(text, best, top_k=top_k_keywords)
        if pred.category == settings.NATURAL_DISASTER and self.event_type_model is not None:
            et_proba = self.event_type_model.predict_proba([text])[0]
            order = np.argsort(-et_proba)
            pred.event_type_probabilities = [(self.event_type_model.classes[i], float(et_proba[i])) for i in order]
            pred.event_type, pred.event_type_confidence = pred.event_type_probabilities[0]
        return pred


# --------------------------------------------------------------------------- #
# Prediction history (powers "Total predictions" and "Recent activity")
# --------------------------------------------------------------------------- #
LOG_COLUMNS = ["timestamp", "mode", "model", "category", "confidence", "text"]


def log_predictions(rows: list[dict], mode: str) -> None:
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not settings.PREDICTION_LOG_FILE.exists()
    now = datetime.now().isoformat(timespec="seconds")
    with open(settings.PREDICTION_LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "timestamp": now, "mode": mode, "model": r["model"], "category": r["category"],
                "confidence": round(float(r["confidence"]), 4), "text": str(r.get("text", ""))[:280],
            })


def load_prediction_log() -> pd.DataFrame:
    if not settings.PREDICTION_LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(settings.PREDICTION_LOG_FILE)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
