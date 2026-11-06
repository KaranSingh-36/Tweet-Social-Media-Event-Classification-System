"""Classical machine-learning models: TF-IDF + SVM and TF-IDF + Logistic Regression.

* **SVM** – ``LinearSVC`` (one-vs-rest, class-balanced). A linear SVM only
  outputs decision scores, which are *not* probabilities, so the final model is
  wrapped in ``CalibratedClassifierCV`` (Platt / sigmoid scaling, 5-fold) to
  obtain calibrated class probabilities.
* **Logistic Regression** – multinomial, class-balanced; natively probabilistic.
* **Disaster event-type model** – TF-IDF + Logistic Regression trained on HumAID
  to predict Earthquake / Hurricane / Flood / Wildfire for disaster tweets.

Hyper-parameter ``C`` is selected on the validation split (never the test split).

Run:  ``python -m src.classical_model``
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from config import settings
from src.data_loader import load_dataset
from src.feature_extraction import build_tfidf_vectorizer, linear_contributions
from src.preprocessing import preprocess_for_classical

CLASSICAL_TRAINING_REPORT = settings.REPORTS_DIR / "classical_training.json"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_svm(c: float) -> LinearSVC:
    return LinearSVC(C=c, class_weight="balanced", max_iter=5000, random_state=settings.RANDOM_STATE)


def _make_logreg(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=settings.RANDOM_STATE)


def _grid_search(make_model, grid, x_train, y_train, x_val, y_val) -> tuple[float, list[dict]]:
    results = []
    for c in grid:
        model = make_model(c).fit(x_train, y_train)
        pred = model.predict(x_val)
        results.append(
            {"C": c, "val_macro_f1": float(f1_score(y_val, pred, average="macro")), "val_accuracy": float(accuracy_score(y_val, pred))}
        )
    best = max(results, key=lambda r: r["val_macro_f1"])
    return best["C"], results


def train_classical_models(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    df = load_dataset() if df is None else df
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]

    encoder = LabelEncoder().fit(df["label"])
    y_train, y_val = encoder.transform(train["label"]), encoder.transform(val["label"])

    t0 = time.perf_counter()
    vectorizer = build_tfidf_vectorizer()
    x_train = vectorizer.fit_transform(train["text_processed"])  # fitted on TRAIN only
    x_val = vectorizer.transform(val["text_processed"])
    tfidf_time = time.perf_counter() - t0
    print(f"TF-IDF: {x_train.shape[1]:,} features from {x_train.shape[0]:,} training tweets")

    report: dict = {"tfidf": {"n_features": int(x_train.shape[1]), "params": {k: str(v) for k, v in settings.TFIDF_PARAMS.items()}, "fit_seconds": tfidf_time}}

    # --- SVM ---------------------------------------------------------------
    best_c_svm, svm_grid = _grid_search(_make_svm, settings.SVM_C_GRID, x_train, y_train, x_val, y_val)
    t0 = time.perf_counter()
    svm = CalibratedClassifierCV(_make_svm(best_c_svm), method="sigmoid", cv=5).fit(x_train, y_train)
    svm_time = time.perf_counter() - t0
    report["svm"] = {"best_C": best_c_svm, "grid": svm_grid, "train_seconds": svm_time + tfidf_time,
                     "calibration": "CalibratedClassifierCV(sigmoid, cv=5)"}
    print(f"SVM: best C={best_c_svm} (val macro-F1 {max(r['val_macro_f1'] for r in svm_grid):.4f}), trained in {svm_time:.1f}s")

    # --- Logistic Regression -------------------------------------------------
    best_c_lr, lr_grid = _grid_search(_make_logreg, settings.LOGREG_C_GRID, x_train, y_train, x_val, y_val)
    t0 = time.perf_counter()
    logreg = _make_logreg(best_c_lr).fit(x_train, y_train)
    lr_time = time.perf_counter() - t0
    report["logreg"] = {"best_C": best_c_lr, "grid": lr_grid, "train_seconds": lr_time + tfidf_time}
    print(f"LogReg: best C={best_c_lr} (val macro-F1 {max(r['val_macro_f1'] for r in lr_grid):.4f}), trained in {lr_time:.1f}s")

    # --- Disaster event-type model ------------------------------------------
    report["event_type"] = train_event_type_model(save=save)

    if save:
        settings.CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, settings.TFIDF_FILE)
        joblib.dump(svm, settings.SVM_FILE)
        joblib.dump(logreg, settings.LOGREG_FILE)
        joblib.dump(encoder, settings.LABEL_ENCODER_FILE)
        CLASSICAL_TRAINING_REPORT.write_text(json.dumps(report, indent=2))
        print(f"Saved classical models to {settings.CLASSICAL_MODEL_DIR}")
    return report


def train_event_type_model(save: bool = True) -> dict:
    df = pd.read_parquet(settings.EVENT_TYPE_DATASET_FILE)
    train, test = df[df["split"].isin(["train", "val"])], df[df["split"] == "test"]
    pipe = Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer(max_features=30_000)),
            ("clf", LogisticRegression(C=5.0, class_weight="balanced", max_iter=3000, random_state=settings.RANDOM_STATE)),
        ]
    )
    t0 = time.perf_counter()
    pipe.fit(train["text_processed"], train["event_type"])
    elapsed = time.perf_counter() - t0
    pred = pipe.predict(test["text_processed"])
    result = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "accuracy": float(accuracy_score(test["event_type"], pred)),
        "macro_f1": float(f1_score(test["event_type"], pred, average="macro")),
        "report": classification_report(test["event_type"], pred, output_dict=True, zero_division=0),
        "train_seconds": elapsed,
        "note": "Evaluated on HumAID's official test split (same 19 events as training).",
    }
    result["unseen_event_cv"] = _unseen_event_cv(df, pipe)
    print(f"Event-type model: accuracy {result['accuracy']:.4f}, macro-F1 {result['macro_f1']:.4f}; "
          f"unseen-event CV accuracy {result['unseen_event_cv']['accuracy']:.4f}")
    if save:
        settings.CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, settings.EVENT_TYPE_MODEL_FILE)
    return result


def _unseen_event_cv(df: pd.DataFrame, pipe: Pipeline, n_splits: int = 5) -> dict:
    """GroupKFold by event name: every test fold contains only disasters the
    model never saw during training – a realistic estimate for new events."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    y_true, y_pred = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(df, df["event_type"], groups=df["event_name"]):
        model = clone(pipe).fit(df["text_processed"].iloc[tr], df["event_type"].iloc[tr])
        y_pred.extend(model.predict(df["text_processed"].iloc[te]))
        y_true.extend(df["event_type"].iloc[te])
    return {"n_splits": n_splits, "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "note": "Grouped by event: test folds contain only disasters unseen in training."}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class ClassicalClassifier:
    """Loads a saved TF-IDF + linear model and produces real probabilities."""

    name: str  # "svm" or "logreg"
    vectorizer: object
    model: object
    classes: list[str]

    DISPLAY_NAMES = {"svm": "TF-IDF + SVM", "logreg": "TF-IDF + Logistic Regression"}

    @classmethod
    def load(cls, name: str = "svm") -> "ClassicalClassifier":
        path = {"svm": settings.SVM_FILE, "logreg": settings.LOGREG_FILE}[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.classical_model` first.")
        vectorizer = joblib.load(settings.TFIDF_FILE)
        encoder = joblib.load(settings.LABEL_ENCODER_FILE)
        return cls(name, vectorizer, joblib.load(path), list(encoder.classes_))

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAMES[self.name]

    @property
    def coef(self) -> np.ndarray:
        """Linear weights (averaged over calibration folds for the SVM)."""
        if hasattr(self.model, "coef_"):
            return self.model.coef_
        return np.mean([c.estimator.coef_ for c in self.model.calibrated_classifiers_], axis=0)

    def predict_proba_processed(self, processed: list[str]) -> np.ndarray:
        return self.model.predict_proba(self.vectorizer.transform(processed))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.predict_proba_processed([preprocess_for_classical(t) for t in texts])

    def explain(self, text: str, class_index: int, top_k: int = 8) -> list[dict]:
        return linear_contributions(self.vectorizer, self.coef, preprocess_for_classical(text), class_index, top_k)


class EventTypeClassifier:
    """Predicts the disaster type for tweets classified as Natural Disaster."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes = list(pipeline.classes_)

    @classmethod
    def load(cls) -> "EventTypeClassifier | None":
        if not settings.EVENT_TYPE_MODEL_FILE.exists():
            return None
        return cls(joblib.load(settings.EVENT_TYPE_MODEL_FILE))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([preprocess_for_classical(t) for t in texts])


if __name__ == "__main__":
    train_classical_models()
