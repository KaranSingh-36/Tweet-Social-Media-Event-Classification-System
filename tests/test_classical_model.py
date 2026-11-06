"""Tests for TF-IDF features and the classical models."""
import numpy as np
import pandas as pd
import pytest

from config import settings
from src.feature_extraction import build_tfidf_vectorizer, linear_contributions

MODELS_READY = settings.SVM_FILE.exists() and settings.LOGREG_FILE.exists()


def test_tfidf_builds_unigrams_and_bigrams():
    vec = build_tfidf_vectorizer(min_df=1, max_df=1.0)
    x = vec.fit_transform(["rescue team search survivor", "football match win", "rescue team arrive"])
    names = set(vec.get_feature_names_out())
    assert "rescue team" in names and "football" in names
    assert x.shape[0] == 3


def test_linear_contributions_ranks_positive_terms():
    vec = build_tfidf_vectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1))
    vec.fit(["earthquake rescue", "football goal"])
    coef = np.zeros((2, len(vec.vocabulary_)))
    coef[0, vec.vocabulary_["earthquake"]] = 2.0
    coef[0, vec.vocabulary_["rescue"]] = 1.0
    out = linear_contributions(vec, coef, "earthquake rescue", 0)
    assert [o["term"] for o in out] == ["earthquake", "rescue"]


def test_train_on_tiny_dataset_without_saving():
    from src.classical_model import train_classical_models

    rows = []
    for i in range(30):
        rows.append({"text_processed": f"earthquake rescue survivor quake{i % 3}", "label": "Natural Disaster", "split": "train" if i < 24 else "val"})
        rows.append({"text_processed": f"football goal match team{i % 3}", "label": "Sports & Gaming", "split": "train" if i < 24 else "val"})
    report = train_classical_models(pd.DataFrame(rows), save=False) if settings.EVENT_TYPE_DATASET_FILE.exists() else None
    if report is None:
        pytest.skip("event-type dataset not prepared")
    assert report["svm"]["best_C"] in settings.SVM_C_GRID
    assert report["logreg"]["best_C"] in settings.LOGREG_C_GRID


@pytest.mark.skipif(not MODELS_READY, reason="classical models not trained")
@pytest.mark.parametrize("name", ["svm", "logreg"])
def test_saved_models_produce_probabilities(name):
    from src.classical_model import ClassicalClassifier

    clf = ClassicalClassifier.load(name)
    proba = clf.predict_proba(["Massive earthquake hits Japan, rescue teams search for survivors",
                               "India wins the cricket match after an incredible final over"])
    assert proba.shape == (2, len(clf.classes))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert clf.classes[proba[0].argmax()] == "Natural Disaster"
    assert clf.classes[proba[1].argmax()] == "Sports & Gaming"


@pytest.mark.skipif(not MODELS_READY, reason="classical models not trained")
def test_event_type_model():
    from src.classical_model import EventTypeClassifier

    et = EventTypeClassifier.load()
    proba = et.predict_proba(["Strong earthquake and aftershocks, buildings collapsed", "Wildfire spreads, homes burned"])
    assert et.classes[proba[0].argmax()] == "Earthquake"
    assert et.classes[proba[1].argmax()] == "Wildfire"
