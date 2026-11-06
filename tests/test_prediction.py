"""Tests for the single-tweet prediction service."""
import pytest

from config import settings

pytestmark = pytest.mark.skipif(not settings.SVM_FILE.exists(), reason="classical models not trained")


@pytest.fixture(scope="module")
def service():
    from src.prediction import EventClassifierService

    return EventClassifierService()


def test_predict_returns_real_probabilities(service):
    pred = service.predict("Massive earthquake reported in Japan. Rescue teams are searching for survivors.", "svm")
    assert pred.category == "Natural Disaster"
    assert 0 < pred.confidence <= 1
    assert abs(sum(p for _, p in pred.top_categories) - 1) < 1e-6
    assert pred.top_categories[0] == (pred.category, pred.confidence)
    assert pred.event_type == "Earthquake"
    assert any(k["term"].startswith("earthquake") for k in pred.keywords)


def test_sports_example(service):
    assert service.predict("India wins the cricket match after an incredible final over", "svm").category == "Sports & Gaming"


def test_empty_text_rejected(service):
    with pytest.raises(ValueError):
        service.predict("   ", "svm")


def test_unknown_model_rejected(service):
    with pytest.raises(ValueError):
        service.get_model("bert-large")


def test_non_disaster_has_no_event_type(service):
    assert service.predict("Great goal in the football match tonight", "svm").event_type is None


@pytest.mark.skipif(not (settings.TRANSFORMER_FINAL_DIR / "config.json").exists(), reason="DistilBERT not trained")
def test_distilbert_prediction(service):
    pred = service.predict("Massive earthquake reported in Japan. Rescue teams are searching for survivors.", "distilbert")
    assert pred.category == "Natural Disaster"
    assert 0 < pred.confidence <= 1
    assert pred.keywords  # occlusion explanation


def test_prediction_log_roundtrip(tmp_path, monkeypatch):
    from src import prediction

    monkeypatch.setattr(settings, "PREDICTION_LOG_FILE", tmp_path / "log.csv")
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    prediction.log_predictions([{"model": "svm", "category": "Business", "confidence": 0.5, "text": "x"}], "live")
    log = prediction.load_prediction_log()
    assert len(log) == 1 and log.iloc[0]["category"] == "Business"
