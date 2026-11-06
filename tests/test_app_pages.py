"""Smoke tests: every dashboard page renders without exceptions (headless)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import settings

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "pages").glob("*.py"))

pytestmark = pytest.mark.skipif(not settings.DATASET_FILE.exists(), reason="dataset not prepared")


@pytest.fixture(autouse=True)
def isolated_prediction_log(tmp_path, monkeypatch):
    """Keep test predictions out of the real logs/prediction_history.csv."""
    monkeypatch.setattr(settings, "PREDICTION_LOG_FILE", tmp_path / "prediction_history.csv")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def _page(name: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.switch_page(name)
    at.run()
    return at


@pytest.mark.skipif(not settings.SVM_FILE.exists(), reason="models not trained")
def test_live_classifier_analyze_button():
    at = _page("pages/1_Live_Classifier.py")
    at.text_area(key="live_text").set_value("Massive earthquake reported in Japan. Rescue teams are searching for survivors.")
    next(b for b in at.button if b.label == "Analyze tweet").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    results = at.session_state["live_results"]
    assert results[0].category == "Natural Disaster"


@pytest.mark.skipif(not settings.SVM_FILE.exists(), reason="models not trained")
def test_bulk_analysis_with_sample_file():
    at = _page("pages/2_Bulk_Analysis.py")
    next(b for b in at.button if b.label == "Load sample test tweets").click()
    at.run()
    next(b for b in at.button if b.label == "Classify tweets").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    result = at.session_state["bulk_result"]["df"]
    assert len(result) == 225 and (result["status"] == "classified").all()
