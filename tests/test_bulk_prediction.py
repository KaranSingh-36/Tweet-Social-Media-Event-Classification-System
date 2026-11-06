"""Tests for CSV bulk classification."""
import pandas as pd
import pytest

from config import settings
from src.bulk_prediction import BulkInputError, guess_text_column, read_csv_upload, to_csv_bytes, validate_text_column

CSV = b"id,tweet,other\n1,Massive earthquake hits Japan,a\n2,,b\n3,India wins the cricket final,c\n"


def test_read_csv_and_guess_column():
    df, warnings = read_csv_upload(CSV)
    assert len(df) == 3 and not warnings
    assert guess_text_column(df) == "tweet"


def test_latin1_fallback():
    df, warnings = read_csv_upload("text\nCaf\xe9 opens downtown\n".encode("latin-1"))
    assert df.iloc[0, 0].startswith("Caf") and warnings


def test_empty_file_rejected():
    with pytest.raises(BulkInputError):
        read_csv_upload(b"")


def test_row_cap():
    data = ("text\n" + "\n".join(f"tweet {i}" for i in range(50))).encode()
    df, warnings = read_csv_upload(data, max_rows=10)
    assert len(df) == 10 and warnings


def test_validation_counts_empty_rows():
    df, _ = read_csv_upload(CSV)
    v = validate_text_column(df, "tweet")
    assert v == {"rows": 3, "valid": 2, "empty": 1, "looks_numeric": False}
    with pytest.raises(BulkInputError):
        validate_text_column(df, "missing")
    with pytest.raises(BulkInputError):
        validate_text_column(pd.DataFrame({"t": [None, ""]}), "t")


@pytest.mark.skipif(not settings.SVM_FILE.exists(), reason="classical models not trained")
def test_classify_dataframe_batches_and_flags_empty_rows():
    from src.bulk_prediction import classify_dataframe
    from src.prediction import EventClassifierService

    df, _ = read_csv_upload(CSV)
    progress = []
    out = classify_dataframe(EventClassifierService(load_transformer=False), df, "tweet", ["svm"], batch_size=1,
                             progress=lambda f, m: progress.append(f))
    assert list(out["status"]) == ["classified", "skipped: empty text", "classified"]
    assert out.loc[0, "predicted_category"] == "Natural Disaster"
    assert out.loc[0, "disaster_event_type"] == "Earthquake"
    assert out.loc[2, "predicted_category"] == "Sports & Gaming"
    assert pd.isna(out.loc[1, "confidence"])
    assert progress[-1] == 1.0
    assert b"predicted_category" in to_csv_bytes(out)


@pytest.mark.skipif(not settings.SVM_FILE.exists(), reason="classical models not trained")
def test_compare_mode_adds_agreement_column():
    from src.bulk_prediction import classify_dataframe
    from src.prediction import EventClassifierService

    df, _ = read_csv_upload(CSV)
    out = classify_dataframe(EventClassifierService(load_transformer=False), df, "tweet", ["svm", "logreg"])
    assert {"predicted_category_svm", "predicted_category_logreg", "models_agree"} <= set(out.columns)
