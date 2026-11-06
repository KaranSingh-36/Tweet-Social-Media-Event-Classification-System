"""Bulk (CSV) classification with validation and batch processing."""
from __future__ import annotations

import io
from typing import Callable

import numpy as np
import pandas as pd

from config import settings
from src.prediction import MODEL_DISPLAY_NAMES, EventClassifierService

TEXT_COLUMN_HINTS = ("tweet", "text", "content", "post", "message", "body", "tweet_text")


class BulkInputError(ValueError):
    """Raised for user-facing CSV problems."""


def read_csv_upload(data: bytes | io.BytesIO, max_rows: int = settings.MAX_BULK_ROWS) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV with encoding fallbacks. Returns (df, warnings)."""
    raw = data.getvalue() if hasattr(data, "getvalue") else data
    if not raw or not raw.strip():
        raise BulkInputError("The uploaded file is empty.")
    warnings: list[str] = []
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip", dtype=str, keep_default_na=False, na_values=[""])
            if enc == "latin-1":
                warnings.append("File was not valid UTF-8; decoded as Latin-1.")
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise BulkInputError(f"Could not parse the CSV file: {exc}") from exc
    if df is None or df.empty or len(df.columns) == 0:
        raise BulkInputError("No rows were found in the CSV file.")
    if len(df) > max_rows:
        warnings.append(f"File has {len(df):,} rows; only the first {max_rows:,} are processed.")
        df = df.head(max_rows)
    return df.reset_index(drop=True), warnings


def guess_text_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for hint in TEXT_COLUMN_HINTS:
        if hint in lower:
            return lower[hint]
    # Fall back to the column with the longest average string length.
    lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
    return max(lengths, key=lengths.get)


def validate_text_column(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise BulkInputError(f"Column '{column}' does not exist in the file.")
    text = df[column]
    empty = text.isna() | (text.astype(str).str.strip() == "")
    if empty.all():
        raise BulkInputError(f"Column '{column}' contains no text.")
    numeric_like = pd.to_numeric(text[~empty], errors="coerce").notna().mean()
    return {
        "rows": int(len(df)),
        "valid": int((~empty).sum()),
        "empty": int(empty.sum()),
        "looks_numeric": bool(numeric_like > 0.8),
    }


def classify_dataframe(
    service: EventClassifierService,
    df: pd.DataFrame,
    column: str,
    model_keys: list[str],
    batch_size: int = settings.BULK_BATCH_SIZE,
    progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Classify every row in batches. Empty rows are flagged, not dropped."""
    validate_text_column(df, column)
    out = df.copy()
    text = out[column].fillna("").astype(str)
    valid_mask = text.str.strip() != ""
    valid_idx = np.flatnonzero(valid_mask.to_numpy())
    valid_texts = text.iloc[valid_idx].tolist()
    out["status"] = np.where(valid_mask, "classified", "skipped: empty text")

    total_batches = max(1, int(np.ceil(len(valid_texts) / batch_size))) * len(model_keys)
    done = 0
    for key in model_keys:
        name = MODEL_DISPLAY_NAMES[key]
        cats = np.full(len(out), "", dtype=object)
        confs = np.full(len(out), np.nan)
        for start in range(0, len(valid_texts), batch_size):
            batch = valid_texts[start : start + batch_size]
            proba, classes = service.predict_proba(batch, key, batch_size=batch_size)
            best = proba.argmax(1)
            rows = valid_idx[start : start + batch_size]
            cats[rows] = [classes[i] for i in best]
            confs[rows] = proba[np.arange(len(batch)), best]
            done += 1
            if progress:
                progress(done / total_batches, f"{name}: {min(start + batch_size, len(valid_texts)):,}/{len(valid_texts):,} tweets")
        suffix = "" if len(model_keys) == 1 else f"_{key}"
        out[f"predicted_category{suffix}"] = cats
        out[f"confidence{suffix}"] = np.round(confs, 4)

    # Disaster event type for rows predicted as Natural Disaster (first model).
    first = "predicted_category" if len(model_keys) == 1 else f"predicted_category_{model_keys[0]}"
    disaster_rows = np.flatnonzero((out[first] == settings.NATURAL_DISASTER).to_numpy())
    out["disaster_event_type"] = ""
    if len(disaster_rows) and service.event_type_model is not None:
        types, _ = service.predict_event_types(text.iloc[disaster_rows].tolist())
        out.loc[out.index[disaster_rows], "disaster_event_type"] = types
    if len(model_keys) > 1:
        cols = [f"predicted_category_{k}" for k in model_keys]
        agree = out[cols].nunique(axis=1) == 1
        out["models_agree"] = np.where(valid_mask, agree, False)
    if progress:
        progress(1.0, "Done")
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
