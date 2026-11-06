"""Bulk CSV classification."""
from __future__ import annotations

import streamlit as st

from config import settings
from src import ui
from src import visualization as viz
from src.bulk_prediction import BulkInputError, classify_dataframe, guess_text_column, read_csv_upload, to_csv_bytes, validate_text_column
from src.prediction import MODEL_DISPLAY_NAMES, log_predictions

ui.page_header("Workspace", "Bulk Tweet Analysis",
               "Upload a CSV of tweets, pick the text column and classify every row in batches. "
               "Results can be explored and downloaded as a new CSV.")

service = ui.require_models()
available = service.available_models()
SAMPLE = settings.SAMPLES_DIR / "sample_test_tweets.csv"

# ----------------------------------------------------------------------------- upload
with st.container(border=True):
    up_col, info_col = st.columns([1.6, 1], gap="large")
    with up_col:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                    help=f"Up to {settings.MAX_BULK_ROWS:,} rows are processed. UTF-8 or Latin-1 encoded.")
    with info_col:
        st.markdown("**No file at hand?**")
        st.caption("Use real held-out test tweets (225 rows, 25 per category, with their true labels).")
        if SAMPLE.exists():
            use_sample = st.button("Load sample test tweets", icon=":material/dataset:", width="stretch")
            st.download_button("Download sample CSV", SAMPLE.read_bytes(), "sample_test_tweets.csv", "text/csv",
                               icon=":material/download:", width="stretch")
        else:
            use_sample = False

if use_sample:
    data, warns = read_csv_upload(SAMPLE.read_bytes())
    st.session_state.update(bulk_df=data, bulk_warnings=warns, bulk_source="sample", bulk_name=SAMPLE.name)
    st.session_state.pop("bulk_result", None)
elif uploaded is not None:
    upload_id = f"upload:{uploaded.name}:{uploaded.size}"
    if st.session_state.get("bulk_upload_id") != upload_id:  # only (re)load a newly uploaded file
        st.session_state.bulk_upload_id = upload_id
        try:
            data, warns = read_csv_upload(uploaded)
            st.session_state.update(bulk_df=data, bulk_warnings=warns, bulk_source=upload_id, bulk_name=uploaded.name)
            st.session_state.pop("bulk_result", None)
        except BulkInputError as exc:
            st.session_state.pop("bulk_df", None)
            st.error(str(exc), icon=":material/error:")
            st.stop()

data = st.session_state.get("bulk_df")
if data is None:
    ui.empty_state("Upload a CSV to get started",
                   "The file needs one column containing tweet text (e.g. “tweet”, “text” or “content”). "
                   "Other columns are kept and returned in the output.", "📄")
    st.stop()

for w in st.session_state.get("bulk_warnings", []):
    st.warning(w, icon=":material/warning:")

# ----------------------------------------------------------------------------- configure
ui.section("Configure", f"File: {st.session_state.get('bulk_name', '')} · {len(data):,} rows · {len(data.columns)} columns")
c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
columns = list(data.columns)
with c1:
    text_col = st.selectbox("Tweet text column", columns, index=columns.index(guess_text_column(data)))
with c2:
    model_opts = [k for k in ("svm", "logreg", "distilbert") if k in available]
    labels = {k: MODEL_DISPLAY_NAMES[k] for k in model_opts}
    if len(model_opts) > 1:
        model_opts.append("compare")
        labels["compare"] = "Compare SVM vs DistilBERT" if "distilbert" in available else "Compare SVM vs LogReg"
    model_choice = st.selectbox("Model", model_opts, format_func=lambda k: labels[k])
with c3:
    truth_opts = ["(none)"] + [c for c in columns if c != text_col]
    default_truth = truth_opts.index("true_category") if "true_category" in truth_opts else 0
    truth_col = st.selectbox("Ground-truth column (optional)", truth_opts, index=default_truth,
                             help="If your file already has labels, accuracy is computed against them.")

try:
    validation = validate_text_column(data, text_col)
except BulkInputError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

ui.metric_cards([
    {"label": "Rows", "value": f"{validation['rows']:,}", "accent": "#38BDF8"},
    {"label": "Valid tweets", "value": f"{validation['valid']:,}", "accent": "#22C55E"},
    {"label": "Missing / empty", "value": f"{validation['empty']:,}", "sub": "Kept and flagged as skipped", "accent": "#F97316"},
])
if validation["looks_numeric"]:
    st.warning(f"Column “{text_col}” looks numeric. Are you sure it contains tweet text?", icon=":material/warning:")
with st.expander("Preview uploaded data", icon=":material/table_view:"):
    st.dataframe(data.head(20), width="stretch", hide_index=True)

if model_choice == "compare":
    model_keys = ["svm", "distilbert"] if "distilbert" in available else ["svm", "logreg"]
else:
    model_keys = [model_choice]
if "distilbert" in model_keys and validation["valid"] > 2000:
    st.info(f"DistilBERT on CPU processes roughly 50–150 tweets per second; {validation['valid']:,} tweets may take a "
            "minute or more. TF-IDF + SVM is near-instant.", icon=":material/schedule:")

if st.button("Classify tweets", type="primary", icon=":material/play_arrow:"):
    bar = st.progress(0.0, text="Starting …")
    try:
        result = classify_dataframe(service, data, text_col, model_keys,
                                    progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
    except BulkInputError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()
    bar.empty()
    st.session_state.bulk_result = {"df": result, "models": model_keys, "text_col": text_col}
    pred_col = "predicted_category" if len(model_keys) == 1 else f"predicted_category_{model_keys[0]}"
    conf_col = "confidence" if len(model_keys) == 1 else f"confidence_{model_keys[0]}"
    ok = result[result["status"] == "classified"]
    log_predictions([{"model": MODEL_DISPLAY_NAMES[model_keys[0]], "category": c, "confidence": p, "text": t}
                     for c, p, t in zip(ok[pred_col], ok[conf_col], ok[text_col])], mode="bulk")
    st.toast(f"Classified {len(ok):,} tweets", icon=":material/check_circle:")

# ----------------------------------------------------------------------------- results
res = st.session_state.get("bulk_result")
if not res:
    st.stop()
result, model_keys = res["df"], res["models"]
primary = model_keys[0]
pred_col = "predicted_category" if len(model_keys) == 1 else f"predicted_category_{primary}"
conf_col = "confidence" if len(model_keys) == 1 else f"confidence_{primary}"
ok = result[result["status"] == "classified"]

st.divider()
ui.section("Results", f"Primary model: {MODEL_DISPLAY_NAMES[primary]}")
cards = [
    {"label": "Classified", "value": f"{len(ok):,}", "accent": "#22C55E"},
    {"label": "Skipped (empty)", "value": f"{(result['status'] != 'classified').sum():,}", "accent": "#F97316"},
    {"label": "Mean confidence", "value": f"{ok[conf_col].mean():.1%}" if len(ok) else "—", "accent": "#38BDF8"},
    {"label": "Top category", "value": ok[pred_col].mode().iat[0] if len(ok) else "—", "small": True, "accent": "#A78BFA"},
]
if truth_col != "(none)" and truth_col in result.columns and len(ok):
    acc = (ok[pred_col].astype(str).str.strip() == ok[truth_col].astype(str).str.strip()).mean()
    cards.append({"label": "Accuracy vs ground truth", "value": f"{acc:.1%}", "sub": f"column “{truth_col}”", "accent": "#EAB308"})
if len(model_keys) > 1 and "models_agree" in ok:
    cards.append({"label": "Model agreement", "value": f"{ok['models_agree'].mean():.1%}", "accent": "#EC4899"})
ui.metric_cards(cards)
if st.session_state.get("bulk_source") == "sample" and len(cards) > 4:
    st.caption("The sample file is class-balanced (25 tweets per category), so its accuracy is close to the models' "
               "macro recall rather than the overall test accuracy, which is dominated by the large categories.")

g1, g2 = st.columns([1.2, 1], gap="large")
with g1:
    st.markdown("**Predicted category distribution**")
    if len(ok):
        ui.plot(viz.category_bar(ok[pred_col].value_counts(), height=360), key="bulk_dist")
with g2:
    st.markdown("**Prediction confidence distribution**")
    if len(ok):
        ui.plot(viz.confidence_values_histogram(ok[conf_col], height=360), key="bulk_conf")

ui.section("Classified tweets")
f1, f2 = st.columns([2, 1])
cats = sorted(ok[pred_col].unique()) if len(ok) else []
chosen = f1.multiselect("Filter by predicted category", cats, placeholder="All categories")
min_conf = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
view = result
if chosen:
    view = view[view[pred_col].isin(chosen)]
if min_conf > 0:
    view = view[view[conf_col].fillna(0) >= min_conf]

conf_cols = [c for c in view.columns if c.startswith("confidence")]
st.dataframe(view, width="stretch", hide_index=True, height=420, column_config={
    **{c: st.column_config.ProgressColumn(c.replace("_", " ").title(), min_value=0.0, max_value=1.0, format="percent") for c in conf_cols},
    res["text_col"]: st.column_config.TextColumn(width="large"),
})
st.caption(f"Showing {len(view):,} of {len(result):,} rows.")

name = st.session_state.get("bulk_name", "tweets.csv").rsplit(".", 1)[0]
st.download_button("Download classified CSV", to_csv_bytes(result), f"{name}_classified.csv", "text/csv",
                   type="primary", icon=":material/download:")
