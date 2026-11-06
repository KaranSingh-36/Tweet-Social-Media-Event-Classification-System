"""Live single-tweet classifier."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import ui
from src import visualization as viz
from src.prediction import MODEL_DISPLAY_NAMES, Prediction, log_predictions
from src.preprocessing import preprocessing_steps

ui.page_header("Workspace", "Live Tweet Classifier",
               "Type or paste any tweet / social-media post. The selected model predicts its event category, "
               "the probability of every category and the words that drove the decision.")

service = ui.require_models()
available = service.available_models()
df = ui.get_dataset() if settings.DATASET_FILE.exists() else pd.DataFrame()

EXAMPLES = {
    "Earthquake report": "Massive earthquake reported in Japan. Rescue teams are searching for survivors.",
    "Cricket final": "India wins the cricket match after an incredible final over!",
    "Product launch": "Apple just unveiled its new AI chip, claiming twice the performance of last year's processor.",
    "Election news": "Parliament votes on the new citizenship bill today as protests continue outside.",
    "Album release": "Taylor Swift's new album dropped at midnight and the music videos are incredible",
}

if "live_text" not in st.session_state:
    st.session_state.live_text = EXAMPLES["Earthquake report"]
    st.session_state.live_truth = None


def _set_example(text: str, truth: str | None = None) -> None:
    st.session_state.live_text = text
    st.session_state.live_truth = truth


left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.container(border=True):
        st.text_area("Tweet text", key="live_text", height=150, max_chars=1000,
                     placeholder="e.g. Flood waters rising in Kerala, families evacuated to relief camps #KeralaFloods")
        def _pick_example() -> None:
            choice = st.session_state.get("live_example")
            if choice:
                _set_example(EXAMPLES[choice])

        st.pills("Try an example", list(EXAMPLES), key="live_example", on_change=_pick_example)
        if not df.empty:
            test = df[df["split"] == "test"]

            def _random_test() -> None:
                row = test.sample(1).iloc[0]
                _set_example(row["text_original"], row["label"])
                st.session_state.live_example = None

            st.button("Load a random tweet from the test set", on_click=_random_test, icon=":material/casino:",
                           help="Load a random real tweet from the held-out test set (its true label is shown).")
        if st.session_state.get("live_truth") and st.session_state.live_text:
            st.html(f'<div style="font-size:.82rem;color:var(--muted);margin-top:.3rem">Ground-truth label of this test tweet: '
                    f'{ui.category_badge(st.session_state.live_truth)}</div>')

        options = [k for k in ("svm", "distilbert", "logreg") if k in available]
        labels = {k: MODEL_DISPLAY_NAMES[k] for k in options}
        if len(options) > 1:
            options.append("compare")
            labels["compare"] = "Compare models"
        mode = st.segmented_control("Model", options, format_func=lambda k: labels[k], default=options[0],
                                    key="live_model", width="stretch")
        if "distilbert" not in available:
            st.caption("DistilBERT appears here once `python -m src.transformer_model` has finished training.")
        analyze = st.button("Analyze tweet", type="primary", width="stretch", icon=":material/auto_awesome:")


def render_prediction(pred: Prediction, compact: bool = False) -> None:
    color = viz.category_color(pred.category)
    event_html = ""
    if pred.event_type:
        others = " · ".join(f"{c} {p:.0%}" for c, p in pred.event_type_probabilities[1:3])
        event_html = (f'<div class="event-type">Disaster event type: <b>{ui.esc(pred.event_type)}</b> '
                      f'({pred.event_type_confidence:.1%})<br><span style="color:var(--muted);font-size:.78rem">'
                      f'Other types: {ui.esc(others)} · HumAID event-type model</span></div>')
    runner_up = pred.top_categories[1] if len(pred.top_categories) > 1 else None
    st.html(f"""
    <div class="result-card" style="--cat:{color}">
      <div class="model">{ui.esc(pred.model_name)}</div>
      <div class="category"><span class="swatch"></span>{ui.esc(pred.category)}</div>
      <div class="desc">{ui.esc(settings.CATEGORY_DESCRIPTIONS.get(pred.category, ''))}</div>
      <div class="conf-track"><div class="conf-fill" style="width:{pred.confidence * 100:.1f}%"></div></div>
      <div class="conf-row"><span>Confidence (model probability)</span><b>{pred.confidence:.1%}</b></div>
      <div class="result-meta">
        {f'<span class="pill">Runner-up: <b>{ui.esc(runner_up[0])}</b> {runner_up[1]:.1%}</span>' if runner_up else ''}
        <span class="pill">Latency <b>{pred.latency_ms:.0f} ms</b></span>
      </div>
      {event_html}
    </div>""")
    top_n = 5 if compact else 9
    st.markdown("**Top predicted categories**")
    ui.plot(viz.probability_bar(pred.top_categories[:top_n], height=60 + 34 * min(top_n, len(pred.top_categories))),
            key=f"prob_{pred.model_key}")
    method = "occlusion: probability drop when the word is removed" if pred.model_key == "distilbert" \
        else "TF-IDF weight × model coefficient"
    st.markdown("**Important keywords**")
    if pred.keywords:
        chips = "".join(f'<span class="kw-chip">{ui.esc(k["term"])}</span>' for k in pred.keywords)
        st.html(f'<div class="kw-wrap">{chips}</div>')
        if not compact:
            ui.plot(viz.keyword_importance_bar(pred.keywords, color, height=40 + 28 * len(pred.keywords), title=method),
                    key=f"kw_{pred.model_key}")
        st.caption(f"Keyword importance method: {method}.")
    else:
        st.caption("No vocabulary words found; the prediction relies on the model prior.")


if analyze:
    text = st.session_state.live_text.strip()
    if not text:
        st.session_state.pop("live_results", None)
        with right:
            st.warning("Please enter a tweet to analyse.", icon=":material/edit_note:")
    else:
        keys = [k for k in ("svm", "logreg", "distilbert") if k in available] if mode == "compare" else [mode or options[0]]
        results = []
        with right, st.spinner("Running the model(s) …"):
            for k in keys:
                try:
                    results.append(service.predict(text, k))
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    st.error(f"{MODEL_DISPLAY_NAMES[k]} failed: {exc}")
        if results:
            st.session_state.live_results = results
            log_predictions([{"model": r.model_name, "category": r.category, "confidence": r.confidence, "text": r.text}
                             for r in results], mode="live")

results = st.session_state.get("live_results")
with right:
    if not results:
        ui.empty_state("Ready when you are", "Enter a tweet on the left, choose a model and press “Analyze tweet”.", "⚡")
    elif len(results) == 1:
        render_prediction(results[0])
    else:
        agree = len({r.category for r in results}) == 1
        if agree:
            st.success(f"All {len(results)} models agree: **{results[0].category}**", icon=":material/check_circle:")
        else:
            st.warning("The models disagree on this tweet — see the side-by-side comparison below.",
                       icon=":material/compare_arrows:")
        summary = pd.DataFrame([{"Model": r.model_name, "Prediction": r.category, "Confidence": r.confidence,
                                 "Latency (ms)": round(r.latency_ms, 1)} for r in results])
        st.dataframe(summary, hide_index=True, width="stretch", column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "Prediction": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent", width="small"),
            "Latency (ms)": st.column_config.NumberColumn("ms", width="small")})

if results and len(results) > 1:
    ui.section("Side-by-side comparison", "Each model's own probabilities and explanation for the same tweet.")
    cols = st.columns(len(results), gap="medium")
    for col, r in zip(cols, results):
        with col:
            render_prediction(r, compact=True)

if results:
    with left, st.expander("How this tweet was preprocessed", icon=":material/account_tree:", expanded=True):
        steps = preprocessing_steps(results[0].text)
        st.dataframe(pd.DataFrame(steps, columns=["Step", "Output"]), hide_index=True, width="stretch",
                     column_config={"Output": st.column_config.TextColumn(width="large")})
        st.caption("DistilBERT receives the lightly-normalised text (step 5); the TF-IDF models receive the fully "
                   "processed tokens (last step).")
