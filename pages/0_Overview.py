"""Overview dashboard (landing page)."""
from __future__ import annotations

import streamlit as st

from config import settings
from src import ui
from src import visualization as viz
from src.prediction import load_prediction_log

df = ui.require_dataset()
report = ui.get_data_report()
metrics = ui.get_metrics()
log = load_prediction_log()

counts = df["label"].value_counts()
n_models = sum([settings.SVM_FILE.exists(), settings.LOGREG_FILE.exists(),
                (settings.TRANSFORMER_FINAL_DIR / "config.json").exists()])

st.html(f"""
<div class="hero">
  <span class="eyebrow">● NLP capstone · ETMMTI303</span>
  <h1>Intelligent <span class="grad">Tweet Event</span> Classification</h1>
  <p>Analyse tweets and social-media posts and classify them into real-world event categories with
  classical machine learning (TF-IDF + SVM / Logistic Regression) and a fine-tuned DistilBERT transformer,
  trained on {len(df):,} human-annotated public tweets.</p>
  <div class="badges">
    <span class="pill"><b>{df['label'].nunique()}</b> event categories</span>
    <span class="pill"><b>{n_models}</b> trained classifiers</span>
    <span class="pill">Sources: <b>TweetTopic</b> + <b>HumAID</b></span>
    <span class="pill">CPU-friendly inference</span>
  </div>
</div>
""")

best = None
if metrics.get("models"):
    best_key = max(metrics["models"], key=lambda k: metrics["models"][k]["f1_macro"])
    best = metrics["models"][best_key]

ui.metric_cards([
    {"label": "Tweets in dataset", "value": f"{len(df):,}", "sub": f"{(df['split'] == 'test').sum():,} held out for testing", "accent": "#38BDF8"},
    {"label": "Event categories", "value": df["label"].nunique(), "sub": f"Largest: {counts.index[0]}", "accent": "#A78BFA"},
    {"label": "Predictions made", "value": f"{len(log):,}", "sub": "Live + bulk, logged locally", "accent": "#22C55E"},
    {"label": "ML models available", "value": f"{n_models} / 3", "sub": "SVM · LogReg · DistilBERT", "accent": "#F97316"},
    {"label": "Best macro-F1 (test)", "value": f"{best['f1_macro']:.3f}" if best else "—",
     "sub": best["display_name"] if best else "Run evaluation", "accent": "#EAB308"},
])

left, right = st.columns([1.15, 1], gap="large")
with left:
    ui.section("Dataset category distribution", "Number of labelled tweets per event category after cleaning.")
    ui.plot(viz.category_bar(counts, height=380), key="ov_bar")
with right:
    ui.section("Share of tweets", "Class imbalance is handled with class-weighted training.")
    ui.plot(viz.category_donut(counts, height=380, center_text=f"{len(df):,}<br>tweets"), key="ov_donut")

ui.section("End-to-end NLP pipeline")
st.html("""
<div class="pipeline">
  <div class="step"><div class="n">01 · DATA</div><div class="t">Collection</div><div class="d">TweetTopic + HumAID public corpora, label mapping, de-duplication</div></div>
  <div class="step"><div class="n">02 · CLEAN</div><div class="t">Preprocessing</div><div class="d">URLs, mentions, hashtags, emojis, stop words, lemmatisation</div></div>
  <div class="step"><div class="n">03 · REPRESENT</div><div class="t">Features</div><div class="d">TF-IDF uni+bigrams vs. contextual DistilBERT embeddings</div></div>
  <div class="step"><div class="n">04 · LEARN</div><div class="t">Models</div><div class="d">Linear SVM (calibrated), Logistic Regression, fine-tuned DistilBERT</div></div>
  <div class="step"><div class="n">05 · MEASURE</div><div class="t">Evaluation</div><div class="d">Same held-out test set, macro/weighted F1, confusion matrix</div></div>
  <div class="step"><div class="n">06 · SERVE</div><div class="t">Application</div><div class="d">Live &amp; bulk classification, analytics and error analysis</div></div>
</div>
""")

c1, c2 = st.columns([1.15, 1], gap="large")
with c1:
    ui.section("Recent prediction activity", "Latest predictions from the Live Classifier and Bulk Analysis pages.")
    if log.empty:
        ui.empty_state("No predictions yet", "Classify a tweet on the Live Classifier page and it will appear here.", "⚡")
    else:
        recent = log.sort_values("timestamp", ascending=False).head(8)
        st.dataframe(
            recent[["timestamp", "mode", "model", "category", "confidence", "text"]],
            hide_index=True, width="stretch",
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM, HH:mm"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent"),
                "text": st.column_config.TextColumn("Tweet", width="large"),
            },
        )
        if log["timestamp"].notna().any():
            ui.plot(viz.activity_timeline(log, height=220), key="ov_activity")
with c2:
    ui.section("Model leaderboard", "Measured on the same unseen test split.")
    if metrics.get("models"):
        table = sorted(metrics["models"].items(), key=lambda kv: -kv[1]["f1_macro"])
        for rank, (key, m) in enumerate(table, 1):
            st.html(f"""
            <div class="card compact" style="display:flex;justify-content:space-between;align-items:center">
              <div><div style="font-size:.7rem;color:var(--faint);font-weight:700;letter-spacing:.08em">#{rank}</div>
                   <div style="font-weight:700">{ui.esc(m['display_name'])}</div>
                   <div style="font-size:.8rem;color:var(--muted)">Accuracy {m['accuracy']:.3f} · Weighted F1 {m['f1_weighted']:.3f}</div></div>
              <div style="text-align:right"><div style="font-size:1.5rem;font-weight:800;color:{viz.MODEL_COLORS.get(key)}">{m['f1_macro']:.3f}</div>
                   <div style="font-size:.72rem;color:var(--faint)">MACRO F1</div></div>
            </div>""")
    else:
        ui.empty_state("No evaluation yet", "Run the evaluation script to populate the leaderboard.", "🏁", "python -m src.evaluation")

    ui.section("Quick navigation")
    q1, q2 = st.columns(2)
    q1.page_link("pages/1_Live_Classifier.py", label="Classify a tweet", icon=":material/bolt:", width="stretch")
    q2.page_link("pages/2_Bulk_Analysis.py", label="Upload a CSV", icon=":material/upload_file:", width="stretch")
    q1.page_link("pages/3_Event_Analytics.py", label="Explore the data", icon=":material/insights:", width="stretch")
    q2.page_link("pages/4_Model_Comparison.py", label="Compare models", icon=":material/leaderboard:", width="stretch")
    q1.page_link("pages/5_Model_Evaluation.py", label="Evaluation report", icon=":material/fact_check:", width="stretch")
    q2.page_link("pages/6_Error_Analysis.py", label="Error analysis", icon=":material/troubleshoot:", width="stretch")

ui.section("Event categories", "Every category is backed by human annotations in the source datasets.")
cards_html = "".join(
    f"""<div class="card compact" style="margin:0">{ui.category_badge(cat)}
    <span style="float:right;font-weight:700">{counts[cat]:,}</span>
    <p style="margin:.45rem 0 0 0">{ui.esc(settings.CATEGORY_DESCRIPTIONS.get(cat, ''))}</p></div>"""
    for cat in counts.index)
st.html(f'<div class="cat-grid">{cards_html}</div>')

src_info = report.get("sources", {})
if src_info:
    ui.section("Data sources")
    s1, s2 = st.columns(2)
    tt, hd = src_info.get("TweetTopic", {}), src_info.get("HumAID", {})
    s1.html(f"""<div class="card"><h4>TweetTopic · cardiffnlp/tweet_topic_multi</h4>
      <p>{tt.get('unique_ids', 0):,} English tweets (2019–2021) annotated with 19 topics by crowd workers
      (Antypas et al., COLING 2022). Mapped to 8 coarse event categories.<br>
      <a href="{tt.get('url', '#')}" target="_blank">{tt.get('url', '')}</a></p></div>""")
    s2.html(f"""<div class="card"><h4>HumAID · QCRI/HumAID-events</h4>
      <p>{hd.get('raw_rows', 0):,} tweets from {hd.get('events', 19)} real natural disasters (2016–2019) with
      humanitarian annotations (Alam et al., ICWSM 2021). Provides the <i>Natural Disaster</i> category and the
      earthquake / hurricane / flood / wildfire event type.<br>
      <a href="{hd.get('url', '#')}" target="_blank">{hd.get('url', '')}</a></p></div>""")
