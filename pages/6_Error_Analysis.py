"""Error analysis on real test-set predictions."""
from __future__ import annotations

import streamlit as st

from config import settings
from src import ui
from src import visualization as viz
from src.error_analysis import (
    available_models,
    correctly_classified,
    error_by_length,
    error_by_topic_count,
    error_rate_by_category,
    misclassified,
    model_disagreements,
    reason_summary,
    top_confusions,
)
from src.prediction import MODEL_DISPLAY_NAMES

ui.page_header("Models", "Error Analysis",
               "Where and why the models fail — using real misclassified tweets from the held-out test set.")

ui.require_metrics()
preds = ui.get_test_predictions()
if preds.empty:
    ui.empty_state("No test predictions", "Run the evaluation to generate per-tweet predictions.", "🧪", "python -m src.evaluation")
    st.stop()

keys = available_models(preds)
c0 = st.columns([1, 2])[0]
key = c0.selectbox("Model", keys, index=keys.index("distilbert") if "distilbert" in keys else 0,
                   format_func=lambda k: MODEL_DISPLAY_NAMES[k], key="err_model")


@st.cache_data(show_spinner="Analysing errors …")
def _wrong(model_key: str, predictions_mtime: float):
    return misclassified(preds, model_key)


wrong = _wrong(key, settings.TEST_PREDICTIONS_FILE.stat().st_mtime)
conf = f"conf_{key}"
high_conf_wrong = int((wrong[conf] >= 0.8).sum())
ui.metric_cards([
    {"label": "Misclassified tweets", "value": f"{len(wrong):,}", "sub": f"of {len(preds):,} test tweets", "accent": "#EF4444"},
    {"label": "Error rate", "value": f"{len(wrong) / len(preds):.1%}", "accent": "#F97316"},
    {"label": "Mean confidence on errors", "value": f"{wrong[conf].mean():.1%}" if len(wrong) else "—", "accent": "#EAB308"},
    {"label": "Confident mistakes (≥80%)", "value": f"{high_conf_wrong:,}", "sub": "often label noise or multi-topic", "accent": "#A78BFA"},
])

tab_overview, tab_wrong, tab_right, tab_disagree = st.tabs(
    ["Error patterns", "Incorrect predictions", "Correct predictions", "Model disagreements"])

with tab_overview:
    a, b = st.columns(2, gap="large")
    with a:
        ui.section("Error distribution by category", "Share of each actual category that was misclassified.")
        ui.plot(viz.error_rate_bar(error_rate_by_category(preds, key), height=400), key="er_cat")
    with b:
        ui.section("Most frequent confusions", "Actual → predicted pairs.")
        conf_pairs = top_confusions(preds, key, 10)
        st.dataframe(conf_pairs, hide_index=True, width="stretch", height=400, column_config={
            "actual": "Actual", "predicted": "Predicted", "count": st.column_config.NumberColumn("Tweets"),
            "share_of_actual_class": st.column_config.NumberColumn("% of class", format="percent", width="small"),
            "related": st.column_config.TextColumn("Why they overlap", width="medium")})
    c, d = st.columns([1.15, 1], gap="large")
    with c:
        ui.section("Possible causes", "Heuristic indicators found in the misclassified tweets (a tweet can have several).")
        rs = reason_summary(wrong)
        if not rs.empty:
            ui.plot(viz.keyword_bar(rs, word_col="reason", height=380, value_title="Misclassified tweets", color="#F472B6"), key="er_reason")
    with d:
        ui.section("Error rate vs tweet length", "Tokens after preprocessing.")
        ui.plot(viz.simple_rate_bar(error_by_length(preds, key), "length_bucket", height=380), key="er_len")
    e, f = st.columns([1, 1.15], gap="large")
    with e:
        ui.section("Error rate vs number of human topics", "TweetTopic tweets that annotators gave 2+ topics are harder.")
        ui.plot(viz.simple_rate_bar(error_by_topic_count(preds, key), "topics", height=380), key="er_topics")
    with f:
        ui.section("Interpretation")
        st.html("""<div class="card"><h4>Why do the models make mistakes?</h4><ul>
      <li><b>Overlapping categories</b> – a tweet about a celebrity's charity work is both Entertainment and News; a hurricane
      tweet from a news outlet is both Natural Disaster and News. Most confusions are between such related pairs.</li>
      <li><b>Multi-topic tweets</b> – annotators gave many tweets two topics; the single primary label we keep is only one
      valid answer, so some “errors” are actually the other correct topic.</li>
      <li><b>Short or context-free tweets</b> – “Can't wait for tonight!!” carries almost no topical signal.</li>
      <li><b>Hashtags, names and slang</b> – many tokens (artist names, team nicknames, K-pop fandom terms) are rare or
      unseen; TF-IDF ignores unseen words, DistilBERT splits them into sub-words.</li>
      <li><b>Sarcasm and humour</b> – literal words point to one topic while the intent is another.</li>
      <li><b>Small classes</b> – Education, Health, Science &amp; Technology and Business have only 260–370 tweets in total,
      so their decision boundaries are learned from few examples.</li></ul></div>""")

with tab_wrong:
    f1, f2, f3 = st.columns([1, 1, 1])
    actual_sel = f1.selectbox("Actual category", ["All"] + sorted(wrong["label"].unique()), key="w_actual")
    pred_sel = f2.selectbox("Predicted category", ["All"] + sorted(wrong[f"pred_{key}"].unique()), key="w_pred")
    order = f3.selectbox("Sort by", ["Most confident mistakes", "Least confident mistakes"], key="w_sort")
    view = wrong
    if actual_sel != "All":
        view = view[view["label"] == actual_sel]
    if pred_sel != "All":
        view = view[view[f"pred_{key}"] == pred_sel]
    view = view.sort_values(conf, ascending=order.startswith("Least"))
    st.caption(f"{len(view):,} misclassified tweets match the filters (showing up to 30).")
    if view.empty:
        ui.empty_state("No errors for this combination", "Pick another actual / predicted pair.", "✅")
    for row in view.head(30).itertuples():
        badges = (f'<span>Actual</span>{ui.category_badge(row.label)}<span>→ Predicted</span>'
                  f'{ui.category_badge(getattr(row, f"pred_{key}"))}<span class="pill">confidence <b>{getattr(row, conf):.1%}</b></span>'
                  f'<span class="pill">P(actual) <b>{getattr(row, f"true_prob_{key}"):.1%}</b></span>'
                  f'<span class="pill">{ui.esc(row.source)}</span>')
        st.html(ui.tweet_card(row.text_original, badges, row.possible_reasons))
    with st.expander("Full table of misclassified tweets", icon=":material/table_view:"):
        tbl = view[["text_original", "label", f"pred_{key}", conf, "original_labels", "possible_reasons"]].copy()
        tbl["possible_reasons"] = tbl["possible_reasons"].apply(lambda r: " | ".join(r))
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            "text_original": st.column_config.TextColumn("Tweet", width="large"), "label": "Actual",
            f"pred_{key}": "Predicted", conf: st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent"),
            "original_labels": "Original dataset labels"})

with tab_right:
    right = correctly_classified(preds, key)
    cat = st.selectbox("Category", ["All"] + sorted(right["label"].unique()), key="r_cat")
    if cat != "All":
        right = right[right["label"] == cat]
    st.caption(f"{len(right):,} correctly classified tweets (showing 15 most confident and 5 least confident).")
    for row in list(right.head(15).itertuples()) + list(right.tail(5).itertuples()):
        badges = f'{ui.category_badge(row.label)}<span class="pill">confidence <b>{getattr(row, conf):.1%}</b></span>'
        st.html(ui.tweet_card(row.text_original, badges))

with tab_disagree:
    if len(keys) < 2:
        ui.empty_state("Only one model evaluated", "Train and evaluate another model to compare disagreements.", "⚖️")
    else:
        d1, d2 = st.columns(2)
        a_key = d1.selectbox("Model A", keys, index=0, format_func=lambda k: MODEL_DISPLAY_NAMES[k], key="dis_a")
        b_opts = [k for k in keys if k != a_key]
        b_key = d2.selectbox("Model B", b_opts, index=len(b_opts) - 1, format_func=lambda k: MODEL_DISPLAY_NAMES[k], key="dis_b")
        dis = model_disagreements(preds, a_key, b_key)
        ui.metric_cards([
            {"label": "Tweets where models disagree", "value": f"{len(dis):,}", "sub": f"{len(dis) / len(preds):.1%} of test set", "accent": "#38BDF8"},
            {"label": f"{MODEL_DISPLAY_NAMES[a_key]} right", "value": f"{(dis['winner'] == a_key).sum():,}", "accent": viz.MODEL_COLORS[a_key]},
            {"label": f"{MODEL_DISPLAY_NAMES[b_key]} right", "value": f"{(dis['winner'] == b_key).sum():,}", "accent": viz.MODEL_COLORS[b_key]},
            {"label": "Neither right", "value": f"{(dis['winner'] == 'neither').sum():,}", "accent": "#EF4444"},
        ])
        ui.plot(viz.disagreement_sankey(preds, a_key, b_key, height=320), key="dis_bar")
        who = st.segmented_control("Show examples where", ["a", "b", "neither"], default="b", key="dis_who",
                                   format_func=lambda w: {"a": f"only {MODEL_DISPLAY_NAMES[a_key]} is right",
                                                          "b": f"only {MODEL_DISPLAY_NAMES[b_key]} is right",
                                                          "neither": "both are wrong"}[w]) or "b"
        target = {"a": a_key, "b": b_key, "neither": "neither"}[who]
        ex = dis[dis["winner"] == target].head(15)
        if ex.empty:
            ui.empty_state("No examples", "No disagreements of this kind in the test set.", "⚖️")
        for row in ex.itertuples():
            badges = (f'<span>Actual</span>{ui.category_badge(row.label)}'
                      f'<span>· {ui.esc(MODEL_DISPLAY_NAMES[a_key])}</span>{ui.category_badge(getattr(row, f"pred_{a_key}"))}'
                      f'<span>· {ui.esc(MODEL_DISPLAY_NAMES[b_key])}</span>{ui.category_badge(getattr(row, f"pred_{b_key}"))}')
            st.html(ui.tweet_card(row.text_original, badges))
