"""Detailed evaluation of one model on the held-out test set."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.evaluation import per_class_table

ui.page_header("Models", "Model Evaluation",
               "Confusion matrix, classification report and per-category precision, recall and F1 on the unseen test set.")

metrics = ui.require_metrics()
preds = ui.get_test_predictions()

top = st.columns([1, 2])
with top[0]:
    key = ui.model_selector(metrics, key="eval_model")
m = metrics["models"][key]
classes = metrics["classes"]

ui.metric_cards([
    {"label": "Accuracy", "value": f"{m['accuracy']:.3f}", "accent": "#38BDF8"},
    {"label": "Precision (macro)", "value": f"{m['precision_macro']:.3f}", "accent": "#A78BFA"},
    {"label": "Recall (macro)", "value": f"{m['recall_macro']:.3f}", "accent": "#22C55E"},
    {"label": "Macro F1", "value": f"{m['f1_macro']:.3f}", "accent": "#F97316"},
    {"label": "Weighted F1", "value": f"{m['f1_weighted']:.3f}", "accent": "#EAB308"},
])

cm_col, rep_col = st.columns([1.25, 1], gap="large")
with cm_col:
    ui.section("Confusion matrix", "Rows = actual category, columns = predicted category.")
    normalize = st.toggle("Normalise by actual category (row %)", value=True, key="cm_norm")
    ui.plot(viz.confusion_matrix_heatmap(m["confusion_matrix"], classes, normalize=normalize, height=580), key="ev_cm")
with rep_col:
    ui.section("Classification report")
    pc = per_class_table(metrics, key)
    st.dataframe(pc, hide_index=True, width="stretch", height=370, column_config={
        "Category": st.column_config.TextColumn(width="medium"),
        "Precision": st.column_config.NumberColumn("Prec.", format="%.3f", width="small"),
        "Recall": st.column_config.NumberColumn("Recall", format="%.3f", width="small"),
        "F1": st.column_config.NumberColumn("F1", format="%.3f", width="small"),
        "Support": st.column_config.NumberColumn("n", width="small", help="Number of test tweets")})
    rep = m["classification_report"]
    avg = pd.DataFrame([
        {"Average": "Macro", "Precision": rep["macro avg"]["precision"], "Recall": rep["macro avg"]["recall"], "F1": rep["macro avg"]["f1-score"]},
        {"Average": "Weighted", "Precision": rep["weighted avg"]["precision"], "Recall": rep["weighted avg"]["recall"], "F1": rep["weighted avg"]["f1-score"]},
    ])
    st.dataframe(avg, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["Precision", "Recall", "F1"]})
    weak = pc.sort_values("F1").head(3)
    st.html('<div class="card compact"><h4>Weakest categories</h4>' + "".join(
        f'<div style="display:flex;justify-content:space-between;margin:.25rem 0">{ui.category_badge(r.Category)}'
        f'<span style="color:var(--muted);font-size:.85rem">F1 {r.F1:.2f} · support {r.Support}</span></div>'
        for r in weak.itertuples()) + "</div>")

a, b = st.columns(2, gap="large")
with a:
    ui.section("Precision vs recall by category", "Bubble size = number of test tweets.")
    ui.plot(viz.precision_recall_scatter(pc, height=430), key="ev_pr")
with b:
    ui.section("F1-score by category")
    f1 = pc.set_index("Category")["F1"].sort_values()
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h", marker_color=[viz.category_color(c) for c in f1.index],
                           text=[f"{v:.2f}" for v in f1.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 1.1], title="F1-score")
    ui.plot(viz.style(fig, 430), key="ev_f1")

if not preds.empty and f"conf_{key}" in preds:
    ui.section("Prediction confidence distribution",
               "A well-behaved model is confident when right and unsure when wrong.")
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        ui.plot(viz.confidence_histogram(preds, key, height=360), key="ev_conf")
    with c2:
        correct = preds["label"] == preds[f"pred_{key}"]
        bins = pd.cut(preds[f"conf_{key}"], [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True,
                      labels=["below 50%", "50 – 70%", "70 – 90%", "90 – 100%"])
        cal = preds.assign(correct=correct, bin=bins).groupby("bin", observed=True).agg(
            tweets=("uid", "count"), accuracy=("correct", "mean")).reset_index()
        cal["bin"] = cal["bin"].astype(str)
        st.markdown("**Accuracy by confidence band**")
        st.dataframe(cal, hide_index=True, width="stretch", column_config={
            "bin": "Confidence", "tweets": "Tweets", "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.0, max_value=1.0, format="percent")})
        st.caption(f"Mean confidence when correct: {preds.loc[correct, f'conf_{key}'].mean():.1%} · "
                   f"when wrong: {preds.loc[~correct, f'conf_{key}'].mean():.1%}")

et = metrics.get("event_type_model")
if et:
    ui.section("Disaster event-type model", "TF-IDF + Logistic Regression trained on HumAID to tell disaster types apart.")
    rep = et["report"]
    types = [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg")]
    ui.metric_cards([
        {"label": "Accuracy", "value": f"{et['accuracy']:.3f}", "accent": "#F97316"},
        {"label": "Macro F1", "value": f"{et['macro_f1']:.3f}", "accent": "#F97316"},
        {"label": "Train / test tweets", "value": f"{et['train_size']:,} / {et['test_size']:,}", "small": True, "accent": "#F97316"},
    ] + [{"label": f"F1 · {t}", "value": f"{rep[t]['f1-score']:.3f}", "accent": "#FB923C"} for t in types], columns=4)
    cv = et.get("unseen_event_cv")
    if cv:
        st.info(f"**Generalisation to unseen disasters:** with {cv['n_splits']}-fold cross-validation grouped by event "
                f"(the test disasters are never seen in training) accuracy is **{cv['accuracy']:.3f}** and macro-F1 "
                f"**{cv['macro_f1']:.3f}**. The official-split score above is higher because train and test tweets come "
                "from the same 19 events.", icon=":material/science:")
    else:
        st.caption(et.get("note", ""))
