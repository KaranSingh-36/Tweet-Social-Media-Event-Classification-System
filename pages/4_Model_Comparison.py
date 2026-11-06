"""Model comparison: TF-IDF + SVM vs TF-IDF + Logistic Regression vs DistilBERT."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.evaluation import comparison_table

ui.page_header("Models", "Model Comparison",
               "All models are evaluated on the same held-out test split that was never used for training or tuning.")

metrics = ui.require_metrics()
table = comparison_table(metrics)
history = ui.get_transformer_history()
classical = ui.get_classical_report()

best_f1 = table.loc[table["Macro F1"].idxmax()]
best_acc = table.loc[table["Accuracy"].idxmax()]
fastest = table.loc[table["Inference (ms/tweet)"].idxmin()]
ui.metric_cards([
    {"label": "Test tweets", "value": f"{metrics['test_size']:,}", "sub": f"{len(metrics['classes'])} categories", "accent": "#38BDF8"},
    {"label": "Best macro-F1", "value": f"{best_f1['Macro F1']:.3f}", "sub": best_f1["Model"], "accent": viz.MODEL_COLORS.get(best_f1["key"])},
    {"label": "Best accuracy", "value": f"{best_acc['Accuracy']:.3f}", "sub": best_acc["Model"], "accent": viz.MODEL_COLORS.get(best_acc["key"])},
    {"label": "Fastest inference", "value": f"{fastest['Inference (ms/tweet)']:.2f} ms", "sub": fastest["Model"], "accent": "#22C55E"},
])

ui.section("Metrics table")
shown = table.drop(columns="key")
styled = shown.style.highlight_max(subset=["Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1", "Weighted F1"],
                                   props="color:#22C55E;font-weight:700") \
    .highlight_min(subset=["Training time (s)", "Inference (ms/tweet)"], props="color:#22C55E;font-weight:700")
st.dataframe(
    styled, hide_index=True, width="stretch",
    column_config={
        "Model": st.column_config.TextColumn(width="medium"),
        **{m: st.column_config.NumberColumn(m, format="%.4f", width="small")
           for m in ["Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1", "Weighted F1"]},
        "Training time (s)": st.column_config.NumberColumn("Train (s)", format="%.1f", width="small"),
        "Inference (ms/tweet)": st.column_config.NumberColumn("Infer (ms/tweet)", format="%.3f", width="small"),
    },
)
st.caption("Best value per column in green. Macro averages weight every category equally (sensitive to small classes); weighted averages weight by "
           "support. Inference time = total time to classify the whole test set (including preprocessing) ÷ number of "
           "tweets, measured on this machine's CPU.")

ui.section("Performance comparison")
ui.plot(viz.metrics_grouped_bar(table, ["Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1", "Weighted F1"], height=420), key="cmp_bar")

a, b = st.columns(2, gap="large")
with a:
    ui.section("Training time", "Wall-clock seconds, log scale.")
    ui.plot(viz.single_metric_bar(table, "Training time (s)", height=320, fmt="{:.1f}s", log=True), key="cmp_train")
with b:
    ui.section("Inference time", "Milliseconds per tweet, log scale.")
    ui.plot(viz.single_metric_bar(table, "Inference (ms/tweet)", height=320, fmt="{:.2f}", log=True), key="cmp_inf")

ui.section("F1-score per category", "Where each model is strong or weak.")
ui.plot(viz.per_class_f1_comparison(metrics, height=440), key="cmp_f1")

# --- auto-generated findings from the actual numbers ----------------------------
classes = metrics["classes"]
f1_by_model = {k: {c: m["classification_report"][c]["f1-score"] for c in classes} for k, m in metrics["models"].items()}
f1_df = pd.DataFrame(f1_by_model)
weakest = f1_df.mean(axis=1).sort_values().head(3)
strongest = f1_df.mean(axis=1).sort_values(ascending=False).head(3)
ranked = table.sort_values("Macro F1", ascending=False)
lines = [f"<li><b>{ranked.iloc[0]['Model']}</b> has the highest macro-F1 ({ranked.iloc[0]['Macro F1']:.3f})"
         + (f", {ranked.iloc[0]['Macro F1'] - ranked.iloc[1]['Macro F1']:+.3f} vs {ranked.iloc[1]['Model']}." if len(ranked) > 1 else ".") + "</li>"]
if "distilbert" in f1_by_model and "svm" in f1_by_model:
    diff = (f1_df["distilbert"] - f1_df["svm"]).sort_values()
    lines.append(f"<li>DistilBERT vs SVM per category: largest gain on <b>{diff.index[-1]}</b> ({diff.iloc[-1]:+.3f} F1), "
                 f"largest loss on <b>{diff.index[0]}</b> ({diff.iloc[0]:+.3f} F1).</li>")
    sp = table.set_index("key")["Inference (ms/tweet)"]
    lines.append(f"<li>DistilBERT is ≈{sp['distilbert'] / sp['svm']:.0f}× slower per tweet than the SVM on CPU "
                 f"({sp['distilbert']:.1f} ms vs {sp['svm']:.2f} ms).</li>")
lines.append(f"<li>Weakest categories (mean F1 over models): " + ", ".join(f"<b>{c}</b> ({v:.2f})" for c, v in weakest.items())
             + ". These are the smallest classes and overlap semantically with News and Lifestyle.</li>")
lines.append(f"<li>Strongest categories: " + ", ".join(f"<b>{c}</b> ({v:.2f})" for c, v in strongest.items()) + ".</li>")
st.html(f'<div class="card"><h4>Key findings (computed from the test results)</h4><ul>{"".join(lines)}</ul></div>')

# --- transformer training ------------------------------------------------------
if history.get("epochs"):
    ui.section("DistilBERT fine-tuning", "Loss curves, validation macro-F1 and learning-rate schedule.")
    p = history["params"]
    ui.metric_cards([
        {"label": "Base model", "value": history.get("base_model", "—"), "small": True, "accent": "#A78BFA"},
        {"label": "Epochs run", "value": f"{len(history['epochs'])} / {p['epochs']}", "sub": f"best epoch {history.get('best_epoch', '—')}", "accent": "#38BDF8"},
        {"label": "Batch size", "value": p["batch_size"], "sub": f"max length {p['max_length']} tokens", "accent": "#22C55E"},
        {"label": "Learning rate", "value": f"{p['learning_rate']:.0e}", "sub": f"warm-up {p['warmup_ratio']:.0%}, linear decay", "accent": "#EAB308"},
        {"label": "Training time", "value": ui.fmt_seconds(history.get("train_seconds")), "sub": f"device: {history.get('device', 'cpu')}", "accent": "#F97316"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        ui.plot(viz.training_curves(history, height=360), key="cmp_curves")
    with b:
        ui.plot(viz.step_loss_curve(history, height=360), key="cmp_steps")
    st.dataframe(pd.DataFrame(history["epochs"]), hide_index=True, width="stretch", column_config={
        "learning_rate_end": st.column_config.NumberColumn("LR at epoch end", format="%.2e"),
        "epoch_seconds": st.column_config.NumberColumn("Epoch time (s)", format="%.0f")})

if classical:
    ui.section("Classical model tuning", "C selected by macro-F1 on the validation split.")
    a, b = st.columns(2, gap="large")
    for col, key, name in ((a, "svm", "Linear SVM"), (b, "logreg", "Logistic Regression")):
        grid = pd.DataFrame(classical[key]["grid"])
        with col:
            st.markdown(f"**{name}** · best C = `{classical[key]['best_C']}`")
            st.dataframe(grid, hide_index=True, width="stretch",
                         column_config={"val_macro_f1": st.column_config.NumberColumn("Val macro-F1", format="%.4f"),
                                        "val_accuracy": st.column_config.NumberColumn("Val accuracy", format="%.4f")})
    st.caption(f"TF-IDF vocabulary: {classical['tfidf']['n_features']:,} uni/bi-gram features "
               f"(min_df=2, max_df=0.9, sublinear tf). SVM probabilities: {classical['svm']['calibration']}.")

ui.section("TF-IDF vs transformer representations")
st.html("""
<div class="card"><table style="width:100%;border-collapse:collapse;font-size:.88rem">
<tr style="text-align:left;color:var(--muted)"><th style="padding:.4rem">Aspect</th><th>TF-IDF + linear model</th><th>DistilBERT</th></tr>
<tr><td style="padding:.4rem"><b>Representation</b></td><td>Sparse vector of word / bigram weights</td><td>Dense 768-d contextual embeddings</td></tr>
<tr><td style="padding:.4rem"><b>Word order & context</b></td><td>Only within bigrams</td><td>Full self-attention over the tweet</td></tr>
<tr><td style="padding:.4rem"><b>Synonyms / unseen words</b></td><td>Unknown words are ignored</td><td>Sub-word tokens + pre-training knowledge</td></tr>
<tr><td style="padding:.4rem"><b>Preprocessing</b></td><td>Aggressive: stop words, lemmatisation</td><td>Light: natural text kept</td></tr>
<tr><td style="padding:.4rem"><b>Explainability</b></td><td>Exact per-word contributions</td><td>Needs post-hoc methods (occlusion)</td></tr>
<tr><td style="padding:.4rem"><b>Cost</b></td><td>Seconds to train, sub-millisecond inference</td><td>Tens of minutes to fine-tune on CPU, ~10× slower inference</td></tr>
</table></div>""")
