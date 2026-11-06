"""Plotly chart builders with a consistent dashboard theme.

Every function receives real data (dataset, metrics or predictions) and
returns a ``plotly.graph_objects.Figure``; nothing is synthesised here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings

MODEL_COLORS = {"svm": "#3B82F6", "logreg": "#22D3EE", "distilbert": "#A78BFA"}
ACCENT = "#38BDF8"

THEMES = {
    "dark": {"muted": "#64748B", "grid": "rgba(148,163,184,0.12)"},
    "light": {"muted": "#64748B", "grid": "rgba(15,23,42,0.08)"},
}
_current_theme = "dark"
# Fixed scale that reads well on both dark and light backgrounds; Plotly picks
# black/white cell text automatically for contrast.
HEATMAP_SCALE = [[0, "#16233F"], [0.08, "#1E3A8A"], [0.4, "#1D4ED8"], [0.7, "#0369A1"], [1, "#0891B2"]]


def set_theme(name: str) -> None:
    global _current_theme
    _current_theme = name if name in THEMES else "dark"


def _t() -> dict:
    return THEMES[_current_theme]


def category_color(cat: str) -> str:
    return settings.CATEGORY_COLORS.get(cat, ACCENT)


def style(fig: go.Figure, height: int = 380, legend: bool = False, margin: dict | None = None) -> go.Figure:
    """Shared layout. Text colours are left to Streamlit's chart theme so the
    figures stay readable in both the dark and the light app theme."""
    t = _t()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        margin=margin or dict(l=10, r=10, t=30, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", title=None),
        hoverlabel=dict(font_size=12, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"])
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"])
    return fig


# --------------------------------------------------------------------------- #
# Dataset charts
# --------------------------------------------------------------------------- #
def category_bar(counts: pd.Series, height: int = 380, horizontal: bool = True) -> go.Figure:
    counts = counts.sort_values(ascending=horizontal)
    colors = [category_color(c) for c in counts.index]
    if horizontal:
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=colors,
                               text=[f"{v:,}" for v in counts.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,} tweets<extra></extra>"))
        fig.update_xaxes(title="Tweets", range=[0, counts.max() * 1.14])
    else:
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=colors,
                               text=[f"{v:,}" for v in counts.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:,} tweets<extra></extra>"))
        fig.update_yaxes(title="Tweets", range=[0, counts.max() * 1.12])
    fig.update_traces(marker_line_width=0)
    return style(fig, height, margin=dict(l=10, r=10, t=10, b=10))


def category_donut(counts: pd.Series, height: int = 380, center_text: str | None = None) -> go.Figure:
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=True,
                           marker=dict(colors=[category_color(c) for c in counts.index], line=dict(width=0)),
                           textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                           hovertemplate="%{label}<br>%{value:,} tweets (%{percent})<extra></extra>"))
    if center_text:
        fig.add_annotation(text=center_text, showarrow=False, font=dict(size=16))
    fig = style(fig, height, legend=True, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5, xanchor="left", yanchor="middle"),
                      uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def imbalance_chart(counts: pd.Series, height: int = 360) -> go.Figure:
    counts = counts.sort_values(ascending=False)
    share = counts / counts.sum() * 100
    uniform = 100 / len(counts)
    fig = go.Figure(go.Bar(x=counts.index, y=share.values, marker_color=[category_color(c) for c in counts.index],
                           text=[f"{v:.1f}%" for v in share.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:.2f}% of dataset<extra></extra>"))
    fig.add_hline(y=uniform, line_dash="dash", line_color=_t()["muted"],
                  annotation_text=f"Perfect balance ({uniform:.1f}%)", annotation_position="top right")
    fig.update_yaxes(title="Share of tweets (%)")
    return style(fig, height)


def length_histogram(df: pd.DataFrame, column: str = "word_count", by_category: bool = False, height: int = 380) -> go.Figure:
    if by_category:
        fig = px.box(df, x="label", y=column, color="label", color_discrete_map=settings.CATEGORY_COLORS, points=False)
        fig.update_xaxes(title=None, categoryorder="median descending")
        fig.update_yaxes(title=column.replace("_", " ").title())
    else:
        fig = px.histogram(df, x=column, nbins=60, color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_line_width=0, opacity=0.9)
        med = df[column].median()
        fig.add_vline(x=med, line_dash="dash", line_color="#F472B6", annotation_text=f"median {med:.0f}")
        fig.update_xaxes(title=column.replace("_", " ").title())
        fig.update_yaxes(title="Tweets")
    return style(fig, height)


def keyword_bar(words: pd.DataFrame, word_col: str = "word", value_col: str = "count", color: str = ACCENT,
                height: int = 420, value_title: str = "Frequency") -> go.Figure:
    words = words.sort_values(value_col)
    fig = go.Figure(go.Bar(x=words[value_col], y=words[word_col], orientation="h",
                           marker=dict(color=words[value_col], colorscale=[[0, "rgba(56,189,248,0.35)"], [1, color]]),
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_xaxes(title=value_title)
    return style(fig, height)


def keyword_treemap(df: pd.DataFrame, height: int = 520) -> go.Figure:
    fig = px.treemap(df, path=["category", "word"], values="count", color="category",
                     color_discrete_map=settings.CATEGORY_COLORS)
    fig.update_traces(root_color="rgba(0,0,0,0)", marker=dict(line=dict(width=1, color="rgba(15,23,42,0.6)")),
                      hovertemplate="%{label}<br>%{value:,} occurrences<extra></extra>")
    return style(fig, height, margin=dict(l=0, r=0, t=10, b=0))


def category_stats_radar(stats: pd.DataFrame, height: int = 420) -> go.Figure:
    metrics = ["avg_words", "avg_hashtags", "avg_processed_tokens"]
    fig = go.Figure()
    for metric, color in zip(metrics, ["#38BDF8", "#A78BFA", "#22C55E"]):
        fig.add_trace(go.Bar(name=metric.replace("avg_", "Avg ").replace("_", " "), x=stats["category"], y=stats[metric], marker_color=color))
    fig.update_layout(barmode="group")
    return style(fig, height, legend=True)


# --------------------------------------------------------------------------- #
# Model performance charts
# --------------------------------------------------------------------------- #
def metrics_grouped_bar(table: pd.DataFrame, metrics: list[str], height: int = 400) -> go.Figure:
    fig = go.Figure()
    for _, row in table.iterrows():
        fig.add_trace(go.Bar(name=row["Model"], x=metrics, y=[row[m] for m in metrics],
                             marker_color=MODEL_COLORS.get(row["key"], ACCENT),
                             text=[f"{row[m]:.3f}" for m in metrics], textposition="outside", cliponaxis=False,
                             hovertemplate=row["Model"] + "<br>%{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 1.08], title="Score")
    return style(fig, height, legend=True)


def single_metric_bar(table: pd.DataFrame, metric: str, height: int = 300, fmt: str = "{:.2f}", log: bool = False) -> go.Figure:
    t = table.dropna(subset=[metric])
    fig = go.Figure(go.Bar(x=t["Model"], y=t[metric], marker_color=[MODEL_COLORS.get(k, ACCENT) for k in t["key"]],
                           text=[fmt.format(v) for v in t[metric]], textposition="outside", cliponaxis=False))
    if log and len(t):
        lo, hi = float(t[metric].min()), float(t[metric].max())
        fig.update_yaxes(type="log", range=[np.log10(lo / 4), np.log10(hi * 3)])
    fig.update_yaxes(title=metric)
    return style(fig, height, margin=dict(l=10, r=10, t=10, b=10))


def per_class_f1_comparison(metrics: dict, height: int = 420) -> go.Figure:
    classes = metrics["classes"]
    fig = go.Figure()
    for key, m in metrics["models"].items():
        rep = m["classification_report"]
        fig.add_trace(go.Bar(name=m["display_name"], x=classes, y=[rep[c]["f1-score"] for c in classes],
                             marker_color=MODEL_COLORS.get(key, ACCENT),
                             hovertemplate=m["display_name"] + "<br>%{x}: F1 %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], title="F1-score")
    return style(fig, height, legend=True)


def confusion_matrix_heatmap(cm: list[list[int]], classes: list[str], normalize: bool = True, height: int = 560) -> go.Figure:
    cm = np.array(cm, dtype=float)
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    text = [[(f"{z[i, j]:.0%}" if normalize else f"{int(cm[i, j])}") if cm[i, j] > 0 else "" for j in range(len(classes))]
            for i in range(len(classes))]
    custom = np.dstack([cm, z])
    fig = go.Figure(go.Heatmap(
        z=z, x=classes, y=classes, colorscale=HEATMAP_SCALE, text=text, texttemplate="%{text}",
        textfont=dict(size=11, color="#F8FAFC"), customdata=custom, showscale=True, xgap=2, ygap=2,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Tweets: %{customdata[0]:.0f}<br>Row share: %{customdata[1]:.1%}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%" if normalize else "d"),
    ))
    fig.update_xaxes(title="Predicted category", tickangle=-40, side="bottom", constrain="domain")
    fig.update_yaxes(title="Actual category", autorange="reversed")
    return style(fig, height, margin=dict(l=10, r=10, t=10, b=10))


def precision_recall_scatter(table: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for _, r in table.iterrows():
        fig.add_trace(go.Scatter(x=[r["Recall"]], y=[r["Precision"]], mode="markers", name=r["Category"],
                                 marker=dict(size=10 + 40 * r["Support"] / table["Support"].max(), color=category_color(r["Category"]),
                                             line=dict(width=0), opacity=0.85),
                                 hovertemplate=f"{r['Category']}<br>Precision %{{y:.3f}}<br>Recall %{{x:.3f}}<br>Support {r['Support']}<extra></extra>"))
    fig.update_xaxes(title="Recall", range=[0, 1.05])
    fig.update_yaxes(title="Precision", range=[0, 1.05])
    fig = style(fig, height, legend=True)
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle", itemsizing="constant"))
    return fig


def confidence_histogram(preds: pd.DataFrame, model_key: str, height: int = 360) -> go.Figure:
    df = preds.assign(Outcome=np.where(preds["label"] == preds[f"pred_{model_key}"], "Correct", "Incorrect"))
    fig = px.histogram(df, x=f"conf_{model_key}", color="Outcome", nbins=30, barmode="overlay",
                       color_discrete_map={"Correct": "#22C55E", "Incorrect": "#EF4444"}, opacity=0.75)
    fig.update_xaxes(title="Model confidence (max probability)", range=[0, 1])
    fig.update_yaxes(title="Test tweets")
    fig.update_traces(marker_line_width=0)
    return style(fig, height, legend=True)


def confidence_values_histogram(values: pd.Series, height: int = 300) -> go.Figure:
    fig = px.histogram(values, nbins=25, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Confidence", range=[0, 1])
    fig.update_yaxes(title="Tweets")
    return style(fig, height)


def error_rate_bar(err: pd.DataFrame, height: int = 380) -> go.Figure:
    err = err.sort_values("error_rate")
    fig = go.Figure(go.Bar(x=err["error_rate"], y=err["category"], orientation="h",
                           marker_color=[category_color(c) for c in err["category"]],
                           text=[f"{r:.0%} ({e}/{t})" for r, e, t in zip(err["error_rate"], err["errors"], err["tweets"])],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Error rate %{x:.1%}<extra></extra>"))
    fig.update_xaxes(title="Error rate", tickformat=".0%", range=[0, min(1, err["error_rate"].max() * 1.35 + 0.05)])
    return style(fig, height)


def simple_rate_bar(df: pd.DataFrame, x: str, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x].astype(str), y=df["error_rate"], marker_color=ACCENT,
                           text=[f"{v:.0%}<br>n={n}" for v, n in zip(df["error_rate"], df["tweets"])], textposition="outside",
                           cliponaxis=False))
    fig.update_yaxes(title="Error rate", tickformat=".0%", range=[0, min(1, df["error_rate"].max() * 1.4 + 0.05)])
    fig.update_xaxes(title=None)
    return style(fig, height)


def probability_bar(top: list[tuple[str, float]], height: int = 320) -> go.Figure:
    top = list(reversed(top))
    fig = go.Figure(go.Bar(x=[p for _, p in top], y=[c for c, _ in top], orientation="h",
                           marker_color=[category_color(c) for c, _ in top],
                           text=[f"{p:.1%}" for _, p in top], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title=None)
    return style(fig, height, margin=dict(l=10, r=30, t=10, b=10))


def keyword_importance_bar(keywords: list[dict], color: str, height: int = 280, title: str = "Contribution") -> go.Figure:
    kw = list(reversed(keywords))
    fig = go.Figure(go.Bar(x=[k["weight"] for k in kw], y=[k["term"] for k in kw], orientation="h", marker_color=color,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=title)
    return style(fig, height, margin=dict(l=10, r=10, t=10, b=10))


def training_curves(history: dict, height: int = 360) -> go.Figure:
    ep = pd.DataFrame(history.get("epochs", []))
    fig = go.Figure()
    if not ep.empty:
        fig.add_trace(go.Scatter(x=ep["epoch"], y=ep["train_loss"], name="Training loss", mode="lines+markers", line=dict(color="#38BDF8", width=3)))
        fig.add_trace(go.Scatter(x=ep["epoch"], y=ep["val_loss"], name="Validation loss", mode="lines+markers", line=dict(color="#F472B6", width=3)))
        fig.add_trace(go.Scatter(x=ep["epoch"], y=ep["val_macro_f1"], name="Validation macro-F1", mode="lines+markers",
                                 line=dict(color="#22C55E", width=3, dash="dot"), yaxis="y2"))
        if history.get("best_epoch"):
            fig.add_vline(x=history["best_epoch"], line_dash="dash", line_color=_t()["muted"], annotation_text="best checkpoint")
    fig = style(fig, height, legend=True)
    fig.update_layout(yaxis=dict(title="Loss"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1], tickmode="linear", dtick=0.2, tickformat=".1f",
                                 title="Validation macro-F1", showgrid=False))
    fig.update_xaxes(title="Epoch", dtick=1)
    return fig


def step_loss_curve(history: dict, height: int = 300) -> go.Figure:
    st_ = pd.DataFrame(history.get("steps", []))
    fig = go.Figure()
    if not st_.empty:
        st_["smoothed"] = st_["loss"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=st_["step"], y=st_["loss"], name="Batch loss", mode="lines", line=dict(color="rgba(56,189,248,0.3)")))
        fig.add_trace(go.Scatter(x=st_["step"], y=st_["smoothed"], name="Smoothed", mode="lines", line=dict(color="#38BDF8", width=2.5)))
        fig.add_trace(go.Scatter(x=st_["step"], y=st_["learning_rate"], name="Learning rate", mode="lines",
                                 line=dict(color="#FBBF24", width=2, dash="dot"), yaxis="y2"))
    fig = style(fig, height, legend=True)
    fig.update_layout(yaxis=dict(title="Loss"),
                      yaxis2=dict(overlaying="y", side="right", title="Learning rate", showgrid=False, tickformat=".0e",
                                 rangemode="tozero"))
    fig.update_xaxes(title="Training step")
    return fig


def disagreement_sankey(preds: pd.DataFrame, a: str, b: str, height: int = 380) -> go.Figure:
    d = preds.assign(a_ok=preds[f"pred_{a}"] == preds["label"], b_ok=preds[f"pred_{b}"] == preds["label"])
    groups = {
        "Both correct": int((d.a_ok & d.b_ok).sum()),
        f"Only {a.upper()} correct": int((d.a_ok & ~d.b_ok).sum()),
        f"Only {b.upper()} correct": int((~d.a_ok & d.b_ok).sum()),
        "Both wrong": int((~d.a_ok & ~d.b_ok).sum()),
    }
    colors = ["#22C55E", MODEL_COLORS.get(a, ACCENT), MODEL_COLORS.get(b, ACCENT), "#EF4444"]
    fig = go.Figure(go.Bar(x=list(groups.keys()), y=list(groups.values()), marker_color=colors,
                           text=[f"{v:,}" for v in groups.values()], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Test tweets")
    return style(fig, height)


def activity_timeline(log: pd.DataFrame, height: int = 260) -> go.Figure:
    df = log.dropna(subset=["timestamp"]).copy()
    df["date"] = df["timestamp"].dt.floor("h")
    agg = df.groupby(["date", "category"]).size().reset_index(name="predictions")
    fig = px.bar(agg, x="date", y="predictions", color="category", color_discrete_map=settings.CATEGORY_COLORS)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Predictions")
    return style(fig, height, legend=False)
