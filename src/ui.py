"""Reusable Streamlit UI components, cached loaders and page guards."""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from config import settings
from src import visualization as viz

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _css() -> str:
    return (settings.ASSETS_DIR / "custom.css").read_text(encoding="utf-8")


def inject_css() -> None:
    st.html(f"<style>{_css()}</style>")
    theme_type = getattr(st.context.theme, "type", None) or "dark"
    viz.set_theme(theme_type)


def esc(text: object) -> str:
    return html.escape(str(text))


# --------------------------------------------------------------------------- #
# Cached resources (loaded once per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading machine-learning models ...")
def get_service():
    from src.prediction import EventClassifierService

    service = EventClassifierService()
    service.warm_up()
    return service


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _dataset(mtime: float) -> pd.DataFrame:
    from src.data_exploration import add_length_features
    from src.data_loader import load_dataset

    return add_length_features(load_dataset())


def get_dataset() -> pd.DataFrame:
    return _dataset(_mtime(settings.DATASET_FILE))


@st.cache_data(show_spinner=False)
def _data_report(mtime: float) -> dict:
    from src.data_loader import load_data_report

    return load_data_report()


def get_data_report() -> dict:
    return _data_report(_mtime(settings.DATA_REPORT_FILE))


@st.cache_data(show_spinner=False)
def _metrics(mtime: float) -> dict:
    from src.evaluation import load_metrics

    return load_metrics()


def get_metrics() -> dict:
    return _metrics(_mtime(settings.METRICS_FILE))


@st.cache_data(show_spinner=False)
def _test_preds(mtime: float) -> pd.DataFrame:
    from src.evaluation import load_test_predictions

    return load_test_predictions()


def get_test_predictions() -> pd.DataFrame:
    return _test_preds(_mtime(settings.TEST_PREDICTIONS_FILE))


def get_transformer_history() -> dict:
    if settings.TRANSFORMER_HISTORY_FILE.exists():
        return json.loads(settings.TRANSFORMER_HISTORY_FILE.read_text())
    return {}


def get_classical_report() -> dict:
    path = settings.REPORTS_DIR / "classical_training.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.html(
        f"""<div class="page-header"><span class="eyebrow">{esc(eyebrow)}</span>
        <h1>{esc(title)}</h1>{f'<p>{esc(subtitle)}</p>' if subtitle else ''}</div>"""
    )


def section(title: str, caption: str = "") -> None:
    st.html(
        f"""<div class="section-title"><h3>{esc(title)}</h3></div>
        {f'<div class="section-caption">{esc(caption)}</div>' if caption else ''}"""
    )


def metric_cards(cards: list[dict], columns: int | None = None) -> None:
    """cards: [{label, value, sub?, accent?, small?}]"""
    cols = columns or len(cards)
    items = []
    for c in cards:
        accent = c.get("accent", "#38BDF8")
        small = " small" if c.get("small") else ""
        sub = f'<div class="sub">{esc(c["sub"])}</div>' if c.get("sub") else ""
        items.append(
            f'<div class="metric-card" style="--card-accent:{accent}"><div class="label">{esc(c["label"])}</div>'
            f'<div class="value{small}">{esc(c["value"])}</div>{sub}</div>'
        )
    st.html(f'<div class="metric-grid" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{"".join(items)}</div>')


def category_badge(category: str) -> str:
    color = viz.category_color(category)
    return f'<span class="cat-badge" style="--cat:{color}"><span class="dot"></span>{esc(category)}</span>'


def empty_state(title: str, message: str, icon: str = "◌", command: str | None = None) -> None:
    cmd = f'<p style="margin-top:.7rem"><code>{esc(command)}</code></p>' if command else ""
    st.html(f'<div class="empty-state"><div class="icon">{icon}</div><h4>{esc(title)}</h4><p>{esc(message)}</p>{cmd}</div>')


def tweet_card(text: str, badges_html: str = "", reasons: list[str] | None = None) -> str:
    reasons_html = "".join(f'<div class="reason">{esc(r)}</div>' for r in (reasons or []))
    return (f'<div class="tweet-box">{esc(text)}<div class="meta">{badges_html}</div>'
            f'{f"<div style=margin-top:.45rem>{reasons_html}</div>" if reasons_html else ""}</div>')


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def fmt_seconds(sec: float | None) -> str:
    if sec is None or pd.isna(sec):
        return "—"
    if sec < 60:
        return f"{sec:.1f} s"
    return f"{sec / 60:.1f} min"


# --------------------------------------------------------------------------- #
# Guards: show helpful empty states instead of stack traces
# --------------------------------------------------------------------------- #
def require_dataset() -> pd.DataFrame:
    if not settings.DATASET_FILE.exists():
        empty_state("Dataset not prepared yet", "Download and preprocess the public tweet datasets first.",
                    "🗂️", "python -m src.data_loader")
        st.stop()
    return get_dataset()


def require_models():
    service = get_service()
    if not service.available_models():
        empty_state("No trained models found", "Train the classical models (takes about a minute) to enable predictions.",
                    "🧠", "python -m src.classical_model")
        st.stop()
    return service


def require_metrics() -> dict:
    metrics = get_metrics()
    if not metrics.get("models"):
        empty_state("No evaluation results yet", "Evaluate the trained models on the held-out test set.",
                    "📊", "python -m src.evaluation")
        st.stop()
    return metrics


def model_selector(metrics: dict, key: str, label: str = "Model") -> str:
    from src.prediction import MODEL_DISPLAY_NAMES

    keys = list(metrics["models"].keys())
    default = keys.index("distilbert") if "distilbert" in keys else 0
    return st.selectbox(label, keys, index=default, format_func=lambda k: MODEL_DISPLAY_NAMES[k], key=key)
