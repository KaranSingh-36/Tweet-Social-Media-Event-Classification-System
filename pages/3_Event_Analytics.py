"""Event analytics: interactive exploration of the labelled dataset."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import ui
from src import visualization as viz
from src.data_exploration import category_length_stats, summary_statistics, top_hashtags, top_words, top_words_by_category

ui.page_header("Insights", "Event Analytics",
               "Explore the labelled tweet corpus: category distribution, class imbalance, vocabulary and tweet length.")

full = ui.require_dataset()
report = ui.get_data_report()

with st.container(border=True):
    f1, f2, f3 = st.columns([2, 1, 1])
    all_cats = sorted(full["label"].unique())
    cats = f1.multiselect("Categories", all_cats, placeholder="All categories")
    source = f2.selectbox("Source dataset", ["All", "TweetTopic", "HumAID"])
    split = f3.selectbox("Split", ["All", "train", "val", "test"])

df = full
if cats:
    df = df[df["label"].isin(cats)]
if source != "All":
    df = df[df["source"] == source]
if split != "All":
    df = df[df["split"] == split]
if df.empty:
    ui.empty_state("No tweets match these filters", "Widen the category, source or split selection.", "🔍")
    st.stop()

stats = summary_statistics(df, report)
counts = df["label"].value_counts()
ui.metric_cards([
    {"label": "Tweets", "value": f"{stats['total_tweets']:,}", "accent": "#38BDF8"},
    {"label": "Categories", "value": stats["n_categories"], "accent": "#A78BFA"},
    {"label": "Avg words / tweet", "value": f"{stats['avg_word_count']:.1f}", "sub": f"{stats['avg_char_length']:.0f} characters", "accent": "#22C55E"},
    {"label": "Imbalance ratio", "value": f"{stats['imbalance_ratio']:.1f}×", "sub": f"{stats['majority_class']} vs {stats['minority_class']}", "accent": "#F97316"},
    {"label": "With hashtags", "value": f"{stats['tweets_with_hashtags_pct']:.0f}%", "accent": "#EC4899"},
])

tab_dist, tab_kw, tab_len, tab_cmp, tab_quality = st.tabs(
    ["Distribution", "Keywords", "Tweet length", "Category comparison", "Data quality"])

with tab_dist:
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        ui.section("Tweets by category")
        ui.plot(viz.category_bar(counts, height=400, horizontal=False), key="an_bar")
    with b:
        ui.section("Category distribution")
        ui.plot(viz.category_donut(counts, height=400), key="an_donut")
    ui.section("Class imbalance", "Share of the dataset per category compared with a perfectly balanced dataset. "
               "All models use class weights to compensate for the small categories.")
    ui.plot(viz.imbalance_chart(counts, height=360), key="an_imb")

with tab_kw:
    a, b = st.columns(2, gap="large")
    with a:
        ui.section("Most frequent words", "After preprocessing (stop words removed, lemmatised).")
        ui.plot(viz.keyword_bar(top_words(df, 25), height=600), key="an_words")
    with b:
        ui.section("Keywords per category")
        cat = st.selectbox("Category", sorted(df["label"].unique()), key="kw_cat")
        ui.plot(viz.keyword_bar(top_words(df, 20, cat), color=viz.category_color(cat), height=520), key="an_words_cat")
    ui.section("Keyword map", "Top 12 words of every category; rectangle size = frequency.")
    ui.plot(viz.keyword_treemap(top_words_by_category(df, 12), height=560), key="an_tree")
    tags = top_hashtags(df, 20)
    if not tags.empty:
        ui.section("Most common hashtags", "Taken from the original, unprocessed tweets.")
        ui.plot(viz.keyword_bar(tags, word_col="hashtag", height=520), key="an_tags")

with tab_len:
    metric = st.segmented_control("Length measure", ["word_count", "char_length", "processed_token_count"],
                                  format_func=lambda m: {"word_count": "Words", "char_length": "Characters",
                                                         "processed_token_count": "Tokens after preprocessing"}[m],
                                  default="word_count", key="len_metric") or "word_count"
    a, b = st.columns(2, gap="large")
    with a:
        ui.section("Tweet length distribution")
        ui.plot(viz.length_histogram(df, metric, height=400), key="an_len_hist")
    with b:
        ui.section("Length by category")
        ui.plot(viz.length_histogram(df, metric, by_category=True, height=400), key="an_len_box")
    s1, s2 = st.columns(2, gap="large")
    for col, key, title in ((s1, "shortest_tweet", "Shortest tweet"), (s2, "longest_tweet", "Longest tweet")):
        t = stats[key]
        col.html(f'<div class="card"><h4>{title} · {t["chars"]} characters</h4>'
                 f'{ui.tweet_card(t["text"], ui.category_badge(t["label"]))}</div>')

with tab_cmp:
    ui.section("Event category comparison", "Average tweet characteristics per category.")
    cstats = category_length_stats(df)
    ui.plot(viz.category_stats_radar(cstats, height=400), key="an_cmp")
    st.dataframe(cstats, hide_index=True, width="stretch", column_config={
        "category": "Category", "tweets": st.column_config.NumberColumn("Tweets", format="%d"),
        "avg_words": "Avg words", "avg_chars": "Avg characters", "avg_hashtags": "Avg hashtags",
        "avg_processed_tokens": "Avg tokens (processed)"})
    ui.section("Composition by source and split")
    comp = df.groupby(["label", "source"]).size().reset_index(name="tweets")
    import plotly.express as px

    fig = px.bar(comp, x="label", y="tweets", color="source", barmode="stack",
                 color_discrete_map={"TweetTopic": "#38BDF8", "HumAID": "#F97316"})
    fig.update_xaxes(title=None)
    ui.plot(viz.style(fig, 360, legend=True), key="an_src")
    split_tbl = pd.crosstab(df["label"], df["split"]).reindex(columns=[c for c in ["train", "val", "test"] if c in df["split"].unique()])
    st.dataframe(split_tbl, width="stretch")

with tab_quality:
    comp = report.get("composite", {})
    cleaning = comp.get("cleaning", {})
    mapping = comp.get("tweettopic_label_mapping", {})
    ui.section("Data cleaning summary", "Applied to the full corpus before splitting (numbers are for the full dataset).")
    ui.metric_cards([
        {"label": "Missing text removed", "value": cleaning.get("missing_text", 0), "accent": "#38BDF8"},
        {"label": "Exact duplicates", "value": cleaning.get("exact_duplicates", 0), "accent": "#A78BFA"},
        {"label": "Near-duplicates / retweets", "value": cleaning.get("near_duplicates_retweets", 0), "accent": "#22C55E"},
        {"label": "Conflicting labels", "value": cleaning.get("conflicting_label_duplicates_removed", 0), "accent": "#F97316"},
        {"label": "Too ambiguous (3+ topics)", "value": mapping.get("too_ambiguous", 0), "accent": "#EF4444"},
    ])
    st.html(f"""<div class="card"><h4>How the labels were built</h4><ul>
      <li><b>TweetTopic</b>: 19 annotated topics mapped to 8 coarse categories. Tweets spanning more than two coarse
      categories ({mapping.get('too_ambiguous', 0)}) were dropped as too ambiguous. For tweets spanning two, the generic
      <i>Lifestyle &amp; Daily Life</i> category is only a fallback (daily-life + news → News); otherwise the more
      specific, less frequent category wins (business + news → Business).</li>
      <li><b>HumAID</b>: {settings.HUMAID_SAMPLE_SIZE:,} humanitarian tweets (balanced over earthquake, hurricane, flood and
      wildfire events) form the <i>Natural Disaster</i> category; {comp.get('humaid_excluded_not_humanitarian', 0):,}
      “not_humanitarian” tweets were excluded because many are unrelated to the disaster.</li>
      <li>Duplicates are detected on normalised text (URLs, handles and punctuation removed) so retweets are caught.</li>
      <li>Stratified split: 70% train · 10% validation · 20% test (random_state={settings.RANDOM_STATE}).</li></ul></div>""")
    rem = int(df[["text_original", "label"]].isna().sum().sum())
    st.caption(f"Missing values remaining in the processed dataset: {rem}.")
