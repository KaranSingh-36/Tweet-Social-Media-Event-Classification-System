"""Unit tests for the preprocessing pipeline."""
from src.preprocessing import (
    clean_for_transformer,
    dedup_key,
    emojis_to_text,
    normalize_mentions,
    preprocess_for_classical,
    preprocessing_steps,
    replace_urls,
    segment_hashtags,
)


def test_urls_removed():
    assert "http" not in replace_urls("see https://t.co/abc and www.example.com now")
    assert "URL" not in clean_for_transformer("news {{URL}} today")


def test_mentions_normalised_and_verified_names_kept():
    assert normalize_mentions("thanks @john_doe!") == "thanks @user!"
    assert normalize_mentions("{@NASA Earth@} says hi") == "NASA Earth says hi"
    assert normalize_mentions("{{USERNAME}} {{USERNAME}} hello").strip().startswith("@user hello")


def test_hashtag_words_are_kept_and_segmented():
    assert segment_hashtags("#HurricaneHarvey2017 update") == "Hurricane Harvey 2017 update"
    assert "earthquake" in preprocess_for_classical("#Earthquake hits the city")


def test_emoji_converted_to_words():
    assert "fire" in emojis_to_text("concert was 🔥")


def test_retweet_marker_removed():
    assert clean_for_transformer("RT @someone: Flood warning issued") == "Flood warning issued"


def test_transformer_text_keeps_natural_language():
    text = "The Rescue teams are searching for survivors."
    assert clean_for_transformer(text) == text  # no lowercasing / stop-word removal


def test_classical_pipeline_lowercases_removes_stopwords_and_lemmatises():
    out = preprocess_for_classical("The Rescue teams are searching for survivors!").split()
    assert "the" not in out and "are" not in out
    assert "rescue" in out and "team" in out and "survivor" in out and "search" in out


def test_hyphenated_words_split():
    assert preprocess_for_classical("Flood-ravaged homes").split()[:2] == ["flood", "ravage"]


def test_empty_and_missing_input():
    assert clean_for_transformer(None) == ""
    assert clean_for_transformer(float("nan")) == ""
    assert preprocess_for_classical("") == ""
    assert clean_for_transformer("@user @user") == ""


def test_dedup_key_ignores_urls_and_handles():
    assert dedup_key("RT @a: Big quake! https://t.co/1") == dedup_key("Big quake https://t.co/2")


def test_preprocessing_steps_trace():
    steps = preprocessing_steps("RT @x: #BigNews 🔥 http://a.b")
    assert steps[0][0] == "Original tweet" and len(steps) == 8
