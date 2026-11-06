"""Central configuration for the Tweet Event Classification System.

Every path, hyper-parameter and label mapping used by the project lives here so
that training scripts, the Streamlit app and the tests stay consistent.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (everything stays inside the project directory)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TWEETTOPIC_RAW_DIR = RAW_DIR / "tweet_topic_multi"
HUMAID_RAW_DIR = RAW_DIR / "humaid_events"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = ROOT_DIR / "models"
CLASSICAL_MODEL_DIR = MODELS_DIR / "classical"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
TRANSFORMER_FINAL_DIR = TRANSFORMER_MODEL_DIR / "distilbert-event-classifier"
TRANSFORMER_CHECKPOINT_DIR = TRANSFORMER_MODEL_DIR / "checkpoints"

REPORTS_DIR = ROOT_DIR / "reports" / "evaluation_results"
LOGS_DIR = ROOT_DIR / "logs"
ASSETS_DIR = ROOT_DIR / "assets"
NLTK_DATA_DIR = ROOT_DIR / "nltk_data"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"

# Keep Hugging Face downloads inside the project directory.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

DATASET_FILE = PROCESSED_DIR / "event_tweets.parquet"
DATASET_CSV_FILE = PROCESSED_DIR / "event_tweets.csv"
EVENT_TYPE_DATASET_FILE = PROCESSED_DIR / "disaster_event_types.parquet"
DATA_REPORT_FILE = PROCESSED_DIR / "data_preparation_report.json"

TFIDF_FILE = CLASSICAL_MODEL_DIR / "tfidf_vectorizer.joblib"
SVM_FILE = CLASSICAL_MODEL_DIR / "svm_calibrated.joblib"
LOGREG_FILE = CLASSICAL_MODEL_DIR / "logistic_regression.joblib"
LABEL_ENCODER_FILE = CLASSICAL_MODEL_DIR / "label_encoder.joblib"
EVENT_TYPE_MODEL_FILE = CLASSICAL_MODEL_DIR / "disaster_event_type_pipeline.joblib"

METRICS_FILE = REPORTS_DIR / "metrics.json"
TEST_PREDICTIONS_FILE = EVALUATION_DIR / "test_predictions.parquet"
TRANSFORMER_HISTORY_FILE = REPORTS_DIR / "transformer_training_history.json"
PREDICTION_LOG_FILE = LOGS_DIR / "prediction_history.csv"

# --------------------------------------------------------------------------- #
# Dataset sources (public, human-annotated tweet corpora)
# --------------------------------------------------------------------------- #
HF_PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/0.parquet"

TWEETTOPIC_REPO = "cardiffnlp/tweet_topic_multi"
TWEETTOPIC_CONFIG = "tweet_topic_multi"
# train_coling2022 + test_coling2022 together contain every tweet of the corpus.
TWEETTOPIC_SPLITS = ["train_coling2022", "test_coling2022"]

HUMAID_REPO = "QCRI/HumAID-events"
HUMAID_SPLITS = ["train", "dev", "test"]
# Event name -> disaster event type. Derived only from the official event name.
HUMAID_EVENTS = {
    "california_wildfires_2018": "Wildfire",
    "canada_wildfires_2016": "Wildfire",
    "greece_wildfires_2018": "Wildfire",
    "cyclone_idai_2019": "Hurricane / Cyclone",
    "hurricane_dorian_2019": "Hurricane / Cyclone",
    "hurricane_florence_2018": "Hurricane / Cyclone",
    "hurricane_harvey_2017": "Hurricane / Cyclone",
    "hurricane_irma_2017": "Hurricane / Cyclone",
    "hurricane_maria_2017": "Hurricane / Cyclone",
    "hurricane_matthew_2016": "Hurricane / Cyclone",
    "ecuador_earthquake_2016": "Earthquake",
    "italy_earthquake_aug_2016": "Earthquake",
    "kaikoura_earthquake_2016": "Earthquake",
    "pakistan_earthquake_2019": "Earthquake",
    "puebla_mexico_earthquake_2017": "Earthquake",
    "kerala_floods_2018": "Flood",
    "maryland_floods_2018": "Flood",
    "midwestern_us_floods_2019": "Flood",
    "srilanka_floods_2017": "Flood",
}
# HumAID tweets annotated "not_humanitarian" are often not about the disaster,
# so they are excluded from the "Natural Disaster" category.
HUMAID_EXCLUDED_LABELS = {"not_humanitarian"}
HUMAID_SAMPLE_SIZE = 2400  # disaster tweets added to the composite dataset
EVENT_TYPE_MAX_PER_TYPE = 6000  # cap per event type for the event-type model

# --------------------------------------------------------------------------- #
# Label taxonomy
# --------------------------------------------------------------------------- #
NATURAL_DISASTER = "Natural Disaster"

# TweetTopic's 19 fine-grained topics -> coarse event categories.
TWEETTOPIC_TO_CATEGORY = {
    "news_&_social_concern": "News & Social Concern",
    "sports": "Sports & Gaming",
    "gaming": "Sports & Gaming",
    "music": "Entertainment",
    "film_tv_&_video": "Entertainment",
    "celebrity_&_pop_culture": "Entertainment",
    "arts_&_culture": "Entertainment",
    "business_&_entrepreneurs": "Business",
    "science_&_technology": "Science & Technology",
    "fitness_&_health": "Health & Fitness",
    "learning_&_educational": "Education",
    "youth_&_student_life": "Education",
    "diaries_&_daily_life": "Lifestyle & Daily Life",
    "family": "Lifestyle & Daily Life",
    "relationships": "Lifestyle & Daily Life",
    "food_&_dining": "Lifestyle & Daily Life",
    "fashion_&_style": "Lifestyle & Daily Life",
    "travel_&_adventure": "Lifestyle & Daily Life",
    "other_hobbies": "Lifestyle & Daily Life",
}
# "diaries_&_daily_life" & co. are catch-all topics: a tweet labelled
# daily_life + news is primarily *news*. Lifestyle is therefore only used as the
# label when a tweet has no other, more specific category.
FALLBACK_CATEGORY = "Lifestyle & Daily Life"
# Tweets whose human labels span more than this many coarse categories are
# considered too ambiguous for single-label classification and are dropped.
MAX_COARSE_CATEGORIES = 2

CATEGORY_COLORS = {
    "Natural Disaster": "#F97316",
    "News & Social Concern": "#EF4444",
    "Sports & Gaming": "#22C55E",
    "Entertainment": "#A855F7",
    "Business": "#EAB308",
    "Science & Technology": "#06B6D4",
    "Health & Fitness": "#EC4899",
    "Education": "#3B82F6",
    "Lifestyle & Daily Life": "#14B8A6",
}
CATEGORY_DESCRIPTIONS = {
    "Natural Disaster": "Earthquakes, hurricanes, floods and wildfires (HumAID)",
    "News & Social Concern": "Politics, crime, public affairs and social issues",
    "Sports & Gaming": "Matches, athletes, leagues, e-sports and video games",
    "Entertainment": "Music, film, TV, celebrities, arts and culture",
    "Business": "Companies, markets, entrepreneurship and the economy",
    "Science & Technology": "Research, gadgets, software, space and innovation",
    "Health & Fitness": "Medicine, wellbeing, workouts and public health",
    "Education": "Learning, schools, students and educational content",
    "Lifestyle & Daily Life": "Personal diaries, family, food, fashion, travel, hobbies",
}

# --------------------------------------------------------------------------- #
# Splitting & model hyper-parameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # fraction of the full dataset

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "max_features": 50_000,
    "sublinear_tf": True,
}
SVM_C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0]
LOGREG_C_GRID = [1.0, 2.0, 5.0, 10.0, 20.0]

TRANSFORMER_BASE_MODEL = "distilbert-base-uncased"
TRANSFORMER_PARAMS = {
    "max_length": 96,
    "batch_size": 32,
    "eval_batch_size": 64,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "epochs": 3,
    "early_stopping_patience": 1,
    "seed": RANDOM_STATE,
}

BULK_BATCH_SIZE = 64
MAX_BULK_ROWS = 20_000
