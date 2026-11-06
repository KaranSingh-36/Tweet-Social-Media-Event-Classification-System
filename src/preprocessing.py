"""Tweet text preprocessing.

Two representations are produced for every tweet:

* ``clean_for_transformer`` – a *light* normalisation that keeps natural
  language intact (casing, stop words, punctuation) so that the DistilBERT
  tokenizer sees realistic sentences. Only noise that carries no topical
  meaning is normalised: URLs, user handles, HTML entities, emojis and
  CamelCase hashtags.
* ``preprocess_for_classical`` – the full classical NLP pipeline used for
  TF-IDF: lowercasing, tokenisation, punctuation/number removal, stop-word
  removal and WordNet lemmatisation.

The original tweet text is never modified; callers keep it separately.
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

import emoji
import nltk
from nltk.tokenize import TweetTokenizer

from config import settings

# --------------------------------------------------------------------------- #
# Regular expressions
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"(\{\{URL\}\}|https?://\S+|www\.\S+|\bt\.co/\S+)", re.IGNORECASE)
# TweetTopic marks verified accounts as {@Display Name@}; keep the readable name.
VERIFIED_USER_RE = re.compile(r"\{@\s*(.*?)\s*@\}")
USERNAME_PLACEHOLDER_RE = re.compile(r"\{\{USERNAME\}\}")
MENTION_RE = re.compile(r"(?<![\w@])@\w{1,30}")
# "RT @handle:" retweet attribution (removed entirely, it is not content).
RETWEET_RE = re.compile(r"^\s*RT\b\s*:?\s*(?:@\w+\s*:?)?\s*", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_SPLIT_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])"
)
REPEATED_USER_RE = re.compile(r"(@user[\s,:]*){2,}")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

USER_TOKEN = "@user"

_tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


# --------------------------------------------------------------------------- #
# NLTK resources (downloaded into the project folder, never globally)
# --------------------------------------------------------------------------- #
if str(settings.NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(settings.NLTK_DATA_DIR))


def ensure_nltk_resources(download: bool = True) -> dict[str, bool]:
    """Make sure the stop-word list and WordNet are available locally."""
    status = {}
    for resource, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
            status[resource] = True
        except LookupError:
            ok = False
            if download:
                settings.NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    ok = nltk.download(resource, download_dir=str(settings.NLTK_DATA_DIR), quiet=True)
                    nltk.data.find(path)
                except Exception:  # noqa: BLE001 - offline fallback below
                    ok = False
            status[resource] = bool(ok)
    return status


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """English stop words (NLTK, with scikit-learn's list as offline fallback)."""
    try:
        from nltk.corpus import stopwords

        words = set(stopwords.words("english"))
    except LookupError:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        words = set(ENGLISH_STOP_WORDS)
    # Twitter noise that carries no topical information.
    words |= {"rt", "amp", "via", "user", "url", "im", "u", "ur", "dont", "cant", "ive", "id"}
    return frozenset(words)


@lru_cache(maxsize=1)
def _get_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # triggers LookupError when WordNet is missing
        return lemmatizer
    except LookupError:
        return None


@lru_cache(maxsize=50_000)
def lemmatize(token: str) -> str:
    lemmatizer = _get_lemmatizer()
    if lemmatizer is None:
        return token
    lemma = lemmatizer.lemmatize(token)  # noun
    if lemma == token:
        lemma = lemmatizer.lemmatize(token, pos="v")  # e.g. "searching" -> "search"
    return lemma


# --------------------------------------------------------------------------- #
# Individual normalisation steps
# --------------------------------------------------------------------------- #
def replace_urls(text: str) -> str:
    """URLs carry no topical meaning in this corpus; remove them."""
    return URL_RE.sub(" ", text)


def normalize_mentions(text: str) -> str:
    """Verified display names are kept, raw handles become ``@user``."""
    text = VERIFIED_USER_RE.sub(lambda m: m.group(1), text)
    text = USERNAME_PLACEHOLDER_RE.sub(USER_TOKEN, text)
    text = MENTION_RE.sub(USER_TOKEN, text)
    return REPEATED_USER_RE.sub(USER_TOKEN + " ", text)


def segment_hashtags(text: str) -> str:
    """``#HurricaneHarvey2017`` -> ``Hurricane Harvey 2017`` (words are kept!)."""
    return HASHTAG_RE.sub(lambda m: CAMEL_SPLIT_RE.sub(" ", m.group(1)).replace("_", " "), text)


def emojis_to_text(text: str) -> str:
    """Convert emojis to words so that the models can use them (🔥 -> fire)."""
    text = emoji.demojize(text, delimiters=(" ", " "))
    # demojize produces tokens such as ``red_heart`` or ``face_with_tears_of_joy``
    return re.sub(r"\b([a-z]+(?:_[a-z]+)+)\b", lambda m: m.group(1).replace("_", " "), text)


def strip_control_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def clean_for_transformer(text: object) -> str:
    """Light normalisation for DistilBERT (natural language is preserved)."""
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return ""
    text = html.unescape(str(text))
    text = strip_control_chars(text)
    text = RETWEET_RE.sub("", text)
    text = replace_urls(text)
    text = normalize_mentions(text)
    text = segment_hashtags(text)
    text = emojis_to_text(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    # A tweet that only consisted of mentions is effectively empty.
    return "" if text.replace(USER_TOKEN, "").strip(" .,:;!?-") == "" else text


def tokenize(text: str) -> list[str]:
    """Twitter-aware tokenisation (handles, elongations like 'sooooo')."""
    return _tweet_tokenizer.tokenize(text)


def preprocess_for_classical(text: object, lemmatize_tokens: bool = True) -> str:
    """Full pipeline for TF-IDF: returns space-joined, normalised tokens."""
    cleaned = clean_for_transformer(text).replace(USER_TOKEN, " ")
    stop = get_stopwords()
    out = []
    for tok in _split_tokens(tokenize(cleaned)):
        if len(tok) < 2 or tok.isdigit() or tok in stop:
            continue
        if lemmatize_tokens:
            tok = lemmatize(tok)
        if tok not in stop:
            out.append(tok)
    return " ".join(out)


def _split_tokens(tokens: list[str]) -> list[str]:
    """Strip punctuation; hyphenated words become separate tokens."""
    out = []
    for tok in tokens:
        out.extend(p for p in NON_ALNUM_RE.split(tok.lower()) if p)
    return out


def dedup_key(text: object) -> str:
    """Aggressive key used only for duplicate / retweet detection."""
    return NON_ALNUM_RE.sub("", clean_for_transformer(text).lower().replace(USER_TOKEN, ""))


def preprocessing_steps(text: str) -> list[tuple[str, str]]:
    """Step-by-step trace of the pipeline (used in the UI for explanation)."""
    steps = [("Original tweet", text)]
    t = html.unescape(text)
    t = RETWEET_RE.sub("", t)
    t = WHITESPACE_RE.sub(" ", replace_urls(t)).strip()
    steps.append(("URLs & retweet marker removed", t))
    t = normalize_mentions(t)
    steps.append(("Mentions normalised to @user", t))
    t = segment_hashtags(t)
    steps.append(("Hashtags segmented (words kept)", t))
    t = WHITESPACE_RE.sub(" ", emojis_to_text(t)).strip()
    steps.append(("Emojis converted to text  ->  DistilBERT input", t))
    tokens = tokenize(t.replace(USER_TOKEN, " "))
    steps.append(("Tokenised & lowercased", " | ".join(tokens)))
    stop = get_stopwords()
    kept = [x for x in _split_tokens(tokens) if len(x) >= 2 and not x.isdigit() and x not in stop]
    steps.append(("Punctuation, numbers & stop words removed", " | ".join(kept)))
    steps.append(("Lemmatised  ->  TF-IDF input", preprocess_for_classical(text)))
    return steps
