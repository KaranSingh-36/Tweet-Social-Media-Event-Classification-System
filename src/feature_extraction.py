"""Text representation: TF-IDF features and linear-model explanations.

TF-IDF (term frequency - inverse document frequency) represents every tweet as
a sparse vector over a vocabulary of unigrams and bigrams. Each dimension is a
word / phrase; its weight grows with how often it appears in the tweet and
shrinks with how common it is across the corpus. It is fast, interpretable and
works well for topical classification, but it ignores word order beyond the
bigram window and has no notion of synonyms.

The transformer representation (see ``transformer_model.py``) instead produces
dense, contextual embeddings where the meaning of a word depends on the whole
sentence, e.g. "fire" in "the new album is fire" vs "fire destroys homes".
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import settings


def build_tfidf_vectorizer(**overrides) -> TfidfVectorizer:
    params = {**settings.TFIDF_PARAMS, **overrides}
    # Text is already tokenised/normalised by preprocessing.preprocess_for_classical
    return TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", lowercase=False, **params)


def linear_contributions(
    vectorizer: TfidfVectorizer, coef: np.ndarray, processed_text: str, class_index: int, top_k: int = 8
) -> list[dict]:
    """Per-term contribution (tf-idf weight x class coefficient) for one tweet.

    For linear models this is an exact decomposition of the class score, so it
    is a faithful explanation of *why* the model chose a category.
    """
    vec = vectorizer.transform([processed_text])
    if vec.nnz == 0:
        return []
    feature_names = vectorizer.get_feature_names_out()
    idx = vec.indices
    weights = vec.data * coef[class_index, idx]
    order = np.argsort(-weights)
    out = []
    for i in order[:top_k]:
        if weights[i] <= 0:
            break
        out.append({"term": str(feature_names[idx[i]]), "weight": float(weights[i])})
    return out
