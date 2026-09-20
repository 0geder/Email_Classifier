"""Classical ML baseline: TF-IDF + multinomial Logistic Regression.

Serves as the naive benchmark the other approaches must beat. With only 44
labeled examples across 5 classes, a sparse bag-of-words model is exactly
the kind of thing that overfits fast and generalizes poorly to unseen
vocabulary -- expected to be the weakest of the three approaches, which is
itself the point of including it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ingestion import Email

# Word (1-2 gram) and character (2-5 gram) TF-IDF, combined. Character n-grams
# were added after comparing against an alternative implementation of this
# same challenge that used them; kept because the ablation in evaluate.py
# showed a genuine macro-F1 gain over word n-grams alone, not on assumption.
USE_CHAR_NGRAMS = True


@dataclass
class Prediction:
    email_id: str
    predicted_category: str
    confidence_score: float


def _build_pipeline(use_char_ngrams: bool) -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
    )
    if not use_char_ngrams:
        features = word_vectorizer
    else:
        features = FeatureUnion(
            [
                ("word", word_vectorizer),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=1,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("tfidf", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


class BaselineClassifier:
    name = "tfidf_logreg"

    def __init__(self, use_char_ngrams: bool = USE_CHAR_NGRAMS) -> None:
        self.pipeline = _build_pipeline(use_char_ngrams)
        self._fitted = False

    def fit(self, emails: list[Email]) -> "BaselineClassifier":
        texts = [e.text for e in emails]
        labels = [e.true_category for e in emails]
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict_proba(self, emails: list[Email]) -> tuple[list[str], np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        texts = [e.text for e in emails]
        probs = self.pipeline.predict_proba(texts)
        return list(self.pipeline.classes_), probs

    def predict(self, emails: list[Email]) -> list[Prediction]:
        classes, probs = self.predict_proba(emails)
        results = []
        for email, row in zip(emails, probs):
            best_idx = int(row.argmax())
            results.append(
                Prediction(
                    email_id=email.email_id,
                    predicted_category=classes[best_idx],
                    confidence_score=float(row[best_idx]),
                )
            )
        return results
