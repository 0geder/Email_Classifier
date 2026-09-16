"""Classical ML baseline: TF-IDF + multinomial Logistic Regression.

Serves as the naive benchmark the other approaches must beat. With only 44
labeled examples across 5 classes, a sparse bag-of-words model is exactly
the kind of thing that overfits fast and generalizes poorly to unseen
vocabulary -- expected to be the weakest of the three approaches, which is
itself the point of including it.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ingestion import Email


@dataclass
class Prediction:
    email_id: str
    predicted_category: str
    confidence_score: float


class BaselineClassifier:
    name = "tfidf_logreg"

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.9,
                    ),
                ),
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
        self._fitted = False

    def fit(self, emails: list[Email]) -> "BaselineClassifier":
        texts = [e.text for e in emails]
        labels = [e.true_category for e in emails]
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, emails: list[Email]) -> list[Prediction]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        texts = [e.text for e in emails]
        probs = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        results = []
        for email, row in zip(emails, probs):
            best_idx = row.argmax()
            results.append(
                Prediction(
                    email_id=email.email_id,
                    predicted_category=classes[best_idx],
                    confidence_score=float(row[best_idx]),
                )
            )
        return results
