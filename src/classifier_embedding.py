"""Primary NLP approach: pretrained sentence-embeddings + nearest-centroid.

With only ~9 labeled examples per class, fine-tuning a transformer is not
viable (nowhere near enough data), and the challenge spec offers no path to
a hosted LLM API that a reviewer could run without their own key. A frozen
sentence-encoder sidesteps both problems: it needs no training beyond
averaging a handful of embeddings per class (a Rocchio / nearest-centroid
classifier), it captures semantic similarity TF-IDF cannot (paraphrase,
synonymy), and it runs fully offline after one small (~90MB) model download.

Confidence is the softmax over cosine similarities to each class centroid,
so it reflects how much closer the email sits to its predicted class than
to the runner-up, not an arbitrary distance threshold.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from classifier_baseline import Prediction
from ingestion import Email

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE = 0.05  # controls how peaked the softmax over similarities is


@dataclass
class _Centroid:
    label: str
    vector: np.ndarray


class EmbeddingClassifier:
    name = "embedding_centroid"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self.centroids: list[_Centroid] = []
        # Keyed by Email.email_id. Evaluation re-fits this classifier on the
        # order of a hundred times (cross-validation plus the nested
        # calibration check); re-encoding the same 44-56 texts with the
        # transformer every time would make that impractically slow, so
        # evaluate.py precomputes embeddings once and passes them in here.
        # Without a cache, the classifier still works standalone by encoding
        # on the fly.
        self._embedding_cache = embedding_cache

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _embed(self, emails: list[Email]) -> np.ndarray:
        if self._embedding_cache is not None:
            missing = [e for e in emails if e.email_id not in self._embedding_cache]
            if not missing:
                return np.stack([self._embedding_cache[e.email_id] for e in emails])
        model = self._get_model()
        texts = [e.text for e in emails]
        return self._normalize(np.asarray(model.encode(texts)))

    def fit(self, emails: list[Email]) -> "EmbeddingClassifier":
        embeddings = self._embed(emails)

        by_label: dict[str, list[np.ndarray]] = defaultdict(list)
        for email, vec in zip(emails, embeddings):
            by_label[email.true_category].append(vec)

        self.centroids = [
            _Centroid(label=label, vector=self._normalize(np.mean(vecs, axis=0, keepdims=True))[0])
            for label, vecs in by_label.items()
        ]
        return self

    def predict_proba(self, emails: list[Email]) -> tuple[list[str], np.ndarray]:
        if not self.centroids:
            raise RuntimeError("Call fit() before predict_proba().")
        embeddings = self._embed(emails)

        centroid_matrix = np.stack([c.vector for c in self.centroids])  # (n_classes, dim)
        labels = [c.label for c in self.centroids]

        similarities = embeddings @ centroid_matrix.T  # (n_emails, n_classes), cosine sim since both normalized

        scaled = similarities / TEMPERATURE
        scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
        weights = np.exp(scaled)
        probs = weights / weights.sum(axis=1, keepdims=True)
        return labels, probs

    def predict(self, emails: list[Email]) -> list[Prediction]:
        labels, probs = self.predict_proba(emails)
        results = []
        for email, row in zip(emails, probs):
            best_idx = int(row.argmax())
            results.append(
                Prediction(
                    email_id=email.email_id,
                    predicted_category=labels[best_idx],
                    confidence_score=float(row[best_idx]),
                )
            )
        return results
