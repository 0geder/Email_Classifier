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

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self.centroids: list[_Centroid] = []

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

    def fit(self, emails: list[Email]) -> "EmbeddingClassifier":
        model = self._get_model()
        texts = [e.text for e in emails]
        embeddings = self._normalize(np.asarray(model.encode(texts)))

        by_label: dict[str, list[np.ndarray]] = defaultdict(list)
        for email, vec in zip(emails, embeddings):
            by_label[email.true_category].append(vec)

        self.centroids = [
            _Centroid(label=label, vector=self._normalize(np.mean(vecs, axis=0, keepdims=True))[0])
            for label, vecs in by_label.items()
        ]
        return self

    def predict(self, emails: list[Email]) -> list[Prediction]:
        if not self.centroids:
            raise RuntimeError("Call fit() before predict().")
        model = self._get_model()
        texts = [e.text for e in emails]
        embeddings = self._normalize(np.asarray(model.encode(texts)))

        centroid_matrix = np.stack([c.vector for c in self.centroids])  # (n_classes, dim)
        labels = [c.label for c in self.centroids]

        similarities = embeddings @ centroid_matrix.T  # (n_emails, n_classes), cosine sim since both normalized

        results = []
        for email, sims in zip(emails, similarities):
            scaled = sims / TEMPERATURE
            scaled -= scaled.max()  # numerical stability
            weights = np.exp(scaled)
            probs = weights / weights.sum()
            best_idx = int(probs.argmax())
            results.append(
                Prediction(
                    email_id=email.email_id,
                    predicted_category=labels[best_idx],
                    confidence_score=float(probs[best_idx]),
                )
            )
        return results
