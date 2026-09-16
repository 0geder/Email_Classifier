"""End-to-end pipeline: load data -> cross-validate both classifiers on the
labeled train set -> fit final models on all labeled data -> predict the
held-out test set -> write results/predictions_<model>.csv.

Run with:  python src/evaluate.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classifier_baseline import BaselineClassifier
from classifier_embedding import EmbeddingClassifier
from ingestion import Email, load_test_set, load_train_set

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

N_FOLDS = 5
CONFIDENCE_BINS = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]


def cross_validate(model_cls, emails: list[Email], n_folds: int = N_FOLDS) -> dict:
    labels = [e.true_category for e in emails]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    y_true_all, y_pred_all, conf_all = [], [], []
    start = time.perf_counter()

    for train_idx, test_idx in skf.split(emails, labels):
        train_fold = [emails[i] for i in train_idx]
        test_fold = [emails[i] for i in test_idx]

        model = model_cls()
        model.fit(train_fold)
        preds = model.predict(test_fold)

        y_true_all.extend(e.true_category for e in test_fold)
        y_pred_all.extend(p.predicted_category for p in preds)
        conf_all.extend(p.confidence_score for p in preds)

    elapsed = time.perf_counter() - start

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    calibration = _calibration_table(y_true_all, y_pred_all, conf_all)

    return {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "macro_f1": f1_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "per_class": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "calibration": calibration,
        "cv_wall_time_seconds": elapsed,
        "n_folds": n_folds,
    }


def _calibration_table(y_true: list[str], y_pred: list[str], conf: list[float]) -> list[dict]:
    conf = np.asarray(conf)
    correct = np.asarray([t == p for t, p in zip(y_true, y_pred)])
    table = []
    for lo, hi in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None
        table.append({"confidence_range": f"[{lo:.2f}, {hi:.2f})", "n": n, "empirical_accuracy": acc})
    return table


def predict_test_set(model_cls, train_emails: list[Email], test_emails: list[Email]) -> pd.DataFrame:
    model = model_cls()
    model.fit(train_emails)
    preds = model.predict(test_emails)
    return pd.DataFrame(
        [
            {
                "email_id": p.email_id,
                "predicted_category": p.predicted_category,
                "confidence_score": round(p.confidence_score, 4),
            }
            for p in preds
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    train_emails = load_train_set(DATA_DIR / "train", DATA_DIR / "train_labels.csv")
    test_emails = load_test_set(DATA_DIR / "test")

    models = {
        "baseline_tfidf_logreg": BaselineClassifier,
        "embedding_centroid": EmbeddingClassifier,
    }

    all_metrics = {}
    for model_name, model_cls in models.items():
        print(f"\n=== {model_name}: {N_FOLDS}-fold cross-validation on train set ===")
        metrics = cross_validate(model_cls, train_emails)
        all_metrics[model_name] = metrics
        print(f"accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
              f"cv_time={metrics['cv_wall_time_seconds']:.2f}s")

        print(f"--- {model_name}: fitting on full train set, predicting {len(test_emails)} test emails ---")
        predictions = predict_test_set(model_cls, train_emails, test_emails)
        out_path = RESULTS_DIR / f"predictions_{model_name}.csv"
        predictions.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "cv_wall_time_seconds": m["cv_wall_time_seconds"],
            }
            for name, m in all_metrics.items()
        ]
    )
    print("\n=== Comparison (5-fold CV on the 44 labeled train emails) ===")
    print(comparison.to_string(index=False))

    metrics_path = RESULTS_DIR / "evaluation_results.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(f"\nwrote {metrics_path}")
    print(f"wrote {RESULTS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
