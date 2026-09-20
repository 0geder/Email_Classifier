"""End-to-end pipeline: load data -> cross-validate both classifiers on the
labeled train set -> fit final models on all labeled data -> predict the
held-out test set -> write results/predictions_<model>.csv.

Cross-validation is repeated stratified k-fold, pooled per sample, so the
headline metrics are not sensitive to a single lucky or unlucky fold split.
Confidence scores are calibrated with a single-parameter temperature fit on
the out-of-fold probabilities (src/calibration.py), and that same calibrated
confidence is what gets written into the submitted predictions.

Run with:  python src/evaluate.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from calibration import (
    apply_temperature,
    bootstrap_accuracy_ci,
    conformal_min_n,
    cost_weighted_error,
    expected_calibration_error,
    fit_temperature,
    multiclass_brier,
    risk_coverage_table,
)
from classifier_baseline import BaselineClassifier
from classifier_embedding import EmbeddingClassifier
from ingestion import Email, load_test_set, load_train_set

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

N_FOLDS = 5
N_REPEATS = 10
NESTED_OUTER_REPEATS = 5
NESTED_INNER_FOLDS = 4
CONFIDENCE_BINS = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
RISK_THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
# Illustrative only: a misrouted regulated request is assumed costlier than a
# misrouted low-stakes one. Set these with compliance before using them for
# anything beyond a demonstration of the metric.
COST_WEIGHTS = {
    "Insurance Claims": 5.0,
    "Loan Processing": 4.0,
    "Investment Advisory": 3.0,
    "Account Management": 2.0,
    "Other": 1.0,
}
PRIMARY_MODEL = "embedding_centroid"


def _pooled_out_of_fold(
    model_cls,
    emails: list[Email],
    model_kwargs: dict,
    n_splits: int,
    n_repeats: int,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Repeated stratified k-fold, averaged per sample. Every email is scored
    out-of-fold n_repeats times (under a different fold partition each time)
    and its predicted-probability vectors are averaged, which reduces the
    variance a single fold assignment would otherwise inject into the
    headline metrics at only 44 labeled examples.
    """
    labels = [e.true_category for e in emails]
    classes = sorted(set(labels))
    n = len(emails)
    prob_sum = np.zeros((n, len(classes)))
    count = np.zeros(n)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for train_idx, test_idx in cv.split(emails, labels):
        train_fold = [emails[i] for i in train_idx]
        test_fold = [emails[i] for i in test_idx]

        model = model_cls(**model_kwargs)
        model.fit(train_fold)
        fold_classes, fold_probs = model.predict_proba(test_fold)
        col_idx = [fold_classes.index(c) for c in classes]
        aligned = fold_probs[:, col_idx]

        for local_i, global_i in enumerate(test_idx):
            prob_sum[global_i] += aligned[local_i]
            count[global_i] += 1

    return classes, prob_sum / count[:, None]


def cross_validate(
    model_cls,
    emails: list[Email],
    model_kwargs: dict | None = None,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
) -> dict:
    model_kwargs = model_kwargs or {}
    y_true = [e.true_category for e in emails]

    start = time.perf_counter()
    classes, P = _pooled_out_of_fold(model_cls, emails, model_kwargs, n_folds, n_repeats)
    elapsed = time.perf_counter() - start

    pred_idx = P.argmax(axis=1)
    y_pred = [classes[i] for i in pred_idx]
    raw_conf = P.max(axis=1)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    calibration = _calibration_table(y_true, y_pred, raw_conf.tolist())

    ece_raw, calibration_table_raw = expected_calibration_error(raw_conf, correct)
    brier_raw = multiclass_brier(P, y_true, classes)

    temperature = fit_temperature(P, y_true, classes)
    P_calibrated = apply_temperature(P, temperature)
    calibrated_conf = P_calibrated.max(axis=1)
    ece_calibrated, calibration_table_calibrated = expected_calibration_error(calibrated_conf, correct)
    brier_calibrated = multiclass_brier(P_calibrated, y_true, classes)

    ci_low, ci_high = bootstrap_accuracy_ci(correct)
    risk_coverage = risk_coverage_table(calibrated_conf, correct, RISK_THRESHOLDS)
    cost_error = cost_weighted_error(y_true, y_pred, COST_WEIGHTS)

    misclassified = [
        {
            "email_id": emails[i].email_id,
            "true_category": y_true[i],
            "predicted_category": y_pred[i],
            "confidence": round(float(calibrated_conf[i]), 4),
            "text_snippet": " ".join(emails[i].text.split())[:80],
        }
        for i in range(len(emails))
        if not correct[i]
    ]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "calibration": calibration,
        "bootstrap_ci_95": {"low": ci_low, "high": ci_high},
        "ece_raw": ece_raw,
        "brier_raw": brier_raw,
        "temperature": temperature,
        "ece_calibrated": ece_calibrated,
        "brier_calibrated": brier_calibrated,
        "calibration_table_raw": calibration_table_raw,
        "calibration_table_calibrated": calibration_table_calibrated,
        "risk_coverage": risk_coverage,
        "cost_weighted_error": cost_error,
        "misclassified": misclassified,
        "classes": classes,
        "n_labeled": len(emails),
        "cv_wall_time_seconds": elapsed,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
    }


def nested_temperature_check(
    model_cls,
    emails: list[Email],
    model_kwargs: dict | None = None,
    outer_splits: int = N_FOLDS,
    outer_repeats: int = NESTED_OUTER_REPEATS,
    inner_splits: int = NESTED_INNER_FOLDS,
    seed: int = 42,
) -> dict:
    """Confirms the calibration gain is not an artifact of fitting the
    temperature on the same out-of-fold probabilities used to measure it.
    The temperature is fit only on an inner cross-validation of the outer
    training fold, then applied to the untouched outer test fold; this is
    repeated across many outer folds so the before/after ECE comparison
    reflects genuine held-out calibration, not in-sample fitting.
    """
    model_kwargs = model_kwargs or {}
    labels = [e.true_category for e in emails]
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed
    )

    ece_before_runs, ece_after_runs = [], []
    for outer_train_idx, outer_test_idx in outer_cv.split(emails, labels):
        outer_train = [emails[i] for i in outer_train_idx]
        outer_test = [emails[i] for i in outer_test_idx]
        outer_train_labels = [e.true_category for e in outer_train]

        inner_classes, inner_P = _pooled_out_of_fold(
            model_cls, outer_train, model_kwargs, inner_splits, 1, seed
        )
        temperature = fit_temperature(inner_P, outer_train_labels, inner_classes)

        model = model_cls(**model_kwargs)
        model.fit(outer_train)
        test_classes, test_P = model.predict_proba(outer_test)
        col_idx = [test_classes.index(c) for c in inner_classes]
        aligned_test_P = test_P[:, col_idx]

        y_test = [e.true_category for e in outer_test]
        pred_idx = aligned_test_P.argmax(axis=1)
        y_pred = [inner_classes[i] for i in pred_idx]
        correct = np.array([t == p for t, p in zip(y_test, y_pred)])

        ece_before, _ = expected_calibration_error(aligned_test_P.max(axis=1), correct)
        calibrated = apply_temperature(aligned_test_P, temperature)
        ece_after, _ = expected_calibration_error(calibrated.max(axis=1), correct)

        ece_before_runs.append(ece_before)
        ece_after_runs.append(ece_after)

    ece_before_runs = np.array(ece_before_runs)
    ece_after_runs = np.array(ece_after_runs)
    return {
        "ece_before_mean": float(ece_before_runs.mean()),
        "ece_before_std": float(ece_before_runs.std()),
        "ece_after_mean": float(ece_after_runs.mean()),
        "ece_after_std": float(ece_after_runs.std()),
        "n_outer_evaluations": len(ece_before_runs),
        "improved_fraction": float(np.mean(ece_after_runs < ece_before_runs)),
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


def _build_embedding_cache(emails: list[Email], model_name: str = None) -> dict[str, np.ndarray]:
    """Encode every email once so cross-validation and the nested calibration
    check, which together refit the embedding classifier well over a hundred
    times, do not each re-run the transformer.
    """
    from sentence_transformers import SentenceTransformer

    from classifier_embedding import DEFAULT_MODEL

    model = SentenceTransformer(model_name or DEFAULT_MODEL)
    texts = [e.text for e in emails]
    vectors = EmbeddingClassifier._normalize(np.asarray(model.encode(texts)))
    return {e.email_id: vec for e, vec in zip(emails, vectors)}


def predict_test_set(
    model_cls,
    train_emails: list[Email],
    test_emails: list[Email],
    model_kwargs: dict | None = None,
    temperature: float | None = None,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    model_kwargs = model_kwargs or {}
    model = model_cls(**model_kwargs)
    model.fit(train_emails)
    pred_classes, probs = model.predict_proba(test_emails)

    if classes is not None:
        col_idx = [pred_classes.index(c) for c in classes]
        probs = probs[:, col_idx]
        pred_classes = classes

    if temperature is not None:
        probs = apply_temperature(probs, temperature)

    best_idx = probs.argmax(axis=1)
    rows = []
    for row_i, email in enumerate(test_emails):
        class_i = int(best_idx[row_i])
        rows.append(
            {
                "email_id": email.email_id,
                "predicted_category": pred_classes[class_i],
                "confidence_score": round(float(probs[row_i, class_i]), 4),
            }
        )
    return pd.DataFrame(rows)


def _format_metrics_report(model_name: str, metrics: dict) -> str:
    lines = [f"=== {model_name} ===", ""]
    ci = metrics["bootstrap_ci_95"]
    lines.append(
        f"Out-of-fold accuracy: {metrics['accuracy']:.3f} "
        f"(95% bootstrap CI {ci['low']:.3f} to {ci['high']:.3f})"
    )
    lines.append(f"Out-of-fold macro F1: {metrics['macro_f1']:.3f}")
    lines.append(
        f"Cross-validation: {metrics['n_folds']} folds x {metrics['n_repeats']} repeats, "
        f"pooled over {metrics['n_labeled']} labeled emails, wall time "
        f"{metrics['cv_wall_time_seconds']:.2f}s"
    )
    lines.append("")
    lines.append("Per-class precision, recall, F1:")
    for label, stats in metrics["per_class"].items():
        lines.append(
            f"  {label:<22} precision {stats['precision']:.2f}  recall {stats['recall']:.2f}  "
            f"f1 {stats['f1-score']:.2f}  support {int(stats['support'])}"
        )
    lines.append("")

    lines.append(
        f"Calibration before temperature scaling: ECE {metrics['ece_raw']:.3f}, "
        f"Brier {metrics['brier_raw']:.3f}"
    )
    for row in metrics["calibration_table_raw"]:
        lines.append(
            f"  {row['confidence_range']:<14} mean confidence {row['mean_confidence']:.2f}  "
            f"empirical accuracy {row['empirical_accuracy']:.2f}  n={row['n']}"
        )
    lines.append("")

    lines.append(
        f"Calibration after temperature scaling (T={metrics['temperature']:.3f}): "
        f"ECE {metrics['ece_calibrated']:.3f}, Brier {metrics['brier_calibrated']:.3f}"
    )
    for row in metrics["calibration_table_calibrated"]:
        lines.append(
            f"  {row['confidence_range']:<14} mean confidence {row['mean_confidence']:.2f}  "
            f"empirical accuracy {row['empirical_accuracy']:.2f}  n={row['n']}"
        )
    lines.append("")

    nested = metrics["nested_calibration_check"]
    lines.append(
        "Nested calibration check (temperature fit on inner folds only, measured on an "
        "untouched outer fold, confirming the gain above is not fit-on-the-same-data leakage):"
    )
    lines.append(f"  ECE before: {nested['ece_before_mean']:.3f} +/- {nested['ece_before_std']:.3f}")
    lines.append(
        f"  ECE after:  {nested['ece_after_mean']:.3f} +/- {nested['ece_after_std']:.3f} "
        f"(improved in {nested['improved_fraction'] * 100:.0f} percent of "
        f"{nested['n_outer_evaluations']} outer evaluations)"
    )
    lines.append("")

    lines.append("Risk-coverage on calibrated confidence (auto-route versus human review):")
    for row in metrics["risk_coverage"]:
        acc = f"{row['accuracy_on_routed']:.3f}" if row["accuracy_on_routed"] is not None else "n/a"
        lines.append(
            f"  threshold {row['threshold']:.2f}: coverage {row['coverage']:.2f} "
            f"({row['n_routed']}/{metrics['n_labeled']})  accuracy on routed {acc}  "
            f"misroutes {row['misroutes']}"
        )
    lines.append("")

    lines.append(f"Cost-weighted error (illustrative weights): {metrics['cost_weighted_error']:.3f} per email")
    lines.append("")

    if metrics["misclassified"]:
        lines.append("Misclassified emails (out-of-fold, calibrated confidence):")
        for row in metrics["misclassified"]:
            lines.append(
                f"  true={row['true_category']:<20} predicted={row['predicted_category']:<20} "
                f"confidence={row['confidence']:.3f}  \"{row['text_snippet']}\""
            )
    else:
        lines.append("No misclassified emails.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    train_emails = load_train_set(DATA_DIR / "train", DATA_DIR / "train_labels.csv")
    test_emails = load_test_set(DATA_DIR / "test")

    print("Building sentence-embedding cache for all labeled and test emails...")
    embedding_cache = _build_embedding_cache(train_emails + test_emails)

    model_configs = {
        "baseline_tfidf_logreg": (BaselineClassifier, {}),
        "embedding_centroid": (EmbeddingClassifier, {"embedding_cache": embedding_cache}),
    }

    label_counts = Counter(e.true_category for e in train_emails)
    conformal_info = {
        "min_labels_per_class_90pct": conformal_min_n(0.10),
        "min_labels_per_class_95pct": conformal_min_n(0.05),
        "actual_labels_per_class": dict(sorted(label_counts.items())),
        "smallest_class_count": min(label_counts.values()),
    }

    report_sections = [
        "Minimum labeled examples per class for a valid class-conditional conformal "
        f"guarantee: {conformal_info['min_labels_per_class_90pct']} at 90 percent coverage, "
        f"{conformal_info['min_labels_per_class_95pct']} at 95 percent coverage. "
        f"Smallest class in the training data has {conformal_info['smallest_class_count']} "
        "examples, so a formal conformal guarantee is not yet feasible at either level.",
        "",
    ]

    all_metrics: dict = {"conformal_feasibility": conformal_info}
    comparison_rows = []

    for model_name, (model_cls, kwargs) in model_configs.items():
        print(f"\n=== {model_name}: {N_FOLDS}-fold x {N_REPEATS}-repeat cross-validation ===")
        metrics = cross_validate(model_cls, train_emails, model_kwargs=kwargs)

        print(f"--- {model_name}: nested calibration check (leakage safety) ---")
        metrics["nested_calibration_check"] = nested_temperature_check(
            model_cls, train_emails, model_kwargs=kwargs
        )

        all_metrics[model_name] = metrics
        print(
            f"accuracy={metrics['accuracy']:.3f} "
            f"(CI {metrics['bootstrap_ci_95']['low']:.3f}-{metrics['bootstrap_ci_95']['high']:.3f})  "
            f"macro_f1={metrics['macro_f1']:.3f}  "
            f"ece_raw={metrics['ece_raw']:.3f}  ece_calibrated={metrics['ece_calibrated']:.3f}  "
            f"cv_time={metrics['cv_wall_time_seconds']:.2f}s"
        )

        print(f"--- {model_name}: fitting on full train set, predicting {len(test_emails)} test emails ---")
        predictions = predict_test_set(
            model_cls,
            train_emails,
            test_emails,
            model_kwargs=kwargs,
            temperature=metrics["temperature"],
            classes=metrics["classes"],
        )
        out_path = RESULTS_DIR / f"predictions_{model_name}.csv"
        predictions.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

        if model_name == PRIMARY_MODEL:
            primary_path = RESULTS_DIR / "predictions.csv"
            predictions.to_csv(primary_path, index=False)
            print(f"wrote {primary_path} (primary submission, model={model_name})")

        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "accuracy_ci_low": metrics["bootstrap_ci_95"]["low"],
                "accuracy_ci_high": metrics["bootstrap_ci_95"]["high"],
                "macro_f1": metrics["macro_f1"],
                "ece_raw": metrics["ece_raw"],
                "ece_calibrated": metrics["ece_calibrated"],
                "temperature": metrics["temperature"],
                "cv_wall_time_seconds": metrics["cv_wall_time_seconds"],
            }
        )
        report_sections.append(_format_metrics_report(model_name, metrics))

    comparison = pd.DataFrame(comparison_rows)
    print(f"\n=== Comparison ({N_FOLDS}-fold x {N_REPEATS}-repeat CV on the 44 labeled train emails) ===")
    print(comparison.to_string(index=False))

    metrics_path = RESULTS_DIR / "evaluation_results.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    report_text = "\n".join(report_sections)
    report_path = RESULTS_DIR / "evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\nwrote {metrics_path}")
    print(f"wrote {RESULTS_DIR / 'comparison.csv'}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
