from dataclasses import replace

from evaluate import _calibration_table, cross_validate
from classifier_baseline import BaselineClassifier
from fixtures import SYNTHETIC_TRAIN

# nested_temperature_check needs at least 2 examples per class left over after
# the outer split before the inner cross-validation can stratify, so double
# up the 2-per-class synthetic fixture to 4 per class for that test only.
NESTED_CHECK_EMAILS = SYNTHETIC_TRAIN + [
    replace(e, email_id=f"{e.email_id}b", source_file=f"dup_{e.source_file}")
    for e in SYNTHETIC_TRAIN
]


def test_calibration_table_buckets_by_confidence():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    conf = [0.9, 0.4, 0.6, 0.95]

    table = _calibration_table(y_true, y_pred, conf)

    high_bin = next(b for b in table if b["confidence_range"] == "[0.85, 0.95)")
    assert high_bin["n"] == 1
    assert high_bin["empirical_accuracy"] == 1.0

    low_bin = next(b for b in table if b["confidence_range"] == "[0.00, 0.50)")
    assert low_bin["n"] == 1
    assert low_bin["empirical_accuracy"] == 0.0


def test_calibration_table_empty_bin_is_none():
    table = _calibration_table(["A"], ["A"], [0.99])
    empty_bin = next(b for b in table if b["confidence_range"] == "[0.50, 0.70)")
    assert empty_bin["n"] == 0
    assert empty_bin["empirical_accuracy"] is None


def test_cross_validate_returns_expected_shape():
    metrics = cross_validate(BaselineClassifier, SYNTHETIC_TRAIN, n_folds=2, n_repeats=2)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics.keys()) >= {
        "accuracy",
        "macro_f1",
        "per_class",
        "calibration",
        "bootstrap_ci_95",
        "ece_raw",
        "ece_calibrated",
        "temperature",
        "risk_coverage",
        "cost_weighted_error",
        "misclassified",
        "classes",
    }
    ci = metrics["bootstrap_ci_95"]
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0
    assert metrics["temperature"] > 0.0


def test_nested_temperature_check_returns_expected_shape():
    from evaluate import nested_temperature_check

    result = nested_temperature_check(
        BaselineClassifier,
        NESTED_CHECK_EMAILS,
        outer_splits=2,
        outer_repeats=1,
        inner_splits=2,
    )
    assert set(result.keys()) >= {
        "ece_before_mean",
        "ece_after_mean",
        "n_outer_evaluations",
        "improved_fraction",
    }
    assert result["n_outer_evaluations"] == 2
