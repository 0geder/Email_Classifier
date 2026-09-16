from evaluate import _calibration_table, cross_validate
from classifier_baseline import BaselineClassifier
from fixtures import SYNTHETIC_TRAIN


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
    metrics = cross_validate(BaselineClassifier, SYNTHETIC_TRAIN, n_folds=2)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics.keys()) >= {"accuracy", "macro_f1", "per_class", "calibration"}
