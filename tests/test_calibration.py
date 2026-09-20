import numpy as np

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


def test_fit_temperature_sharpens_underconfident_model():
    # Every prediction is correct, but the raw softmax is spread thin across
    # 3 classes (max prob 0.5) -- an under-confident model, mirroring the
    # baseline TF-IDF classifier's behavior on the real data. Temperature
    # scaling should sharpen it (T < 1) toward the true, higher confidence.
    classes = ["A", "B", "C"]
    P = np.array([[0.5, 0.3, 0.2]] * 30)
    y = ["A"] * 30

    T = fit_temperature(P, y, classes)
    assert T < 1.0

    calibrated = apply_temperature(P, T)
    assert calibrated[:, 0].mean() > P[:, 0].mean()
    # temperature scaling is monotone: the argmax class is unchanged
    assert (calibrated.argmax(axis=1) == P.argmax(axis=1)).all()


def test_expected_calibration_error_matches_hand_computation():
    # One bucket [0.90, 1.00]: mean confidence 0.90, but only 1/2 correct.
    conf = np.array([0.9, 0.9])
    correct = np.array([True, False])
    ece, table = expected_calibration_error(conf, correct, n_bins=10)
    assert abs(ece - 0.4) < 1e-9
    assert len(table) == 1
    assert table[0]["n"] == 2


def test_multiclass_brier_zero_for_perfect_confident_predictions():
    classes = ["A", "B"]
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = ["A", "B"]
    assert multiclass_brier(P, y, classes) == 0.0


def test_bootstrap_accuracy_ci_bounds_are_ordered_and_valid():
    correct = np.array([True] * 40 + [False] * 4)
    lo, hi = bootstrap_accuracy_ci(correct, n_boot=500, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < correct.mean() < hi or lo <= correct.mean() <= hi


def test_risk_coverage_table_is_monotonic_in_threshold():
    conf = np.array([0.2, 0.4, 0.6, 0.8, 0.95, 0.99])
    correct = np.array([True, False, True, True, True, True])
    table = risk_coverage_table(conf, correct, [0.0, 0.5, 0.9])
    coverages = [row["coverage"] for row in table]
    assert coverages == sorted(coverages, reverse=True)
    assert table[0]["n_routed"] == 6
    assert table[-1]["n_routed"] == 2  # only 0.95 and 0.99 clear the 0.9 threshold


def test_conformal_min_n_matches_known_closed_form_values():
    assert conformal_min_n(0.10) == 9
    assert conformal_min_n(0.05) == 19


def test_cost_weighted_error_hand_computation():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "A"]  # positions 1 and 3 are wrong
    cost_map = {"A": 2.0, "B": 1.0}
    # wrong true labels: "A" (cost 2.0) and "B" (cost 1.0) -> total 3.0 / 4
    assert cost_weighted_error(y_true, y_pred, cost_map) == 0.75
