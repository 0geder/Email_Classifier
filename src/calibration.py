"""Calibration and statistical-rigor utilities used by evaluate.py.

These functions turn a raw classifier confidence score into a trustworthy
routing signal and put honest uncertainty bounds on the headline metrics.
Each function is a small, independently testable piece: temperature scaling
(fit_temperature / apply_temperature), calibration measurement
(expected_calibration_error, multiclass_brier), sampling uncertainty
(bootstrap_accuracy_ci), the operating-point tradeoff for auto-routing versus
human review (risk_coverage_table), a feasibility check for a
distribution-free coverage guarantee (conformal_min_n), and a
business-weighted error metric (cost_weighted_error).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def fit_temperature(P: np.ndarray, y: list[str], classes: list[str]) -> float:
    """Fit a single scalar T that minimizes negative log-likelihood of the
    true labels under softmax(log(P) / T). T < 1 sharpens an under-confident
    model; T > 1 softens an over-confident one. The predicted class does not
    change, since this is a monotone rescaling of the same logits.
    """
    idx = np.array([classes.index(label) for label in y])
    log_p = np.log(np.clip(P, 1e-12, 1.0))

    def negative_log_likelihood(temperature: float) -> float:
        scaled = log_p / temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return -float(np.mean(np.log(probs[np.arange(len(y)), idx] + 1e-12)))

    result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(P: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability matrix by a fitted temperature."""
    scaled = np.log(np.clip(P, 1e-12, 1.0)) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    return probs / probs.sum(axis=1, keepdims=True)


def expected_calibration_error(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """Expected calibration error over top-label confidence: the gap between
    what the model claims (mean confidence in a bucket) and what actually
    happens (empirical accuracy in that bucket), weighted by bucket size.
    Returns the scalar ECE and a reliability table of the non-empty buckets.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(conf)
    ece = 0.0
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_last_bin = hi == edges[-1]
        mask = (conf >= lo) & (conf <= hi if is_last_bin else conf < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_confidence = float(conf[mask].mean())
        empirical_accuracy = float(correct[mask].mean())
        ece += (n / n_total) * abs(mean_confidence - empirical_accuracy)
        bracket = "]" if is_last_bin else ")"
        table.append(
            {
                "confidence_range": f"[{lo:.2f}, {hi:.2f}{bracket}",
                "n": n,
                "mean_confidence": round(mean_confidence, 4),
                "empirical_accuracy": round(empirical_accuracy, 4),
            }
        )
    return ece, table


def multiclass_brier(P: np.ndarray, y: list[str], classes: list[str]) -> float:
    """Mean squared error between predicted probability vectors and the
    one-hot true label, averaged over samples. Lower is better calibrated.
    """
    onehot = np.zeros_like(P)
    for row, label in zip(onehot, y):
        row[classes.index(label)] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def bootstrap_accuracy_ci(
    correct: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> tuple[float, float]:
    """95 percent bootstrap confidence interval on accuracy, resampling the
    per-sample correctness array with replacement. At 44 labeled examples,
    the point estimate alone overstates how precisely accuracy is known.
    """
    correct = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    resamples = rng.choice(correct, size=(n_boot, len(correct)), replace=True)
    boot_accuracies = resamples.mean(axis=1)
    lo, hi = np.percentile(boot_accuracies, [2.5, 97.5])
    return float(lo), float(hi)


def risk_coverage_table(
    conf: np.ndarray, correct: np.ndarray, thresholds: list[float]
) -> list[dict]:
    """For each candidate auto-routing confidence threshold: what fraction of
    emails clear it (coverage), how accurate the auto-routed ones are, and
    how many of them would have been misrouted. This is the table an
    operating threshold should be chosen from, not a single accuracy figure.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_total = len(conf)
    table = []
    for threshold in thresholds:
        mask = conf >= threshold
        n_routed = int(mask.sum())
        accuracy_on_routed = float(correct[mask].mean()) if n_routed > 0 else None
        misroutes = int((~correct[mask]).sum()) if n_routed > 0 else 0
        table.append(
            {
                "threshold": threshold,
                "coverage": round(n_routed / n_total, 4),
                "n_routed": n_routed,
                "accuracy_on_routed": round(accuracy_on_routed, 4)
                if accuracy_on_routed is not None
                else None,
                "misroutes": misroutes,
            }
        )
    return table


def conformal_min_n(alpha: float) -> int:
    """Minimum number of calibration examples per class needed for a valid
    class-conditional (Mondrian) conformal prediction guarantee at coverage
    1 - alpha. Split conformal prediction requires
    ceil((1 - alpha) * (n + 1)) <= n; below this n the required quantile is
    undefined and the prediction set degenerates to the full label space.
    """
    target = 1.0 - alpha
    n = 1
    while math.ceil(target * (n + 1)) > n:
        n += 1
    return n


def cost_weighted_error(
    y_true: list[str], y_pred: list[str], cost_map: dict[str, float]
) -> float:
    """Mean cost of the misclassifications, where the cost of an error is
    charged against the true category (misrouting a regulated request is
    assumed more expensive than misrouting a low-stakes one). Weights are
    illustrative and should be set with compliance, not engineering.
    """
    if not y_true:
        return 0.0
    total_cost = sum(
        cost_map.get(true, 1.0) for true, pred in zip(y_true, y_pred) if true != pred
    )
    return total_cost / len(y_true)
