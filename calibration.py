"""
TrustLens — Confidence Calibration Module
------------------------------------------
The evaluation found that the phishing classifier is strongly *saturated*: its
probabilities sit at ~0.0 or ~1.0, so a benign message it gets wrong is wrong
*confidently*, and no decision-threshold change can fix it. This module adds a
principled response to that finding: **temperature scaling**, a standard,
post-hoc calibration technique (Guo et al., 2017) that softens over-confident
probabilities without changing the ranking of predictions.

Why this matters for TrustLens
------------------------------
A softened probability lets the multi-signal aggregator actually *move* a
borderline case — a saturated 0.999 that swamps every other signal becomes,
say, 0.86, so the emotion signal, link analysis and benign-context evidence can
influence the final level. Calibration is therefore the enabling step for the
orchestration to add value on the exact cases the classifier gets wrong.

Everything here is pure-Python/math (no model download), so it runs anywhere
and is fully unit-tested. It is **opt-in**: the default pipeline is unchanged,
preserving all previously reported numbers. Calibration is applied only when a
caller explicitly asks for it (orchestrator `calibrate=True`, or the CLI
`--calibrated` flag).
"""

import math

DEFAULT_TEMPERATURE = 1.8  # >1 softens; 1.0 is a no-op. Tunable via fit_temperature.
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def soften_probability(p: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Temperature-scale a single probability. temperature=1 returns p unchanged;
    temperature>1 pulls the value toward 0.5 (less over-confident) while
    preserving order (monotonic). Never flips a decision at the 0.5 boundary.
    """
    t = max(_EPS, float(temperature))
    return round(_sigmoid(_logit(p) / t), 4)


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """
    ECE: average gap between confidence and accuracy across probability bins.
    Lower is better-calibrated. `probs`/`labels` are equal-length sequences.
    """
    probs = [float(x) for x in probs]
    labels = [int(x) for x in labels]
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (p > lo or (b == 0 and p >= lo)) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return round(ece, 4)


def brier_score(probs, labels) -> float:
    """Mean squared error between predicted probability and label. Lower better."""
    probs = [float(x) for x in probs]
    labels = [int(x) for x in labels]
    if not probs:
        return 0.0
    return round(sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs), 4)


def _nll(probs, labels, t):
    loss = 0.0
    for p, y in zip(probs, labels):
        q = _sigmoid(_logit(p) / t)
        q = min(1.0 - _EPS, max(_EPS, q))
        loss += -(y * math.log(q) + (1 - y) * math.log(1.0 - q))
    return loss / max(1, len(probs))


def fit_temperature(probs, labels, grid=None) -> float:
    """
    Fit the scalar temperature that minimises negative log-likelihood on a
    held-out set (a simple, robust grid search — enough for a single scalar).
    Returns the best temperature (>0).
    """
    probs = [float(x) for x in probs]
    labels = [int(x) for x in labels]
    if not probs:
        return 1.0
    grid = grid or [round(0.5 + 0.1 * i, 1) for i in range(0, 46)]  # 0.5 .. 5.0
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        loss = _nll(probs, labels, t)
        if loss < best_loss:
            best_loss, best_t = loss, t
    return best_t


if __name__ == "__main__":
    # Saturated toy example: model is confident and mostly right, but 2 confident errors.
    probs = [0.999, 0.998, 1.0, 0.995, 0.002, 0.001, 0.999, 0.998]  # last 2 benign
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    print("Raw ECE  :", expected_calibration_error(probs, labels))
    print("Raw Brier:", brier_score(probs, labels))
    t = fit_temperature(probs, labels)
    print("Fitted temperature:", t)
    soft = [soften_probability(p, t) for p in probs]
    print("Softened ECE  :", expected_calibration_error(soft, labels))
    print("Softened Brier:", brier_score(soft, labels))
    print("Example: 0.999 ->", soften_probability(0.999, t))
