"""
TrustLens — Test Suite
----------------------
Runs without any model download (the transformer classifier is stubbed where
needed), so it works in CI and offline. Its most important job is REGRESSION
PROTECTION: `test_backward_compat_*` assert that the default scoring path still
reproduces, to 4 decimal places, the exact numbers recorded in
`evaluation_results.csv` — i.e. the advanced features added on top have not
changed the original behaviour.

Run:  pytest -q            (or)   python tests/test_trustlens.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from aggregator import (aggregate_risk, aggregate_risk_calibrated,
                        generate_explanation)
from indicators import extract_indicators
from link_analysis import analyse_links, analyse_url
from context_signals import detect_benign_context
import calibration

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "evaluation_results.csv")


def _dummy_indicators(n):
    return [{"name": f"ind{i}", "explanation": "x"} for i in range(int(n))]


# ----------------------------------------------------------- backward compatibility
def test_backward_compat_scores_reproduced():
    """Default text-only aggregation must reproduce every recorded score & level."""
    df = pd.read_csv(CSV)
    for _, row in df.iterrows():
        a = aggregate_risk(float(row["phishing_prob"]), _dummy_indicators(row["indicators"]))
        assert abs(a["score"] - float(row["score"])) < 1e-4, row["text"][:40]
        assert a["level"] == row["level"], row["text"][:40]


def test_text_only_weighting_exact():
    assert aggregate_risk(1.0, _dummy_indicators(4))["score"] == 1.0
    assert aggregate_risk(0.0, [])["score"] == 0.0
    # 0.65 * 1.0 + 0.35 * (2/4) = 0.825
    assert abs(aggregate_risk(1.0, _dummy_indicators(2))["score"] - 0.825) < 1e-9


def test_none_emotion_equals_original():
    a = aggregate_risk(0.8, _dummy_indicators(3))
    b = aggregate_risk(0.8, _dummy_indicators(3), emotion_pressure=None)
    assert a == b


def test_levels_boundaries():
    assert aggregate_risk(0.0, [])["level"] == "Safe"
    assert aggregate_risk(0.3, [])["level"] == "Safe"          # 0.3*0.65 = 0.195 -> Safe
    assert aggregate_risk(0.4, [])["level"] == "Caution"       # 0.26 -> Caution
    assert aggregate_risk(0.8, [])["level"] == "Suspicious"    # 0.52 -> Suspicious
    assert aggregate_risk(1.0, _dummy_indicators(4))["level"] == "High Risk"


def test_explanation_backward_compat():
    inds = _dummy_indicators(2)
    a = aggregate_risk(0.9, inds)
    exp = generate_explanation(a, inds, "preview text")   # no new kwargs
    for key in ("summary", "indicators", "emotion_note", "reasoning",
                "safe_action", "disclaimer"):
        assert key in exp
    assert "link_analysis" not in exp       # not added unless supplied
    assert "benign_context" not in exp


# ------------------------------------------------------------------ new modules
def test_indicators_detect_known():
    hits = {h["name"] for h in extract_indicators(
        "URGENT: verify your password at http://192.168.1.45/login")}
    assert "urgency" in hits and "credential_request" in hits and "suspicious_url_ip" in hits


def test_link_analysis_flags_and_clean():
    r = analyse_links("Verify at http://paypal.com.secure-login.tk/auth")
    names = {f["name"] for f in r["findings"]}
    assert "deceptive_subdomain" in names and "suspicious_tld" in names
    assert r["link_risk"] >= 0.8
    clean = analyse_links("Order update: https://www.amazon.co.uk/orders")
    assert clean["link_risk"] == 0.0
    assert analyse_url("http://192.168.1.45/login")["link_risk"] >= 0.9


def test_context_signals():
    r = detect_benign_context(
        "Your Amazon order #404-1234567-8901234 has shipped. No action is required. "
        "https://www.amazon.co.uk/orders")
    names = {s["name"] for s in r["signals"]}
    assert "order_reference" in names and "known_good_domain" in names
    assert r["benign_context"] > 0.5
    assert detect_benign_context("Hi mum, dinner at 7")["benign_context"] == 0.0


def test_calibration_softens_and_scores():
    assert calibration.soften_probability(0.999, 1.0) == 0.999          # T=1 is a no-op
    softened = calibration.soften_probability(0.999, 2.0)
    assert 0.5 < softened < 0.999                                       # pulled toward 0.5
    probs = [0.99, 0.98, 0.01, 0.999, 0.998]
    labels = [1, 1, 0, 1, 0]
    assert calibration.expected_calibration_error(probs, labels) >= 0.0
    assert calibration.brier_score(probs, labels) >= 0.0
    assert calibration.fit_temperature(probs, labels) > 0.0


def test_calibrated_aggregator_safety_floor():
    # Strong scam evidence + high benign context must NOT fall below Suspicious.
    a = aggregate_risk_calibrated(0.99, _dummy_indicators(4),
                                  link_risk=0.9, benign_context=1.0)
    assert a["level"] in ("Suspicious", "High Risk")
    assert a["calibrated"] is True
    assert "raw_classifier_prob" in a and "link_risk" in a


# ------------------------------------------------------------- orchestration wiring
def test_orchestrator_default_unchanged_and_calibrated_path():
    import orchestrator
    orchestrator.classify_text = lambda t: {
        "phishing_prob": 0.99 if "verify" in t.lower() else 0.02,
        "label": "phishing", "available": True}
    orchestrator.analyse_emotion = lambda t: {
        "available": False, "dominant": "unknown", "pressure_score": 0.0, "distribution": {}}

    msg = "Please verify your account at http://paypal.com.secure-login.tk/auth"
    default = orchestrator.analyse_text(msg)
    # Default path score must equal a direct aggregate_risk call (unchanged behaviour).
    inds = extract_indicators(msg)
    ref = aggregate_risk(0.99, inds)
    assert default["assessment"]["score"] == ref["score"]
    assert "calibrated" not in default["assessment"]

    cal = orchestrator.analyse_text(msg, calibrate=True)
    assert cal["assessment"].get("calibrated") is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
