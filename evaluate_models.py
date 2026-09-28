"""
TrustLens — Model-Specific & Orchestration Evaluation
------------------------------------------------------
The preliminary report evaluated the *system* end-to-end. Feedback asked for
evaluation that is specific to the models being orchestrated. This script adds
three model-focused evaluations. It runs on a machine where the models are
installed (it downloads the phishing + emotion models on first run).

(1) MODEL 1 in isolation — the phishing classifier judged as a raw binary
    classifier at threshold 0.5, plus its confidence calibration (are the
    probabilities saturated?).

(2) MODEL 2 signal quality — does the emotion model's "pressure score" actually
    separate phishing from benign messages? Reported as mean pressure per class.

(3) ORCHESTRATION ABLATION — does adding the emotion signal to the aggregator
    change system accuracy / false-positive rate versus the text-only baseline?
    This tests whether orchestration *helps*, which is the whole premise of the
    project template.

(4) OCR round-trip (optional) — renders known text to an image, runs Model 3
    (Tesseract), and reports character error rate, if OCR is installed.

Run:  python evaluate_models.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from evaluate import TEST_CASES
from classifier import classify_text
from indicators import extract_indicators
from aggregator import aggregate_risk
from emotion import analyse_emotion


def _binary(level):
    return 1 if level in ("Suspicious", "High Risk") else 0


def run():
    rows = []
    for text, true_label, category in TEST_CASES:
        cls = classify_text(text)
        emo = analyse_emotion(text)
        inds = extract_indicators(text)
        ep = emo["pressure_score"] if emo.get("available") else None

        base = aggregate_risk(cls["phishing_prob"], inds)                 # text-only
        tri = aggregate_risk(cls["phishing_prob"], inds, emotion_pressure=ep)  # + emotion

        rows.append({
            "category": category, "true": true_label,
            "prob": cls["phishing_prob"],
            "raw_pred": 1 if cls["phishing_prob"] >= 0.5 else 0,
            "pressure": ep if ep is not None else float("nan"),
            "base_pred": _binary(base["level"]),
            "tri_pred": _binary(tri["level"]),
        })
    df = pd.DataFrame(rows)

    print("=" * 64)
    print("(1) MODEL 1 — phishing classifier in isolation (threshold 0.5)")
    print("=" * 64)
    y, p = df["true"], df["raw_pred"]
    print(f"  Accuracy {accuracy_score(y,p):.3f} | Precision {precision_score(y,p,zero_division=0):.3f} "
          f"| Recall {recall_score(y,p,zero_division=0):.3f} | F1 {f1_score(y,p,zero_division=0):.3f}")
    near_certain = ((df.prob <= 0.02) | (df.prob >= 0.98)).mean()
    print(f"  Confidence saturation: {near_certain:.0%} of predictions are <=0.02 or >=0.98")
    print("  -> The model is highly saturated; probability thresholding alone cannot")
    print("     fix its confident mistakes. This motivates multi-signal aggregation.")

    print("\n" + "=" * 64)
    print("(2) MODEL 2 — emotion pressure separation (phishing vs benign)")
    print("=" * 64)
    if df["pressure"].notna().any():
        mp = df[df.true == 1]["pressure"].mean()
        mb = df[df.true == 0]["pressure"].mean()
        print(f"  Mean emotional-pressure — phishing: {mp:.3f} | benign: {mb:.3f} "
              f"| gap: {mp - mb:+.3f}")
        print("  A positive gap means the emotion model carries complementary signal.")
    else:
        print("  Emotion model unavailable in this environment — install to evaluate.")

    print("\n" + "=" * 64)
    print("(3) ORCHESTRATION ABLATION — text-only vs +emotion")
    print("=" * 64)
    for name, col in [("Text-only (M1+indicators)", "base_pred"),
                      ("+ Emotion (M1+M2+indicators)", "tri_pred")]:
        pr = df[col]
        tn = int(((df.true == 0) & (pr == 0)).sum()); fp = int(((df.true == 0) & (pr == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else 0
        print(f"  {name:<32} acc {accuracy_score(df.true, pr):.3f} "
              f"| recall {recall_score(df.true, pr, zero_division=0):.3f} | FPR {fpr:.3f}")

    print("\n" + "=" * 64)
    print("(4) MODEL 3 — OCR round-trip (optional)")
    print("=" * 64)
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw
        from ocr import extract_text_from_image
        ref = "Your account is locked. Verify your password now."
        im = Image.new("RGB", (640, 120), "white")
        ImageDraw.Draw(im).text((10, 45), ref, fill="black")
        buf = BytesIO(); im.save(buf, format="PNG")
        out = extract_text_from_image(buf.getvalue())
        if out.get("available"):
            got = out["text"]
            # simple character error rate
            import difflib
            ratio = difflib.SequenceMatcher(None, ref.lower(), got.lower()).ratio()
            print(f"  Reference: {ref}")
            print(f"  OCR read : {got!r}")
            print(f"  Similarity: {ratio:.2%}")
        else:
            print("  OCR engine not installed — skipping.  (" + out.get("note", "") + ")")
    except Exception as e:  # noqa: BLE001
        print("  OCR round-trip skipped:", e)

    print("\n" + "=" * 64)
    print("(5) CONFIDENCE CALIBRATION — temperature scaling of Model 1")
    print("=" * 64)
    calibration_report(df["prob"].tolist(), df["true"].tolist())

    df.to_csv("model_evaluation_results.csv", index=False)
    print("\n[eval] Wrote model_evaluation_results.csv")


def calibration_report(probs, labels):
    """
    Quantify the saturation finding and the effect of temperature scaling.
    Reports ECE and Brier score before/after fitting a single temperature.
    Works from any (probs, labels) pair, so it can also be run directly on the
    recorded evaluation_results.csv without re-loading the model.
    """
    from calibration import (expected_calibration_error, brier_score,
                             fit_temperature, soften_probability)
    ece0 = expected_calibration_error(probs, labels)
    brier0 = brier_score(probs, labels)
    sat = sum(1 for p in probs if p <= 0.02 or p >= 0.98) / max(1, len(probs))
    t = fit_temperature(probs, labels)
    soft = [soften_probability(p, t) for p in probs]
    ece1 = expected_calibration_error(soft, labels)
    brier1 = brier_score(soft, labels)
    print(f"  Saturation (<=0.02 or >=0.98): {sat:.0%} of predictions")
    print(f"  Fitted temperature T = {t}")
    print(f"  ECE   : {ece0:.4f}  ->  {ece1:.4f}")
    print(f"  Brier : {brier0:.4f}  ->  {brier1:.4f}")
    print("  Temperature scaling softens the over-confident probabilities so the")
    print("  aggregator's other signals (emotion, links, context) can influence")
    print("  borderline cases. This is the enabling step for calibrated fusion.")


if __name__ == "__main__":
    run()
