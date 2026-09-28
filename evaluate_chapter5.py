"""
TrustLens - Chapter 5 evaluation runner
=======================================
Runs every technical evaluation used in Chapter 5 with the REAL models and saves
the results as CSV tables and PNG figures in ./eval_outputs.

    python evaluate_chapter5.py            # full run (text, OCR, voice, timing)
    python evaluate_chapter5.py --quick    # skip the slow word-attribution timing

What it measures (sections refer to Chapter 5):
  E1  Detection on the 25-message set, default vs calibrated path (with emotion)
  E2  Accuracy per message category, both paths
  E3  Classifier saturation and calibration (ECE, Brier) for T = 1.0, 1.8, fitted
  E4  Emotion-model separation: pressure score, phishing vs benign
  E5  Fusion-weight sensitivity (classifier weight 0.40-0.80)
  E6  OCR accuracy: character error rate (CER) on rendered message images, and
      whether the screenshot route gives the same risk level as the text route
  E7  Whisper accuracy: word error rate (WER) on your recorded voice notes
      (put audio files + references.csv in ./eval_inputs/voice/ - see below)
  E8  Processing time per route (text, text + word explanation, screenshot, voice)

Voice-note input (E7)
---------------------
Record a few messages (e.g. 5-10) in QuickTime and save them in eval_inputs/voice/.
Create eval_inputs/voice/references.csv with two columns:
    filename,reference
    msg01.m4a,"Your PayPal account has been locked. Verify your password now."
The reference is exactly what you read aloud.

Optional real screenshots (E6)
------------------------------
Put phone screenshots in eval_inputs/screenshots/ with a references.csv in the
same format (filename,reference = the exact text shown in the screenshot).
"""

import argparse
import csv
import os
import re
import statistics
import sys
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaluate import TEST_CASES                                    # noqa: E402
from orchestrator import analyse_text, analyse_image, analyse_audio  # noqa: E402
from aggregator import THRESHOLDS                                  # noqa: E402
from calibration import (expected_calibration_error, brier_score,  # noqa: E402
                         fit_temperature, soften_probability)

OUT = os.path.join(HERE, "eval_outputs")
IN_VOICE = os.path.join(HERE, "eval_inputs", "voice")
IN_SHOTS = os.path.join(HERE, "eval_inputs", "screenshots")
os.makedirs(OUT, exist_ok=True)

# Figure style (validated two-colour palette; text in neutral ink)
INK, MUTED, GRID = "#1f2430", "#5a6472", "#e3e6ec"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b8bec9",
                     "axes.spines.top": False, "axes.spines.right": False})

LEVELS = ["Safe", "Caution", "Suspicious", "High Risk"]
POSITIVE = ("Suspicious", "High Risk")
SUMMARY = []                     # lines printed at the end and saved to summary.txt


def say(line=""):
    print(line)
    SUMMARY.append(line)


def binary(level):
    return 1 if level in POSITIVE else 0


def metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": round((tp + tn) / len(y_true), 3),
            "Precision": round(prec, 3), "Recall": round(rec, 3),
            "F1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "FPR": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def level_from_score(score):
    for threshold, name in THRESHOLDS:
        if score < threshold:
            return name
    return "High Risk"


# ----------------------------------------------------------------- text normalisation
def _norm_words(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9$£@./:'\-# ]+", " ", s)
    return [w.strip(".,:;'") for w in s.split() if w.strip(".,:;'")]


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    r, h = _norm_words(reference), _norm_words(hypothesis)
    return _edit_distance(r, h) / max(1, len(r))


def cer(reference, hypothesis):
    r = " ".join(_norm_words(reference))
    h = " ".join(_norm_words(hypothesis))
    return _edit_distance(list(r), list(h)) / max(1, len(r))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# ======================================================================= E1-E4
def run_text_paths():
    say("=" * 70)
    say("E1-E4  Text route on the 25-message test set (live models)")
    say("=" * 70)

    # Warm-up: load both models once so later timings exclude loading.
    _, load_s = timed(analyse_text, "Warm-up message to load the models.")
    say(f"First call (includes model loading): {load_s:.1f} s")

    rows = []
    for text, label, category in TEST_CASES:
        d, t_d = timed(analyse_text, text, calibrate=False)
        c, t_c = timed(analyse_text, text, calibrate=True)
        emo = d.get("emotion", {})
        rows.append({
            "message": text[:70] + ("..." if len(text) > 70 else ""),
            "category": category, "true": label,
            "raw_prob": d["assessment"]["classifier_prob"],
            "indicators": d["assessment"]["indicator_count"],
            "emotion_available": bool(emo.get("available")),
            "emotion_dominant": emo.get("dominant"),
            "pressure": emo.get("pressure_score", float("nan")),
            "link_risk": d["links"]["link_risk"],
            "benign_context": d["benign_context"]["benign_context"],
            "default_score": d["assessment"]["score"],
            "default_level": d["assessment"]["level"],
            "calibrated_score": c["assessment"]["score"],
            "calibrated_level": c["assessment"]["level"],
            "text_seconds": round(t_d, 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "E1_per_message_results.csv"), index=False)

    # ---- E1 detection metrics, both paths
    y = df["true"].tolist()
    m_def = metrics(y, [binary(l) for l in df["default_level"]])
    m_cal = metrics(y, [binary(l) for l in df["calibrated_level"]])
    table = pd.DataFrame([{"Path": "Default", **m_def}, {"Path": "Calibrated", **m_cal}])
    table.to_csv(os.path.join(OUT, "E1_detection_metrics.csv"), index=False)
    say("\nE1  Detection metrics (Suspicious/High Risk counted as 'phishing')")
    say(table.to_string(index=False))
    if not df["emotion_available"].all():
        say("NOTE: the emotion model was unavailable for some messages.")

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, (name, m) in zip(axes, [("Default path", m_def), ("Calibrated path", m_cal)]):
        mat = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(13, m["TP"]))
        for i in range(2):
            for j in range(2):
                v = mat[i][j]
                ax.text(j, i, f"{['TN','FP','FN','TP'][i*2+j]}\n{v}", ha="center",
                        va="center", fontsize=12, color="white" if v > 8 else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. benign", "Pred. phishing"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True benign", "True phishing"])
        ax.set_title(f"{name}\nRecall {m['Recall']:.2f} · FPR {m['FPR']:.2f}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_E1_confusion_both_paths.png"), dpi=300)
    plt.close(fig)

    # ---- E2 per-category accuracy
    df["default_correct"] = [binary(l) == t for l, t in zip(df["default_level"], df["true"])]
    df["calibrated_correct"] = [binary(l) == t for l, t in zip(df["calibrated_level"], df["true"])]
    cat = df.groupby("category").agg(n=("true", "size"),
                                     default_correct=("default_correct", "sum"),
                                     calibrated_correct=("calibrated_correct", "sum")).reset_index()
    cat.to_csv(os.path.join(OUT, "E2_category_accuracy.csv"), index=False)
    say("\nE2  Correct per category")
    say(cat.to_string(index=False))
    wrong = df[~df["default_correct"] | ~df["calibrated_correct"]]
    if len(wrong):
        say("\nMessages misclassified on at least one path:")
        for _, r in wrong.iterrows():
            say(f"  [{r['category']}] default={r['default_level']} calibrated={r['calibrated_level']}"
                f" p={r['raw_prob']} | {r['message']}")

    # ---- E3 saturation + calibration
    probs, labels = df["raw_prob"].tolist(), df["true"].tolist()
    saturated = sum(1 for p in probs if p <= 0.02 or p >= 0.98) / len(probs)
    t_fit = fit_temperature(probs, labels)
    cal_rows = []
    for t, note in [(1.0, "raw (no scaling)"), (1.8, "TrustLens default"),
                    (t_fit, "fitted on this set (in-sample)")]:
        soft = [soften_probability(p, t) for p in probs]
        cal_rows.append({"Temperature": t, "Note": note,
                         "ECE": expected_calibration_error(soft, labels),
                         "Brier": brier_score(soft, labels)})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(OUT, "E3_calibration.csv"), index=False)
    say(f"\nE3  Saturation: {saturated:.0%} of classifier outputs are <=0.02 or >=0.98")
    say(cal.to_string(index=False))

    # ---- E4 emotion separation
    if df["emotion_available"].any():
        e = df[df["emotion_available"]]
        mp = e[e["true"] == 1]["pressure"].mean()
        mb = e[e["true"] == 0]["pressure"].mean()
        emo_cat = e.groupby("category")["pressure"].mean().round(3).reset_index()
        emo_cat.to_csv(os.path.join(OUT, "E4_emotion_by_category.csv"), index=False)
        say(f"\nE4  Mean emotional pressure: phishing {mp:.3f} | benign {mb:.3f} | gap {mp-mb:+.3f}")
        say("    Dominant emotion counts (phishing): "
            + str(e[e['true'] == 1]["emotion_dominant"].value_counts().to_dict()))
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for k, (lab, col) in enumerate([(1, ORANGE), (0, BLUE)]):
            vals = e[e["true"] == lab]["pressure"].tolist()
            jitter = [k + (i % 5 - 2) * 0.04 for i in range(len(vals))]
            ax.scatter(vals, jitter, s=60, color=col, edgecolor="white", linewidth=1.5, zorder=3)
            ax.plot([statistics.mean(vals)] * 2, [k - 0.25, k + 0.25], color=INK, lw=2)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Phishing (n=13)", "Benign (n=12)"])
        ax.set_xlim(0, 1); ax.set_xlabel("Emotional-pressure score (fear + anger + surprise + disgust)")
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
        ax.set_title("Emotion model: pressure score by class (black line = mean)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_E4_emotion_pressure.png"), dpi=300)
        plt.close(fig)
    else:
        say("\nE4  Emotion model unavailable - separation not measured.")
    return df


# ======================================================================= E5
def run_weight_sensitivity(df):
    say("\n" + "=" * 70)
    say("E5  Fusion-weight sensitivity (default path, recorded live values)")
    say("=" * 70)
    y = df["true"].tolist()
    rows = []
    for w in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # Keep the emotion weight at 0.15 and give the rest to the rule indicators.
        w_ind = round(0.85 - w, 2)
        preds = []
        for _, r in df.iterrows():
            ind = min(r["indicators"] / 4, 1.0)
            ep = r["pressure"] if r["emotion_available"] else 0.0
            score = w * r["raw_prob"] + w_ind * ind + 0.15 * ep
            preds.append(binary(level_from_score(score)))
        m = metrics(y, preds)
        rows.append({"classifier_weight": w, "indicator_weight": w_ind, "emotion_weight": 0.15, **m})
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(OUT, "E5_weight_sensitivity.csv"), index=False)
    say(sens[["classifier_weight", "indicator_weight", "Recall", "FPR", "Accuracy", "FP", "FN"]]
        .to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(sens["classifier_weight"], sens["Recall"], marker="o", ms=7, color=BLUE, lw=2, label="Recall")
    ax.plot(sens["classifier_weight"], sens["FPR"], marker="s", ms=7, color=ORANGE, lw=2,
            label="False-positive rate")
    ax.axvline(0.55, color=MUTED, ls="--", lw=1)
    ax.text(0.555, 0.5, "chosen weight 0.55", color=MUTED, fontsize=9)
    ax.set_xlabel("Classifier weight (indicator weight = 0.85 − classifier weight)")
    ax.set_ylabel("Rate"); ax.set_ylim(-0.05, 1.05)
    ax.grid(color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_E5_weight_sensitivity.png"), dpi=300)
    plt.close(fig)


# ======================================================================= E6
def _render_message_image(text):
    """Draw a message as a phone-style chat bubble (a controlled OCR input)."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    font = None
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            font = ImageFont.truetype(path, 30)
            break
    if font is None:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, 38)
    w, h = 750, 120 + 44 * len(lines)
    img = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([30, 30, w - 30, h - 30], radius=28, fill="#e9e9eb")
    for i, line in enumerate(lines):
        d.text((60, 58 + 44 * i), line, fill="#111111", font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _read_refs(folder):
    path = os.path.join(folder, "references.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["filename"].strip(), r["reference"].strip()) for r in csv.DictReader(f)]


def run_ocr(df_text):
    say("\n" + "=" * 70)
    say("E6  OCR accuracy and screenshot-route consistency")
    say("=" * 70)
    text_level = dict(zip(df_text["message"], df_text["calibrated_level"]))
    rows = []
    items = [("rendered", t, _render_message_image(t)) for t, _, _ in TEST_CASES]
    for fname, ref in _read_refs(IN_SHOTS):
        with open(os.path.join(IN_SHOTS, fname), "rb") as f:
            items.append((f"real:{fname}", ref, f.read()))
    for kind, ref, img in items:
        res, secs = timed(analyse_image, img, calibrate=True)
        if "error" in res and "assessment" not in res:
            rows.append({"source": kind, "reference": ref[:60], "error": res["error"]})
            continue
        got = res["ocr"]["text"]
        key = ref[:70] + ("..." if len(ref) > 70 else "")
        txt_level = text_level.get(key)
        rows.append({"source": kind, "reference": ref[:60],
                     "CER": round(cer(ref, got), 4), "WER": round(wer(ref, got), 4),
                     "image_level": res["assessment"]["level"], "text_level": txt_level,
                     "same_level": (txt_level == res["assessment"]["level"]) if txt_level else None,
                     "seconds": round(secs, 2)})
    ocr = pd.DataFrame(rows)
    ocr.to_csv(os.path.join(OUT, "E6_ocr_results.csv"), index=False)
    if "CER" not in ocr or ocr["CER"].isna().all():
        say("OCR unavailable: " + str(ocr.get("error", pd.Series(["?"])).iloc[0]))
        return ocr
    ok = ocr.dropna(subset=["CER"])
    say(f"Images analysed: {len(ok)}  | mean CER {ok['CER'].mean():.3f}"
        f" | mean WER {ok['WER'].mean():.3f} | max CER {ok['CER'].max():.3f}")
    same = ok["same_level"].dropna()
    if len(same):
        say(f"Same risk level as the text route: {int(same.sum())}/{len(same)}")
    say(f"Mean time per screenshot: {ok['seconds'].mean():.2f} s")
    return ocr


# ======================================================================= E7
def run_whisper():
    say("\n" + "=" * 70)
    say("E7  Whisper word error rate on recorded voice notes")
    say("=" * 70)
    refs = _read_refs(IN_VOICE)
    if not refs:
        say(f"No voice notes found. Add audio + references.csv to {IN_VOICE} and re-run.")
        return pd.DataFrame()
    rows = []
    for fname, ref in refs:
        path = os.path.join(IN_VOICE, fname)
        with open(path, "rb") as f:
            data = f.read()
        suffix = os.path.splitext(fname)[1] or ".wav"
        res, secs = timed(analyse_audio, data, suffix=suffix, calibrate=True)
        if "error" in res and "assessment" not in res:
            rows.append({"file": fname, "error": res["error"]})
            continue
        got = res["stt"]["text"]
        text_res = analyse_text(ref, calibrate=True)
        rows.append({"file": fname, "reference": ref, "transcript": got,
                     "WER": round(wer(ref, got), 4), "CER": round(cer(ref, got), 4),
                     "voice_level": res["assessment"]["level"],
                     "text_level": text_res["assessment"]["level"],
                     "same_level": res["assessment"]["level"] == text_res["assessment"]["level"],
                     "seconds": round(secs, 2)})
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(OUT, "E7_whisper_results.csv"), index=False)
    if "WER" in v and v["WER"].notna().any():
        ok = v.dropna(subset=["WER"])
        say(f"Voice notes: {len(ok)} | mean WER {ok['WER'].mean():.3f} | max WER {ok['WER'].max():.3f}")
        say(f"Same risk level as reading the reference as text: {int(ok['same_level'].sum())}/{len(ok)}")
        say(f"Mean time per voice note: {ok['seconds'].mean():.2f} s")
        for _, r in ok[ok["WER"] > 0].iterrows():
            say(f"  {r['file']}: WER {r['WER']:.2f} | heard: {r['transcript'][:80]}")
    else:
        say("Whisper unavailable: " + str(v.get("error", pd.Series(["?"])).iloc[0]))
    return v


# ======================================================================= E8
def run_timing(df_text, ocr, voice, quick):
    say("\n" + "=" * 70)
    say("E8  Processing time per route (models already loaded)")
    say("=" * 70)
    rows = [{"Route": "Text (default path)", "n": len(df_text),
             "mean_s": df_text["text_seconds"].mean(), "max_s": df_text["text_seconds"].max()}]
    if not quick:
        secs, words = [], []
        for text, _, _ in TEST_CASES[:5] + TEST_CASES[13:15]:
            _, s = timed(analyse_text, text, calibrate=True, explain=True)
            secs.append(s)
            words.append(len(text.split()))
        rows.append({"Route": "Text + word explanation", "n": len(secs),
                     "mean_s": statistics.mean(secs), "max_s": max(secs)})
        say(f"Word explanation: mean {statistics.mean(secs):.1f} s for messages of "
            f"{min(words)}-{max(words)} words")
    if "seconds" in ocr and ocr["seconds"].notna().any():
        rows.append({"Route": "Screenshot (OCR)", "n": int(ocr["seconds"].notna().sum()),
                     "mean_s": ocr["seconds"].mean(), "max_s": ocr["seconds"].max()})
    if len(voice) and "seconds" in voice and voice["seconds"].notna().any():
        rows.append({"Route": "Voice note (Whisper)", "n": int(voice["seconds"].notna().sum()),
                     "mean_s": voice["seconds"].mean(), "max_s": voice["seconds"].max()})
    t = pd.DataFrame(rows).round(2)
    t.to_csv(os.path.join(OUT, "E8_timing.csv"), index=False)
    say(t.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * len(t)))
    ax.barh(t["Route"], t["mean_s"], color=BLUE, height=0.55)
    for i, v in enumerate(t["mean_s"]):
        ax.text(v, i, f"  {v:.1f} s", va="center", color=INK, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Mean seconds per analysis (models loaded)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.set_xlim(0, max(t["mean_s"].max(), 0.1) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_E8_timing.png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip word-explanation timing")
    args = ap.parse_args()
    import platform
    say(f"Machine: {platform.platform()} | Python {platform.python_version()}")
    df = run_text_paths()
    run_weight_sensitivity(df)
    ocr = run_ocr(df)
    voice = run_whisper()
    run_timing(df, ocr, voice, args.quick)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY))
    print(f"\nAll tables and figures saved in {OUT}")


if __name__ == "__main__":
    main()
