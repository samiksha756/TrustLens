"""
TrustLens — Figure Generator
----------------------------
Builds the evaluation figures used in the final report directly from the real
per-message results in `evaluation_results.csv`. Running this regenerates every
chart, so the figures always match the data.

Outputs (into ./assets):
    fig_confusion_matrix.png
    fig_category_accuracy.png
    fig_prob_distribution.png
    fig_threshold_sweep.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "evaluation_results.csv"))

# ---------------------------------------------------------------- confusion matrix
tp = int(((df.true == 1) & (df.pred == 1)).sum())
tn = int(((df.true == 0) & (df.pred == 0)).sum())
fp = int(((df.true == 0) & (df.pred == 1)).sum())
fn = int(((df.true == 1) & (df.pred == 0)).sum())

fig, ax = plt.subplots(figsize=(4.2, 3.6))
mat = [[tn, fp], [fn, tp]]
ax.imshow(mat, cmap="Blues")
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{mat[i][j]}", ha="center", va="center",
                fontsize=13, color="black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Benign", "Pred Phishing"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Benign", "True Phishing"])
ax.set_title("TrustLens confusion matrix (n=25)")
fig.tight_layout(); fig.savefig(f"{ASSETS}/fig_confusion_matrix.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- per-category accuracy
cat = df.groupby("category")["correct"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6.4, 3.8))
colors = ["#c62828" if v < 1 else "#2e7d32" for v in cat.values]
ax.barh(cat.index, cat.values * 100, color=colors)
ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 105)
ax.set_title("Accuracy by message category")
for i, v in enumerate(cat.values):
    ax.text(v * 100 + 1, i, f"{v*100:.0f}%", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(f"{ASSETS}/fig_category_accuracy.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- probability distribution
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bins = [i / 20 for i in range(21)]
ax.hist([df[df.true == 1]["phishing_prob"], df[df.true == 0]["phishing_prob"]],
        bins=bins, label=["Phishing", "Benign"], color=["#c62828", "#2e7d32"])
ax.axvline(0.5, ls="--", color="#444", label="0.5 threshold")
ax.set_xlabel("Classifier phishing probability"); ax.set_ylabel("Count")
ax.set_title("Classifier probability: phishing vs benign (note saturation at 0 and 1)")
ax.annotate("2 benign msgs\nmisread as phishing", xy=(0.97, 2), xytext=(0.55, 6),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="#444"))
ax.legend()
fig.tight_layout(); fig.savefig(f"{ASSETS}/fig_prob_distribution.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- threshold sweep (model-specific)
# Evaluate Model 1 in isolation as a raw classifier across decision thresholds.
thresholds = [i / 100 for i in range(5, 100, 5)]
prec, rec, f1s, fprs = [], [], [], []
for t in thresholds:
    pred = (df.phishing_prob >= t).astype(int)
    tp_ = int(((df.true == 1) & (pred == 1)).sum())
    fp_ = int(((df.true == 0) & (pred == 1)).sum())
    fn_ = int(((df.true == 1) & (pred == 0)).sum())
    tn_ = int(((df.true == 0) & (pred == 0)).sum())
    p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0
    r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
    f = 2 * p * r / (p + r) if (p + r) else 0
    fpr = fp_ / (fp_ + tn_) if (fp_ + tn_) else 0
    prec.append(p); rec.append(r); f1s.append(f); fprs.append(fpr)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(thresholds, prec, marker="o", label="Precision")
ax.plot(thresholds, rec, marker="s", label="Recall")
ax.plot(thresholds, f1s, marker="^", label="F1")
ax.plot(thresholds, fprs, marker="x", label="False-positive rate")
ax.set_xlabel("Decision threshold on phishing probability")
ax.set_ylabel("Metric value")
ax.set_title("Model 1 (classifier) — metric vs threshold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{ASSETS}/fig_threshold_sweep.png", dpi=150); plt.close(fig)

print("Figures written to", ASSETS)
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Accuracy={ (tp+tn)/len(df):.3f}  Precision={ tp/(tp+fp):.3f}  "
      f"Recall={ tp/(tp+fn):.3f}  FPR={ fp/(fp+tn):.3f}")
