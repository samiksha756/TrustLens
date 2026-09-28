"""Generate the TrustLens layered architecture diagram into ./assets."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 6.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333", fs=9, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=fc, ec=ec, lw=1.4, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", color="#555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, lw=1.3, color=color))

# Layer labels
for y, lab in [(9.9, "INPUT"), (8.2, "FRONTEND"), (6.6, "BACKEND"),
               (3.4, "MODELS / ORCHESTRATION"), (1.4, "OUTPUT")]:
    ax.text(0.1, y, lab, fontsize=8, color="#888", rotation=90, va="center")

# Input layer
box(1.2, 9.4, 2.4, 1.0, "Typed text", "#e3f2fd")
box(3.9, 9.4, 2.4, 1.0, "Screenshot\n(planned→now)", "#fff3e0", dashed=True)
box(6.6, 9.4, 2.4, 1.0, "Voice note\n(optional)", "#fce4ec", dashed=True)

# Frontend
box(1.2, 7.9, 7.8, 0.9, "Web UI  —  tabs, upload controls, result cards  (HTML / CSS / JS)", "#f5f5f5")

# Backend
box(1.2, 6.3, 7.8, 0.9, "FastAPI  —  /api/analyse · /api/analyse-image · /api/analyse-audio", "#eeeeee")

# Orchestrator band
box(1.2, 5.1, 7.8, 0.8, "orchestrator.py  —  every modality reduced to text → shared pipeline", "#ede7f6")

# Models row
box(0.9, 3.5, 2.0, 1.2, "Model 1\nPhishing\nclassifier\n(text)", "#c8e6c9")
box(3.05, 3.5, 2.0, 1.2, "Model 2\nEmotion /\nmanipulation\n(text)", "#c8e6c9")
box(5.2, 3.5, 2.0, 1.2, "Model 3\nTesseract OCR\n(image)", "#ffe0b2")
box(7.35, 3.5, 1.75, 1.2, "Model 4\nWhisper STT\n(audio)", "#f8bbd0")

box(2.4, 2.1, 2.6, 0.9, "Indicator extractor\n(rule-based evidence)", "#fff9c4")
box(5.2, 2.1, 3.0, 0.9, "Risk aggregator\n(fuse signals → level)", "#d1c4e9")

# Output
box(1.2, 0.5, 7.8, 0.9, "Risk report  —  level · indicators · reasoning · safe action · disclaimer", "#b3e5fc")

# Arrows input→frontend
for x in (2.4, 5.1, 7.8):
    arrow(x, 9.4, x, 8.8)
arrow(5.1, 7.9, 5.1, 7.2)         # frontend→backend
arrow(5.1, 6.3, 5.1, 5.9)         # backend→orchestrator
# orchestrator→models
for x in (1.9, 4.05, 6.2, 8.2):
    arrow(x, 5.1, x, 4.7)
# models→aggregator (via indicators/aggregator)
arrow(3.7, 3.5, 3.7, 3.0)
arrow(6.7, 3.5, 6.7, 3.0)
arrow(3.7, 2.1, 5.2, 1.7, color="#555")   # indicators→aggregator area
arrow(6.7, 2.1, 6.7, 1.4)                 # aggregator→output

ax.text(5, 10.7, "TrustLens — System Architecture", ha="center", fontsize=13, weight="bold")
ax.text(5, 0.15, "Dashed = image/audio routes (multimodal extension).  Solid = text route.",
        ha="center", fontsize=7.5, color="#666")

fig.tight_layout()
fig.savefig(f"{ASSETS}/fig_architecture.png", dpi=150, bbox_inches="tight")
print("Wrote", f"{ASSETS}/fig_architecture.png")
