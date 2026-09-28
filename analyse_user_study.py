"""
TrustLens - user-study analysis for Chapter 5
=============================================
Reads the responses exported from the Google Form and produces the numbers,
tables and charts for the user-study section.

How to export: open the form -> Responses -> "Link to Sheets" (or the green
Sheets icon) -> in the sheet, File -> Download -> Comma-separated values (.csv).
Save it in the project folder as  user_study_responses.csv  and run:

    python analyse_user_study.py user_study_responses.csv

Outputs go to ./eval_outputs (US_*.csv, fig_US_*.png, user_study_summary.txt).
"""

import os
import sys
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_outputs")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1f2430", "#5a6472", "#e3e6ec"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b8bec9"})

# Correct answer for each test message (first letter of the grid row).
CORRECT = {"A": "Scam", "B": "Not a scam", "C": "Scam", "D": "Scam"}
AGREE = ["Disagree", "Partially disagree", "Neither agree nor disagree",
         "Partially agree", "Agree"]
# Diverging palette: two hues + neutral grey midpoint.
AGREE_COLOURS = ["#c2410c", "#f4a67c", "#c8ccd4", "#86b6ef", "#1c5cab"]

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def cols_with(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def row_label(col):
    return col.split("[", 1)[1].rstrip("]") if "[" in col else col


def sus_score(row, sus_cols):
    total = 0
    for i, c in enumerate(sus_cols, start=1):
        v = int(str(row[c]).strip()[0])            # "1 Strongly disagree" -> 1
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return total * 2.5


def main(path):
    df = pd.read_csv(path)
    n = len(df)
    say(f"Responses: {n}")

    # ---------------------------------------------------------- participants
    part = {}
    for key, label in [("Age group", "Age group"),
                       ("Do you work or study in computing", "Technical background"),
                       ("In the last 3 months", "Suspected scam in last 3 months")]:
        col = [c for c in df.columns if c.startswith(key)]
        if col:
            counts = df[col[0]].value_counts()
            part[label] = counts
            say(f"{label}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    conf_dig = [c for c in df.columns if c.startswith("How confident are you using")]
    if conf_dig:
        say(f"Digital confidence (1-5): mean {df[conf_dig[0]].mean():.2f}")

    # ---------------------------------------------------------- before / after
    before = cols_with(df, "Is each message a scam? (before")
    after = cols_with(df, "Is each message a scam? (after")
    rows = []
    for b_col, a_col in zip(before, after):
        letter = row_label(b_col).strip()[0]
        right = CORRECT[letter]
        b_ok = (df[b_col] == right).sum()
        a_ok = (df[a_col] == right).sum()
        b_unsure = (df[b_col] == "Not sure").sum()
        a_unsure = (df[a_col] == "Not sure").sum()
        rows.append({"Message": letter, "Correct answer": right,
                     "Correct before": f"{b_ok}/{n}", "Correct after": f"{a_ok}/{n}",
                     "Not sure before": int(b_unsure), "Not sure after": int(a_unsure),
                     "_b": b_ok, "_a": a_ok})
    ba = pd.DataFrame(rows)
    tot_b, tot_a = ba["_b"].sum(), ba["_a"].sum()
    ba.drop(columns=["_b", "_a"]).to_csv(os.path.join(OUT, "US_before_after.csv"), index=False)
    say("\nDecision accuracy before vs after using TrustLens")
    say(ba.drop(columns=["_b", "_a"]).to_string(index=False))
    say(f"Overall correct judgements: before {tot_b}/{4*n} ({tot_b/(4*n):.0%}) -> "
        f"after {tot_a}/{4*n} ({tot_a/(4*n):.0%})")
    per_person = []
    for _, r in df.iterrows():
        b = sum(r[c] == CORRECT[row_label(c).strip()[0]] for c in before)
        a = sum(r[c] == CORRECT[row_label(c).strip()[0]] for c in after)
        per_person.append(a - b)
    say(f"Participants who improved: {sum(d > 0 for d in per_person)}, "
        f"unchanged: {sum(d == 0 for d in per_person)}, worse: {sum(d < 0 for d in per_person)}")

    conf_b = [c for c in df.columns if c.startswith("Overall, how confident are you in these answers?")]
    conf_a = [c for c in df.columns if c.startswith("Overall, how confident are you in these answers now")]
    if conf_b and conf_a:
        say(f"Confidence (1-5): before {df[conf_b[0]].mean():.2f} -> after {df[conf_a[0]].mean():.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(ba))
    ax.bar([i - 0.2 for i in x], ba["_b"] / n * 100, width=0.38, color="#c8ccd4", label="Before TrustLens")
    ax.bar([i + 0.2 for i in x], ba["_a"] / n * 100, width=0.38, color="#2a78d6", label="After TrustLens")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m}\n({c})" for m, c in zip(ba["Message"], ba["Correct answer"])])
    ax.set_ylabel("Participants correct (%)"); ax.set_ylim(0, 110)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_US_before_after.png"), dpi=300); plt.close(fig)

    # ---------------------------------------------------------- SUS
    sus_cols = cols_with(df, "System Usability Scale")
    if len(sus_cols) == 10:
        scores = [sus_score(r, sus_cols) for _, r in df.iterrows()]
        pd.DataFrame({"participant": range(1, n + 1), "SUS": scores}).to_csv(
            os.path.join(OUT, "US_sus_scores.csv"), index=False)
        sd = statistics.stdev(scores) if n > 1 else 0.0
        say(f"\nSUS: mean {statistics.mean(scores):.1f}, median {statistics.median(scores):.1f}, "
            f"SD {sd:.1f}, range {min(scores):.1f}-{max(scores):.1f} (benchmark average = 68)")
        item_means = pd.DataFrame({"Statement": [row_label(c) for c in sus_cols],
                                   "Mean (1-5)": [round(df[c].astype(str).str[0].astype(int).mean(), 2)
                                                  for c in sus_cols]})
        item_means.to_csv(os.path.join(OUT, "US_sus_items.csv"), index=False)
        fig, ax = plt.subplots(figsize=(6.4, 1.9))
        ys, seen = [], {}
        for v in scores:                      # stack equal scores so none hide
            k = seen.get(v, 0); seen[v] = k + 1; ys.append(k * 0.22)
        ax.scatter(scores, ys, s=90, color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=3)
        ax.axvline(68, color=MUTED, ls="--", lw=1); ax.text(68.5, 0.3, "benchmark 68", color=MUTED, fontsize=9)
        ax.axvline(statistics.mean(scores), color=INK, lw=2)
        ax.text(statistics.mean(scores) - 0.8, -0.35, f"mean {statistics.mean(scores):.1f}", fontsize=9, ha="right")
        ax.set_xlim(0, 100); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.set_xlabel("SUS score per participant (0-100)")
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_US_sus.png"), dpi=300); plt.close(fig)
    else:
        say(f"\nSUS columns found: {len(sus_cols)} (expected 10) - check the export.")

    # ---------------------------------------------------------- feature statements
    feat = cols_with(df, "How far do you agree with each statement?")
    if feat:
        counts = pd.DataFrame({row_label(c): df[c].value_counts().reindex(AGREE, fill_value=0)
                               for c in feat}).T
        counts.to_csv(os.path.join(OUT, "US_feature_statements.csv"))
        agree_share = ((counts["Agree"] + counts["Partially agree"]) / n * 100).round(0)
        say("\nFeature statements (% agree or partially agree)")
        for s, v in agree_share.items():
            say(f"  {v:>4.0f}%  {s}")
        import textwrap
        fig, ax = plt.subplots(figsize=(9.0, 1.2 + 0.62 * len(counts)))
        labels = ["\n".join(textwrap.wrap(s, 48)) for s in counts.index]
        left = [0] * len(counts)
        for cat, col in zip(AGREE, AGREE_COLOURS):
            vals = (counts[cat] / n * 100).tolist()
            ax.barh(labels, vals, left=left, color=col, edgecolor="white", linewidth=2, label=cat)
            left = [l + v for l, v in zip(left, vals)]
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Participants (%)")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_US_feature_statements.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    pref = [c for c in df.columns if c.startswith("Which view did you prefer")]
    if pref:
        say("\nPreferred view: " + ", ".join(f"{k} {v}" for k, v in df[pref[0]].value_counts().items()))

    # ---------------------------------------------------------- open answers
    say("\nOpen answers (copy themes and short quotes into the report):")
    for q in ["What did you find most useful?", "Was anything confusing, slow or difficult?",
              "What one change would make TrustLens better?"]:
        col = [c for c in df.columns if c.startswith(q)]
        if col:
            say(f"\n{q}")
            for a in df[col[0]].dropna():
                if str(a).strip():
                    say(f"  - {str(a).strip()}")

    with open(os.path.join(OUT, "user_study_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\nSaved results in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "user_study_responses.csv")
