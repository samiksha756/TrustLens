"""
TrustLens — Token Attribution Module (occlusion-based explainability)
----------------------------------------------------------------------
"Why these words?" — a transparent, model-agnostic explanation of WHICH words
in a message pushed the phishing classifier toward its decision.

How it works (occlusion / leave-one-out attribution)
----------------------------------------------------
1. Run the classifier on the full message to get a baseline phishing probability.
2. For each word, remove that one word and re-run the classifier.
3. If removing a word makes the message look *less* like phishing, that word was
   pushing the score UP (toward phishing). If removing it makes the score look
   *more* like phishing, it was pushing the score DOWN (toward safe).
The size of that change is the word's contribution.

This is the same intuition as SHAP / LIME feature attribution, implemented with
pure Python over the EXISTING classifier — so it needs no extra heavy
dependency, it cannot contradict the model (it queries the real model), and it
fits TrustLens's "explainability by construction" principle: the evidence is
legible, not a black box.

Design notes
------------
* Opt-in and additive. Nothing here runs on the default analysis path, so all
  previously reported numbers are unchanged. It is invoked only when a caller
  asks for it (orchestrator `explain=True`, or the web UI's "explain words"
  toggle).
* Graceful degradation. If the classifier is unavailable, this returns
  `available=False` and an empty list instead of raising.
* Bounded cost. Occlusion runs one model pass per word, so the number of words
  scored is capped (MAX_TOKENS) to keep latency predictable on modest hardware.
"""

import re

from classifier import classify_text

_WORD_RE = re.compile(r"\S+")

# Occlusion runs one classifier pass per word, so cap the number of words scored
# to keep latency predictable on CPU-only hardware. Longer inputs are truncated
# for the attribution pass only (the classifier itself still sees the full text).
MAX_TOKENS = 60

# Words whose removal changes the probability by less than this are treated as
# non-influential and dropped, so the UI shows only words that actually mattered.
MIN_ABS_DELTA = 0.01

# Very common function words carry little scam signal; kept out of the ranked
# output so the highlighted words are the meaningful ones. (Attribution is still
# computed over the full message — this only tidies what is shown.)
_STOP = {
    "the", "a", "an", "to", "of", "and", "or", "is", "are", "was", "were",
    "be", "been", "in", "on", "at", "it", "this", "that", "your", "you",
    "for", "with", "as", "by", "has", "have", "will", "i", "we", "they",
}


def _words(text: str):
    return _WORD_RE.findall(text or "")


def attribute_tokens(text: str, top_k: int = 8) -> dict:
    """
    Attribute the phishing decision to individual words by occlusion.

    Returns
    -------
    dict with:
        available  : bool  — False if the classifier could not be queried
        base_prob  : float — phishing probability of the full message
        base_label : "phishing" | "benign"
        tokens     : list of {token, delta, weight, direction}, ranked by the
                     size of their effect (most influential first). `weight` is
                     the effect normalised to [0,1] for display; `direction` is
                     "phishing" (word raised the score) or "safe" (word lowered it).
        note       : short human-readable status
    """
    text = (text or "").strip()
    if not text:
        return {"available": True, "base_prob": 0.0, "base_label": "benign",
                "tokens": [], "note": "No text to explain."}

    base = classify_text(text)
    if not base.get("available", True):
        return {"available": False, "base_prob": 0.0, "base_label": "benign",
                "tokens": [],
                "note": base.get("note", "Classifier unavailable — word "
                                          "explanation skipped.")}

    base_prob = float(base["phishing_prob"])
    words = _words(text)
    scored_words = words[:MAX_TOKENS]

    raw = []
    for i, w in enumerate(scored_words):
        occluded = " ".join(scored_words[:i] + scored_words[i + 1:])
        if not occluded.strip():
            continue
        occ_prob = float(classify_text(occluded)["phishing_prob"])
        delta = base_prob - occ_prob  # >0 => word pushed toward phishing
        raw.append({"token": w, "delta": round(delta, 4)})

    # Rank by magnitude of effect; drop tiny effects and bare stop-words.
    influential = [
        r for r in raw
        if abs(r["delta"]) >= MIN_ABS_DELTA
        and re.sub(r"[^\w]", "", r["token"]).lower() not in _STOP
    ]
    influential.sort(key=lambda r: abs(r["delta"]), reverse=True)
    influential = influential[:top_k]

    max_abs = max((abs(r["delta"]) for r in influential), default=0.0)
    tokens = []
    for r in influential:
        weight = round(abs(r["delta"]) / max_abs, 3) if max_abs > 0 else 0.0
        tokens.append({
            "token": r["token"],
            "delta": r["delta"],
            "weight": weight,
            "direction": "phishing" if r["delta"] > 0 else "safe",
        })

    if tokens:
        note = (f"{len(tokens)} influential word(s) identified by leave-one-out "
                f"occlusion over the phishing classifier.")
    else:
        note = ("No single word changed the classifier's decision appreciably; "
                "the judgement rests on the message as a whole.")

    return {
        "available": True,
        "base_prob": round(base_prob, 4),
        "base_label": base.get("label", "benign"),
        "tokens": tokens,
        "note": note,
    }


if __name__ == "__main__":
    for s in [
        "Your PayPal account has been locked. Verify your password immediately "
        "at http://192.168.1.45/login or your account will be suspended.",
        "Hi mum, just letting you know I'll be home for dinner tonight.",
    ]:
        r = attribute_tokens(s)
        print(f"\nInput: {s[:70]}...")
        print(f"  available={r['available']}  base_prob={r['base_prob']}  ({r['note']})")
        for t in r["tokens"]:
            arrow = "↑ phishing" if t["direction"] == "phishing" else "↓ safe"
            print(f"   {t['token']:>18}  weight={t['weight']:.2f}  {arrow}  (Δ={t['delta']})")
