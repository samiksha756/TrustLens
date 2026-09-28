"""
TrustLens — Emotion / Manipulation Model (Model 2, text data-space, different task)
------------------------------------------------------------------------------------
Model 1 (classifier.py) answers "is this phishing?".
Model 2 answers a *different* question in the same data-space: "is this message
trying to manipulate the reader emotionally?".

Social-engineering attacks work by inducing fear, urgency, or excitement. A
phishing classifier can miss a novel scam whose *wording* is unusual but whose
*emotional pressure* is textbook. Adding an emotion model gives the orchestrator
an independent, complementary signal, which is exactly the point of the CM3020
"orchestrate multiple models" template: the models must contribute *different*
evidence, not duplicate each other.

Pre-trained model:  j-hartmann/emotion-english-distilroberta-base
    - 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
    - We map fear / anger / surprise / disgust -> "pressure" emotions used by scams.

Design notes
------------
* The model is loaded LAZILY on first use so importing this module is cheap and
  the app still starts if the model is not present.
* If transformers/torch or the weights are unavailable, `analyse_emotion`
  returns a clearly-flagged fallback result (available=False) instead of raising,
  so the orchestration pipeline degrades gracefully rather than crashing.
"""

from functools import lru_cache

# Emotions that scam / social-engineering messages typically try to provoke.
PRESSURE_EMOTIONS = {"fear", "anger", "surprise", "disgust"}

_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def _load_pipeline():
    """Lazily build a HF text-classification pipeline. Cached after first call."""
    from transformers import pipeline  # imported here so module import stays light
    clf = pipeline(
        "text-classification",
        model=_EMOTION_MODEL,
        top_k=None,          # return the full distribution over emotions
        truncation=True,
        max_length=256,
    )
    return clf


def analyse_emotion(text: str) -> dict:
    """
    Return the dominant emotion and an aggregate 'pressure score' in [0, 1]
    describing how strongly the message leans on fear/anger/surprise/disgust.

    On any failure (missing deps, offline, empty input) returns a fallback dict
    with available=False and a neutral, non-penalising pressure_score of 0.0.
    """
    if not text or not text.strip():
        return {"available": True, "dominant": "neutral",
                "pressure_score": 0.0, "distribution": {}}

    try:
        clf = _load_pipeline()
        scores = clf(text)[0]  # list of {label, score}
        dist = {d["label"].lower(): float(d["score"]) for d in scores}
        pressure = sum(v for k, v in dist.items() if k in PRESSURE_EMOTIONS)
        dominant = max(dist, key=dist.get)
        return {
            "available": True,
            "dominant": dominant,
            "pressure_score": round(min(pressure, 1.0), 4),
            "distribution": {k: round(v, 4) for k, v in dist.items()},
        }
    except Exception as exc:  # noqa: BLE001 — deliberately broad for graceful degrade
        return {
            "available": False,
            "dominant": "unknown",
            "pressure_score": 0.0,
            "distribution": {},
            "error": f"emotion model unavailable: {type(exc).__name__}",
        }


if __name__ == "__main__":
    for s in [
        "URGENT: your account will be permanently deleted in 2 hours unless you verify now!",
        "Hi mum, I'll be home for dinner at 7. See you then.",
        "Congratulations!! You are today's lucky winner of a brand new iPhone!",
    ]:
        print(f"\n{s[:70]}...")
        print(analyse_emotion(s))
