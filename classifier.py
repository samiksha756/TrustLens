"""
TrustLens — Text Classifier Module (Model 1, TEXT data-space)
--------------------------------------------------------------
Wraps a pre-trained phishing-detection transformer from Hugging Face and
returns a phishing probability for any input string. This is the core
detector around which the rest of the orchestration is built.

Pre-trained model:  ealvaradob/bert-finetuned-phishing  (BERT-family)

Upgrade note (v2)
-----------------
The model is now loaded LAZILY on first use (cached), instead of at import
time. This mirrors the emotion / OCR / speech modules and has three benefits:
  * importing the module is cheap, so tooling, tests and the CLI start fast;
  * if the weights or `transformers`/`torch` are unavailable, the module
    degrades gracefully (available=False) instead of crashing the whole app;
  * the classification RESULT for a given input is unchanged from v1 — the
    same tokenisation (truncation, max_length=256) and soft-max read-off are
    used, so all previously reported numbers are reproduced exactly.
"""

from functools import lru_cache

MODEL_NAME = "ealvaradob/bert-finetuned-phishing"


@lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once and cache. Raises if deps/weights missing."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch  # noqa: F401  (imported to fail fast if torch is absent)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


def classify_text(text: str) -> dict:
    """
    Run the phishing classifier on the input text.

    Returns a dict with:
        phishing_prob : float in [0,1]
        label         : "phishing" | "benign"
        available     : bool  (False only if the model could not be loaded)

    On empty input returns a benign, zero-probability result. On a missing
    model it returns a clearly-flagged fallback (available=False, prob 0.0)
    so the orchestration pipeline keeps working rather than crashing.
    """
    if not text or not text.strip():
        return {"phishing_prob": 0.0, "label": "benign", "available": True}

    try:
        import torch
        tokenizer, model = _load()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        phishing_prob = float(probs[0][1].item())
        return {
            "phishing_prob": round(phishing_prob, 4),
            "label": "phishing" if phishing_prob >= 0.5 else "benign",
            "available": True,
        }
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        return {
            "phishing_prob": 0.0,
            "label": "benign",
            "available": False,
            "note": (f"phishing model unavailable ({type(exc).__name__}). "
                     f"Run `pip install -r requirements.txt` to enable it."),
        }


if __name__ == "__main__":
    for s in [
        "Your account has been locked. Verify your password within 24 hours: http://192.168.1.45/login",
        "Hi mum, just letting you know I'll be home for dinner tonight.",
        "Congratulations! You have won a $1000 Amazon gift card. Click here to claim now!",
    ]:
        print(f"\nInput: {s[:80]}...")
        print("Result:", classify_text(s))
