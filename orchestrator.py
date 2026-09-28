"""
TrustLens — Orchestrator
------------------------
The single place where the multi-model pipeline is wired together. Every input
modality (typed text, screenshot image, voice note) is reduced to text and then
run through the SAME shared analysis pipeline. This is the heart of the CM3020
"orchestrate AI models to achieve a goal" template: independent pre-trained
models across different data spaces feed one coordinated decision.

Flow
----
    image  --(OCR, Model 3)-->  text  \
    audio  --(Whisper, Model 4)-> text --> [ classifier (M1) + emotion (M2)
    text   ----------------------------->     + indicators + link forensics ]
                                                     |
                                                     v
                                          aggregate_risk  ->  explanation

Two aggregation paths
---------------------
* Default (calibrate=False): the ORIGINAL scoring is used, so every previously
  reported number is reproduced exactly. Results are additionally *enriched*
  with advanced link-analysis and benign-context evidence in the explanation
  (this only adds fields; it does not change the score or level).
* Calibrated (calibrate=True): an opt-in advanced path that temperature-scales
  the saturated classifier and folds link risk + benign context into the score
  (see aggregator.aggregate_risk_calibrated). Used by the CLI `--calibrated`
  flag and the model-specific evaluation.

If an optional model (emotion / OCR / Whisper) is unavailable, its signal is
skipped and the pipeline still returns a valid result — graceful degradation.
"""

from classifier import classify_text
from indicators import extract_indicators
from aggregator import aggregate_risk, aggregate_risk_calibrated, generate_explanation
from emotion import analyse_emotion
from link_analysis import analyse_links
from context_signals import detect_benign_context


def analyse_text(text: str, source: str = "text", calibrate: bool = False,
                 explain: bool = False) -> dict:
    """
    Run the full orchestrated pipeline on a piece of text.

    `source`    records where the text came from (text / image / audio).
    `calibrate` selects the aggregation path. False (default) reproduces the
                original behaviour and numbers; True uses the calibrated,
                link-aware, benign-context-aware fusion.
    `explain`   when True, adds opt-in occlusion-based token attribution
                ("why these words?") under the "attribution" key. Default False,
                so the standard path and all reported numbers are unchanged.
                Note: attribution runs one classifier pass per word, so it is
                slower and is only requested explicitly.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "No text to analyse."}

    # Model 1 — phishing classifier (text space)
    cls = classify_text(text)

    # Model 2 — emotion / manipulation (text space, different task)
    emo = analyse_emotion(text)
    emotion_pressure = emo["pressure_score"] if emo.get("available") else None

    # Rule-based structured evidence + advanced link forensics + benign context
    inds = extract_indicators(text)
    links = analyse_links(text)
    ctx = detect_benign_context(text)

    if calibrate:
        assessment = aggregate_risk_calibrated(
            cls["phishing_prob"], inds,
            emotion_pressure=emotion_pressure,
            link_risk=links["link_risk"],
            benign_context=ctx["benign_context"],
        )
    else:
        # ORIGINAL path — unchanged score/level (preserves reported numbers).
        assessment = aggregate_risk(cls["phishing_prob"], inds,
                                    emotion_pressure=emotion_pressure)

    explanation = generate_explanation(assessment, inds, text,
                                        emotion=emo, link_findings=links, context=ctx)

    result = {
        "source": source,
        "input_preview": text[:280],
        "assessment": assessment,
        "explanation": explanation,
        "emotion": emo,
        "links": links,
        "benign_context": ctx,
        "classifier_available": cls.get("available", True),
    }

    # Opt-in word-level explanation (occlusion attribution). Additive only.
    if explain:
        from attribution import attribute_tokens
        result["attribution"] = attribute_tokens(text)

    return result


def analyse_image(image_bytes: bytes, calibrate: bool = False,
                  explain: bool = False) -> dict:
    """Screenshot route: OCR (Model 3) -> shared text pipeline."""
    from ocr import extract_text_from_image
    ocr = extract_text_from_image(image_bytes)
    if not ocr.get("available"):
        return {"error": ocr.get("note", "OCR unavailable."), "ocr": ocr}
    if not ocr.get("text"):
        return {"error": "No readable text found in the image.", "ocr": ocr}
    result = analyse_text(ocr["text"], source="image", calibrate=calibrate,
                          explain=explain)
    result["ocr"] = ocr
    return result


def analyse_audio(audio_bytes: bytes, suffix: str = ".wav", calibrate: bool = False,
                  explain: bool = False) -> dict:
    """Voice-note route: Whisper STT (Model 4) -> shared text pipeline."""
    from speech import transcribe_audio
    stt = transcribe_audio(audio_bytes, suffix=suffix)
    if not stt.get("available"):
        return {"error": stt.get("note", "Speech-to-text unavailable."), "stt": stt}
    if not stt.get("text"):
        return {"error": "No speech detected in the audio.", "stt": stt}
    result = analyse_text(stt["text"], source="audio", calibrate=calibrate,
                          explain=explain)
    result["stt"] = stt
    return result
