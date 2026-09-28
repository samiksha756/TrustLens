"""
TrustLens — Risk Aggregator & Explanation Module
-------------------------------------------------
Combines the evidence produced by the orchestrated models into ONE risk score,
a risk level, and a plain-language explanation for a non-specialist user.

Signals fused here:
    1. classifier probability   (Model 1, phishing DistilBERT)   — "is this phishing?"
    2. rule-based indicators     (indicators.py)                  — structured evidence
    3. emotional-pressure score  (Model 2, emotion model)        — social-engineering cue

Backward compatibility
----------------------
`aggregate_risk(phishing_prob, indicators)` with NO emotion argument reproduces
the exact behaviour and numbers of the preliminary-report prototype
(classifier 0.65 / indicators 0.35). This keeps the original text-only
evaluation reproducible. When an emotion pressure score IS supplied, the
weighting is redistributed to 0.55 / 0.30 / 0.15 to fold in the third signal.

The explanation is deliberately TEMPLATED, not free-form LLM text, so the
recommended safe-action is deterministic and cannot hallucinate. This is a
safety-driven design choice, discussed in the report.
"""

INDICATOR_CAP = 4

THRESHOLDS = [
    (0.25, "Safe"),
    (0.50, "Caution"),
    (0.75, "Suspicious"),
    (1.01, "High Risk"),
]


def aggregate_risk(phishing_prob: float, indicators: list, emotion_pressure=None) -> dict:
    """
    Fuse the model signals into a single risk score and level.

    emotion_pressure : float in [0,1] or None.
        None  -> text-only mode, original 0.65/0.35 weighting (reproduces prelim results).
        float -> tri-signal mode, 0.55/0.30/0.15 weighting.
    """
    norm_indicators = min(len(indicators) / INDICATOR_CAP, 1.0)

    if emotion_pressure is None:
        score = 0.65 * phishing_prob + 0.35 * norm_indicators
    else:
        ep = max(0.0, min(float(emotion_pressure), 1.0))
        score = 0.55 * phishing_prob + 0.30 * norm_indicators + 0.15 * ep

    level = "Safe"
    for threshold, name in THRESHOLDS:
        if score < threshold:
            level = name
            break

    out = {
        "score": round(score, 4),
        "level": level,
        "classifier_prob": round(phishing_prob, 4),
        "indicator_count": len(indicators),
    }
    if emotion_pressure is not None:
        out["emotion_pressure"] = round(float(emotion_pressure), 4)
    return out


def aggregate_risk_calibrated(phishing_prob: float, indicators: list,
                              emotion_pressure=None, link_risk: float = 0.0,
                              benign_context: float = 0.0,
                              temperature: float = None) -> dict:
    """
    OPT-IN advanced fusion (v2). This does NOT replace `aggregate_risk`; the
    default pipeline still uses the original, so all previously reported numbers
    are preserved. This path adds three things the original lacks:

      1. Confidence calibration — the saturated classifier probability is first
         temperature-scaled (calibration.soften_probability) so it no longer
         swamps every other signal.
      2. Link-analysis risk — the forensic URL risk (link_analysis) becomes a
         first-class signal.
      3. Benign-context dampening — legitimate transactional / marketing cues
         (context_signals) reduce the score, targeting the false positives the
         evaluation identified.

    A safety floor prevents benign context from ever hiding a strongly-indicated
    scam: with >=3 indicators or a high link risk, the level cannot drop below
    "Suspicious".
    """
    from calibration import soften_probability, DEFAULT_TEMPERATURE
    t = DEFAULT_TEMPERATURE if temperature is None else temperature
    softened = soften_probability(phishing_prob, t)
    norm_indicators = min(len(indicators) / INDICATOR_CAP, 1.0)
    link_risk = max(0.0, min(float(link_risk), 1.0))
    benign_context = max(0.0, min(float(benign_context), 1.0))

    if emotion_pressure is None:
        score = 0.55 * softened + 0.28 * norm_indicators + 0.17 * link_risk
    else:
        ep = max(0.0, min(float(emotion_pressure), 1.0))
        score = (0.48 * softened + 0.24 * norm_indicators
                 + 0.12 * ep + 0.16 * link_risk)

    score *= (1.0 - 0.35 * benign_context)   # benign-context dampening
    score = max(0.0, min(score, 1.0))

    level = "Safe"
    for threshold, name in THRESHOLDS:
        if score < threshold:
            level = name
            break

    # Safety floor: never let benign context mask a strongly-indicated scam.
    if (len(indicators) >= 3 or link_risk >= 0.85) and level in ("Safe", "Caution"):
        level = "Suspicious"

    out = {
        "score": round(score, 4),
        "level": level,
        "classifier_prob": round(softened, 4),
        "raw_classifier_prob": round(float(phishing_prob), 4),
        "indicator_count": len(indicators),
        "link_risk": round(link_risk, 4),
        "benign_context": round(benign_context, 4),
        "calibrated": True,
        "temperature": t,
    }
    if emotion_pressure is not None:
        out["emotion_pressure"] = round(float(emotion_pressure), 4)
    return out


SAFE_ACTIONS = {
    "Safe": (
        "This message appears safe based on the current analysis. "
        "Continue to remain cautious and verify any unusual request through official channels."
    ),
    "Caution": (
        "Treat this message with caution. Do not click any links or share any personal information "
        "until you have verified the sender through a trusted channel, such as the official website "
        "or a known phone number."
    ),
    "Suspicious": (
        "Do not click any links, download attachments, or reply with personal information. "
        "Verify the message by contacting the supposed sender directly using contact details "
        "you already trust, not the ones provided in the message."
    ),
    "High Risk": (
        "This message shows strong signs of being a phishing or scam attempt. "
        "Do not interact with it. Do not click any links, do not reply, and do not call any numbers it contains. "
        "If you have already shared information or clicked a link, change your passwords immediately and "
        "contact your bank or service provider through their official website."
    ),
}


def generate_explanation(assessment: dict, indicators: list, input_preview: str,
                         emotion: dict = None, link_findings: dict = None,
                         context: dict = None) -> dict:
    level = assessment["level"]

    if level == "Safe":
        summary = "No strong indicators of phishing or scam content were detected in this message."
    elif level == "Caution":
        summary = "Some suspicious patterns were detected. Treat this message with caution."
    elif level == "Suspicious":
        summary = "This message contains several patterns commonly used in phishing or scam attempts."
    else:
        summary = "This message strongly resembles a phishing or scam attempt. Do not engage with it."

    indicator_list = [
        {"name": ind["name"].replace("_", " ").title(), "reason": ind["explanation"]}
        for ind in indicators
    ]

    # Optional emotional-manipulation note, only when the emotion model contributed.
    emotion_note = None
    if emotion and emotion.get("available") and emotion.get("dominant") not in (None, "neutral", "joy"):
        if emotion.get("pressure_score", 0) >= 0.5:
            emotion_note = (
                f"The message reads as emotionally charged (dominant tone: "
                f"'{emotion['dominant']}'). Scam messages often use fear, urgency or "
                f"excitement to push you into acting before you think."
            )

    parts = [
        f"The phishing classifier rated this message at a probability of "
        f"{assessment['classifier_prob']:.2f}."
    ]
    if indicators:
        parts.append(
            f"{len(indicators)} rule-based warning sign(s) were matched by the "
            f"indicator extractor."
        )
    else:
        parts.append("No rule-based warning signs were matched.")
    if "emotion_pressure" in assessment:
        parts.append(
            f"The emotion model scored the emotional-pressure of the message at "
            f"{assessment['emotion_pressure']:.2f}."
        )
    parts.append(
        f"Combining these signals gives a risk score of {assessment['score']:.2f}, "
        f"placing the message in the '{level}' category."
    )
    reasoning = " ".join(parts)

    out = {
        "summary": summary,
        "indicators": indicator_list,
        "emotion_note": emotion_note,
        "reasoning": reasoning,
        "safe_action": SAFE_ACTIONS[level],
        "disclaimer": (
            "TrustLens provides risk guidance based on automated analysis. "
            "It does not guarantee that a message is safe or unsafe. "
            "Always verify suspicious content through trusted channels."
        ),
    }

    # Optional richer evidence blocks (only present when the caller supplies them,
    # so existing consumers of this function are unaffected).
    if link_findings and link_findings.get("findings"):
        out["link_analysis"] = {
            "note": link_findings.get("note", ""),
            "link_risk": link_findings.get("link_risk", 0.0),
            "findings": [{"name": f["name"].replace("_", " ").title(),
                          "reason": f["reason"]}
                         for f in link_findings["findings"]],
        }
    if context and context.get("signals"):
        out["benign_context"] = {
            "note": context.get("note", ""),
            "score": context.get("benign_context", 0.0),
            "signals": [{"name": s["name"].replace("_", " ").title(),
                         "reason": s["reason"]}
                        for s in context["signals"]],
        }
    return out


if __name__ == "__main__":
    from indicators import extract_indicators
    sample = "Your PayPal account has been locked. Verify immediately at http://192.168.1.45/login"
    inds = extract_indicators(sample)
    # tri-signal demo
    a = aggregate_risk(0.92, inds, emotion_pressure=0.8)
    print(a)
    print(generate_explanation(a, inds, sample,
          emotion={"available": True, "dominant": "fear", "pressure_score": 0.8}))
