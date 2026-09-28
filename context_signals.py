"""
TrustLens — Benign-Context Signals Module
------------------------------------------
The evaluation showed the system's only errors were two *benign* messages (a
genuine Amazon shipping notice and a marketing message) that the saturated
classifier pushed to "phishing". The report named the fix as future work:
add benign-context signals so ordinary transactional / marketing language is
not mistaken for a scam.

This module implements that. It looks for positive, *legitimate-looking* cues
and returns a `benign_context` score in [0,1] with transparent reasons. It is a
counter-weight to the scam indicators, used only by the OPT-IN calibrated
aggregation path — the default pipeline and all previously reported numbers are
unchanged.

Cues detected
-------------
  * order/tracking references  — realistic order IDs, tracking numbers
  * unsubscribe / marketing footer — legitimate bulk mail is required to carry these
  * no-action-required phrasing — scams push action; genuine notices often don't
  * known-good domain           — link resolves to a real brand's own domain
  * appointment / receipt language — routine transactional wording

Important: benign context can *lower* concern but never forces "Safe" on its
own — a message with strong scam indicators stays risky regardless. This keeps
the tool safe-by-default.
"""

import re
from link_analysis import extract_urls, _registrable_domain, KNOWN_BRANDS

# A brand -> its legitimate registrable domain(s). Small, curated, explainable.
KNOWN_GOOD_DOMAINS = {
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.sg"},
    "paypal": {"paypal.com"},
    "microsoft": {"microsoft.com", "office.com", "live.com"},
    "apple": {"apple.com", "icloud.com"},
    "google": {"google.com", "youtube.com"},
    "netflix": {"netflix.com"},
    "dhl": {"dhl.com"},
    "fedex": {"fedex.com"},
}

_BENIGN_RULES = {
    "order_reference": (
        r"\border(\s*(number|no\.?|#|id))?\s*[:#]?\s*[a-z0-9]{3,}-?[a-z0-9]{3,}\b",
        0.30,
        "The message quotes a concrete order/reference number, typical of a genuine "
        "transactional notice.",
    ),
    "tracking_number": (
        r"\btracking\s*(number|no\.?|#)?\s*[:#]?\s*[a-z0-9]{8,}\b",
        0.25,
        "The message includes a tracking number, typical of a real delivery update.",
    ),
    "unsubscribe_footer": (
        r"\b(unsubscribe|manage\s+your\s+preferences|you\s+are\s+receiving\s+this|"
        r"update\s+your\s+email\s+preferences)\b",
        0.30,
        "The message carries an unsubscribe / preferences footer, which legitimate "
        "bulk email is legally required to include.",
    ),
    "no_action_required": (
        r"\b(no\s+action\s+(is\s+)?required|for\s+your\s+records|this\s+is\s+a\s+"
        r"confirmation|receipt\s+for\s+your)\b",
        0.25,
        "The message says no action is required — scams almost always demand action.",
    ),
    "appointment_receipt": (
        r"\b(your\s+appointment|has\s+shipped|out\s+for\s+delivery|order\s+confirmed|"
        r"payment\s+received|booking\s+confirmed)\b",
        0.20,
        "Routine transactional wording (shipping / booking / receipt) consistent with "
        "a genuine service message.",
    ),
}


def _known_good_link(text: str) -> bool:
    urls = extract_urls(text)
    if not urls:
        return False
    for url in urls:
        host = url.split("://")[-1].split("/")[0].lower()
        reg, _ = _registrable_domain(host)
        for brand in KNOWN_BRANDS:
            if brand in text.lower() and reg in KNOWN_GOOD_DOMAINS.get(brand, set()):
                return True
    return False


def detect_benign_context(text: str) -> dict:
    """
    Return positive benign-context evidence.

    Returns
    -------
    dict with:
        benign_context : float in [0,1]  (higher = more legitimate-looking)
        signals        : list of {name, reason}
        note           : short human-readable status
    """
    if not text or not text.strip():
        return {"benign_context": 0.0, "signals": [], "note": "No text."}

    signals, score = [], 0.0
    for name, (pattern, weight, reason) in _BENIGN_RULES.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            signals.append({"name": name, "reason": reason})
            score += weight

    if _known_good_link(text):
        signals.append({
            "name": "known_good_domain",
            "reason": "A link in the message resolves to the brand's own official "
                      "domain, not a look-alike.",
        })
        score += 0.40

    score = round(min(score, 1.0), 3)
    note = (f"{len(signals)} legitimate-context signal(s) found."
            if signals else "No benign-context signals found.")
    return {"benign_context": score, "signals": signals, "note": note}


if __name__ == "__main__":
    tests = [
        "Your Amazon order #404-1234567-8901234 has shipped and will arrive Friday. "
        "No action is required. Track at https://www.amazon.co.uk/orders",
        "New season sale! Shop the new collection now. Unsubscribe here.",
        "Your PayPal account has been locked. Verify at http://192.168.1.45/login",
        "Hi mum, I'll be home for dinner tonight.",
    ]
    for t in tests:
        r = detect_benign_context(t)
        print(f"\n{t[:70]}...")
        print(f"  benign_context={r['benign_context']}  ({r['note']})")
        for s in r["signals"]:
            print(f"   + {s['name']}: {s['reason']}")
