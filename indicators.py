"""
TrustLens — Indicator Extractor Module
--------------------------------------
Rule-based extractor for phishing/scam indicators.
Uses curated regex patterns derived from common scam-message taxonomies
(urgency, credential requests, suspicious URLs, payment pressure, etc.)

This module supports the pre-trained classifier by catching structural
cues that pure language-model classifiers can miss — e.g. raw-IP URLs,
brand/URL mismatches, and explicit credential-request phrasing.
"""

import re
from urllib.parse import urlparse


INDICATOR_RULES = {
    "urgency": (
        r"\b(urgent|immediately|within\s+\d+\s+(hours?|minutes?|days?)|act\s+now|"
        r"final\s+notice|last\s+chance|expires?\s+(soon|today))\b",
        "The message uses urgency to pressure quick action without verification."
    ),
    "credential_request": (
        r"(verify\s+your\s+(account|password|identity)|"
        r"confirm\s+your\s+(login|credentials|password)|"
        r"re-?enter\s+your\s+password|update\s+your\s+payment)",
        "The message asks the user to enter sensitive credentials, a common phishing tactic."
    ),
    "suspicious_url_ip": (
        r"https?://\d{1,3}(\.\d{1,3}){3}",
        "The message contains a link pointing to a raw IP address instead of a domain name."
    ),
    "payment_pressure": (
        r"(payment\s+failed|unpaid\s+invoice|wire\s+transfer|"
        r"outstanding\s+balance|overdue\s+payment|tax\s+refund)",
        "The message references payment problems or refunds, a frequent scam hook."
    ),
    "prize_lure": (
        r"(you\s+have\s+won|congratulations.*(prize|gift\s+card|reward)|"
        r"claim\s+your\s+(prize|reward)|free\s+(iphone|gift))",
        "The message offers a prize or reward, a known lure tactic."
    ),
    "impersonation_brand": (
        r"\b(paypal|amazon|netflix|microsoft|apple|google|hsbc|barclays|dbs|posb|"
        r"singpass|iras|hmrc|fedex|dhl|ups)\b",
        "The message references a well-known brand or institution; verify directly with the official site."
    ),
    "account_threat": (
        r"(account\s+(locked|suspended|disabled|compromised)|"
        r"suspicious\s+activity\s+detected|unauthorized\s+access)",
        "The message threatens account suspension to provoke a panic response."
    ),
}


def _extract_urls(text: str) -> list:
    url_pattern = r"https?://[^\s\)\]\>]+"
    return re.findall(url_pattern, text, flags=re.IGNORECASE)


def _check_url_mismatch(text: str) -> bool:
    brand_pattern = r"\b(paypal|amazon|netflix|microsoft|apple|google|hsbc|barclays|dbs)\b"
    brands_in_text = re.findall(brand_pattern, text, flags=re.IGNORECASE)
    urls = _extract_urls(text)
    if not brands_in_text or not urls:
        return False
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
            for brand in brands_in_text:
                if brand.lower() not in domain:
                    return True
        except Exception:
            continue
    return False


def extract_indicators(text: str) -> list:
    if not text:
        return []

    hits = []
    for name, (pattern, explanation) in INDICATOR_RULES.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append({"name": name, "explanation": explanation})

    if _check_url_mismatch(text):
        hits.append({
            "name": "brand_url_mismatch",
            "explanation": "The message mentions a well-known brand but links to an unrelated domain."
        })

    return hits


if __name__ == "__main__":
    samples = [
        "Your PayPal account has been locked. Verify immediately at http://192.168.1.45/login",
        "Hi mum, just letting you know I'll be home for dinner tonight.",
        "Congratulations! You have won a $1000 Amazon gift card. Click here to claim now!",
        "Your tax refund of $450 is pending. Confirm your details urgently.",
    ]
    for s in samples:
        print(f"\nInput: {s}")
        for ind in extract_indicators(s):
            print(f"  - {ind['name']}: {ind['explanation']}")
