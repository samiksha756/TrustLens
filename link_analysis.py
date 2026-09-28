"""
TrustLens — Link Analysis Module (advanced URL / domain forensics)
-------------------------------------------------------------------
The rule-based `indicators.py` layer catches a couple of coarse URL cues
(a raw-IP link, a crude brand/URL mismatch). This module is a substantial
step up: it performs structured forensic analysis of every URL in a message
and returns human-readable findings with per-finding severity.

It is a *transparent, explainable* component — every finding carries a plain
reason the user can understand — which fits TrustLens's design principle that
evidence must be legible, not a black box.

Checks performed per URL
------------------------
  * ip_literal          — host is a raw IP address (no domain)
  * userinfo_obfuscation — an "@" before the host hides the true destination
  * punycode            — "xn--" host, a classic homograph / look-alike trick
  * url_shortener       — link hidden behind a shortener (bit.ly, tinyurl, ...)
  * suspicious_tld      — cheap/abused TLD (.tk, .xyz, .top, .zip, ...)
  * deceptive_subdomain — a brand appears as a sub-domain of another registrable
                          domain, e.g. paypal.com.secure-login.tk
  * brand_not_registrable — a brand name appears in the host/path but is NOT the
                          registrable domain (paypal.evil.com, x.com/paypal)
  * excessive_subdomains — unusually deep host, a common obfuscation
  * non_standard_port   — explicit non-web port
  * hex_or_encoded      — %-encoding or hex in the host

Design: pure standard-library (regex + urllib), so it runs anywhere with no
model downloads, and it is fully unit-tested.
"""

import re
from urllib.parse import urlparse

# --- reference data (small, curated, explainable) --------------------------------

KNOWN_BRANDS = {
    "paypal", "amazon", "netflix", "microsoft", "apple", "google", "hsbc",
    "barclays", "dbs", "posb", "ocbc", "uob", "singpass", "iras", "hmrc",
    "fedex", "dhl", "ups", "facebook", "instagram", "whatsapp", "linkedin",
    "outlook", "office365", "icloud", "gov",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "lnkd.in",
}

# TLDs that are disproportionately abused in phishing / low-cost & anonymous.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "click",
    "country", "kim", "work", "link", "support", "review", "info", "live",
    "rest", "fit", "loan", "mom", "date",
}

# Multi-label public suffixes so we compute the *registrable* domain correctly
# (e.g. hsbc.co.uk -> registrable "hsbc.co.uk", not "co.uk").
MULTI_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "co.nz", "org.nz", "com.au",
    "net.au", "org.au", "com.sg", "gov.sg", "edu.sg", "com.my", "co.in",
    "co.jp", "com.br", "co.za",
}

_URL_RE = re.compile(r"""(?xi)
    (?<![@\w.-])(
        (?:https?://|www\.)          # scheme or www.
        [^\s<>"')\]]+                # the rest of the URL
      |
        # Bare domain typed without a scheme, e.g. "bit.ly/xyz" or
        # "hmrc-refund-claim.com" -- common in SMS scams. Requires a
        # 2+ letter TLD; the look-behind skips e-mail addresses.
        (?:[a-z0-9-]+\.)+[a-z]{2,}
        (?:/[^\s<>"')\]]*)?
    )
""")


def _registrable_domain(host: str):
    """Return (registrable_domain, subdomain_labels) using a small suffix list."""
    host = host.lower().strip(".")
    labels = host.split(".")
    if len(labels) < 2:
        return host, []
    last2 = ".".join(labels[-2:])
    last3 = ".".join(labels[-3:]) if len(labels) >= 3 else None
    if last3 and last3 in MULTI_TLDS:  # e.g. foo.hsbc.co.uk
        reg = ".".join(labels[-3:])
        sub = labels[:-3]
    elif last2 in MULTI_TLDS and len(labels) >= 3:
        reg = ".".join(labels[-3:])
        sub = labels[:-3]
    else:
        reg = last2
        sub = labels[:-2]
    return reg, sub


def _looks_like_ip(host: str) -> bool:
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        return True
    if ":" in host and re.search(r"[0-9a-f]", host, re.I):  # crude IPv6
        return re.fullmatch(r"[0-9a-f:]+", host, re.I) is not None
    return False


def _normalise(raw: str) -> str:
    return raw if "://" in raw else "http://" + raw


def analyse_url(raw_url: str) -> dict:
    """Forensically analyse a single URL and return findings + a 0..1 risk."""
    findings = []
    parsed = urlparse(_normalise(raw_url))
    userinfo_present = "@" in parsed.netloc
    host = parsed.hostname or ""
    port = parsed.port
    path = parsed.path or ""
    reg, sub = _registrable_domain(host) if host else ("", [])
    tld = reg.rsplit(".", 1)[-1] if "." in reg else ""

    def add(name, severity, reason):
        findings.append({"name": name, "severity": severity, "reason": reason})

    if host and _looks_like_ip(host):
        add("ip_literal", 0.9,
            "The link points to a raw IP address rather than a named domain — "
            "legitimate organisations rarely do this.")
    if userinfo_present:
        add("userinfo_obfuscation", 0.9,
            "The URL contains an '@', which can hide the real destination after "
            "a trusted-looking name.")
    if "xn--" in host:
        add("punycode", 0.8,
            "The domain uses punycode (xn--), often used to imitate a real brand "
            "with look-alike characters.")
    if reg in URL_SHORTENERS:
        add("url_shortener", 0.5,
            "The link uses a URL shortener, which hides where it actually leads.")
    if tld in SUSPICIOUS_TLDS:
        add("suspicious_tld", 0.55,
            f"The domain ends in .{tld}, a top-level domain frequently used in scams.")
    if len(sub) >= 3:
        add("excessive_subdomains", 0.4,
            "The link has an unusually deep sub-domain, a common way to disguise "
            "the true domain.")
    if port and port not in (80, 443):
        add("non_standard_port", 0.45,
            f"The link uses a non-standard network port ({port}).")
    if "%" in host or re.search(r"0x[0-9a-f]+", host, re.I):
        add("hex_or_encoded", 0.6,
            "The host name is encoded/obfuscated rather than written plainly.")

    # Brand-placement checks.
    hay_host = host.lower()
    hay_path = path.lower()
    for brand in KNOWN_BRANDS:
        in_reg = brand in reg.split(".")
        in_sub = any(brand in s for s in sub)
        in_host = brand in hay_host
        in_path = brand in hay_path
        if in_sub and not in_reg:
            add("deceptive_subdomain", 0.85,
                f"'{brand}' appears as a sub-domain of '{reg}', not as the real "
                f"domain — a classic impersonation pattern.")
            break
        if (in_host or in_path) and not in_reg:
            add("brand_not_registrable", 0.7,
                f"'{brand}' is named in the link but the actual domain is '{reg}'. "
                f"The link is not controlled by {brand}.")
            break

    link_risk = min(1.0, max([f["severity"] for f in findings], default=0.0))
    return {
        "url": raw_url,
        "registrable_domain": reg,
        "findings": findings,
        "link_risk": round(link_risk, 3),
    }


def extract_urls(text: str) -> list:
    return [m.group(1).rstrip(".,;)]") for m in _URL_RE.finditer(text or "")]


def analyse_links(text: str) -> dict:
    """
    Analyse every URL in a message.

    Returns
    -------
    dict with:
        urls        : list of raw URLs found
        per_url     : list of per-URL analyses
        findings    : flattened, de-duplicated findings across all URLs
        link_risk   : max risk across URLs, in [0,1]
        note        : short human-readable status
    """
    urls = extract_urls(text)
    if not urls:
        return {"urls": [], "per_url": [], "findings": [], "link_risk": 0.0,
                "note": "No links found in the message."}

    per_url = [analyse_url(u) for u in urls]
    seen, findings = set(), []
    for pu in per_url:
        for f in pu["findings"]:
            if f["name"] not in seen:
                seen.add(f["name"])
                findings.append(f)
    link_risk = max((pu["link_risk"] for pu in per_url), default=0.0)
    if findings:
        note = f"{len(urls)} link(s) analysed; {len(findings)} suspicious link trait(s) found."
    else:
        note = f"{len(urls)} link(s) analysed; no suspicious link traits found."
    return {"urls": urls, "per_url": per_url, "findings": findings,
            "link_risk": round(link_risk, 3), "note": note}


if __name__ == "__main__":
    tests = [
        "Verify now at http://paypal.com.secure-login.tk/auth",
        "Your parcel: https://bit.ly/3xR9 track here",
        "Log in at http://192.168.1.45/login",
        "Update billing: https://account.microsoft.evil-domain.xyz/reset",
        "See your order at https://www.amazon.co.uk/orders — all good",
        "Meeting notes: https://docs.google.com/document/d/123",
    ]
    for t in tests:
        r = analyse_links(t)
        print(f"\n{t}")
        print(f"  link_risk={r['link_risk']}  ({r['note']})")
        for f in r["findings"]:
            print(f"   - {f['name']} [{f['severity']}]: {f['reason']}")