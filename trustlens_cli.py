"""
TrustLens — Command-Line Interface
-----------------------------------
Scan messages from the terminal without starting the web server. Handy for
quick checks, scripting, and for the demo video.

Usage
-----
    python trustlens_cli.py "Your account is locked, verify now at http://..."
    python trustlens_cli.py --file messages.txt        # one message per line
    python trustlens_cli.py --calibrated "..."          # use the calibrated path
    python trustlens_cli.py --json "..."                # machine-readable output

Exit code is 0 for Safe/Caution and 2 for Suspicious/High Risk, so the tool can
be used in shell pipelines.
"""

import argparse
import json
import sys

from orchestrator import analyse_text

_COLOURS = {"Safe": "\033[92m", "Caution": "\033[93m",
            "Suspicious": "\033[95m", "High Risk": "\033[91m"}
_RESET = "\033[0m"


def _print_human(result: dict):
    a = result["assessment"]
    e = result["explanation"]
    level = a["level"]
    colour = _COLOURS.get(level, "")
    print("\n" + "=" * 68)
    print(f" RISK LEVEL: {colour}{level}{_RESET}   (score {a['score']})")
    print("=" * 68)
    print(f" {e['summary']}")
    if e["indicators"]:
        print("\n Indicators:")
        for ind in e["indicators"]:
            print(f"   - {ind['name']}: {ind['reason']}")
    if e.get("link_analysis"):
        print(f"\n Link analysis ({e['link_analysis']['note']}):")
        for f in e["link_analysis"]["findings"]:
            print(f"   - {f['name']}: {f['reason']}")
    if e.get("benign_context"):
        print(f"\n Benign context ({e['benign_context']['note']}):")
        for s in e["benign_context"]["signals"]:
            print(f"   + {s['name']}: {s['reason']}")
    if e.get("emotion_note"):
        print(f"\n Emotion: {e['emotion_note']}")
    print(f"\n Reasoning: {e['reasoning']}")
    print(f"\n Recommended action:\n   {e['safe_action']}")
    if not result.get("classifier_available", True):
        print("\n [!] Phishing model not installed here — score reflects rules/links only.")
    print(f"\n {e['disclaimer']}")


def main(argv=None):
    p = argparse.ArgumentParser(description="TrustLens phishing/scam risk CLI")
    p.add_argument("message", nargs="?", help="the message text to analyse")
    p.add_argument("--file", help="analyse each line of this file")
    p.add_argument("--calibrated", action="store_true",
                   help="use the calibrated, link-aware, context-aware scoring")
    p.add_argument("--json", action="store_true", help="output raw JSON")
    args = p.parse_args(argv)

    messages = []
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            messages = [ln.strip() for ln in fh if ln.strip()]
    elif args.message:
        messages = [args.message]
    else:
        p.error("provide a message or --file")

    worst = 0
    for msg in messages:
        result = analyse_text(msg, calibrate=args.calibrated)
        if "error" in result:
            print("Error:", result["error"]); continue
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_human(result)
        if result["assessment"]["level"] in ("Suspicious", "High Risk"):
            worst = 2
    return worst


if __name__ == "__main__":
    sys.exit(main())
