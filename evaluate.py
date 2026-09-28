"""
TrustLens — Evaluation Script (hand-crafted out-of-domain test set)
--------------------------------------------------------------------
This evaluation uses a manually constructed test set of realistic
phishing/scam and benign messages. This avoids the data-leakage issue
of evaluating a pre-trained model on the very dataset it was trained on
(which would produce artificially perfect scores).

The test set covers multiple scam categories named in the literature:
    - Account-lock / suspension scams
    - Payment-pressure / invoice scams
    - Prize / lottery scams
    - Delivery / parcel scams
    - Tax-refund scams
    - Brand-impersonation scams
    - Plus benign messages of five types (personal, transactional, marketing,
      work, and adversarial "tricky" messages with scam-like words)

Each test message is annotated with:
    - Ground-truth label (phishing=1, benign=0)
    - Scam category (or "benign")

Run with:
    python evaluate.py
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from classifier import classify_text
from indicators import extract_indicators
from aggregator import aggregate_risk


TEST_CASES = [
    # Account-lock / suspension scams
    ("Your PayPal account has been temporarily restricted due to suspicious activity. "
     "Verify your password immediately at http://192.168.1.45/verify or your account will be closed.",
     1, "account_lock"),
    ("URGENT: We detected unauthorised access to your Microsoft account. "
     "Confirm your identity within 24 hours: http://ms-secure-login.tk/auth",
     1, "account_lock"),
    ("Your Apple ID has been locked. Re-enter your password here to unlock: "
     "http://appleid-verify.support/reset",
     1, "account_lock"),

    # Payment-pressure / invoice scams
    ("Your recent payment of $429.99 has failed. Update your billing information immediately "
     "to avoid service disruption: http://billing-update-now.net",
     1, "payment_pressure"),
    ("Final notice: Your unpaid invoice #INV-77321 is overdue. Wire transfer required within 48 hours "
     "to avoid legal action.",
     1, "payment_pressure"),
    ("Your tax refund of £540 is pending. Confirm your bank details urgently at hmrc-refund-claim.com",
     1, "tax_refund"),

    # Prize / lottery scams
    ("CONGRATULATIONS! You have been selected to receive a $1000 Amazon gift card. "
     "Click here to claim your reward before it expires today!",
     1, "prize_lure"),
    ("You won a brand new iPhone 15 Pro! Claim now at free-iphone-winner.xyz",
     1, "prize_lure"),
    ("Lucky winner! Your number was drawn in our monthly giveaway. "
     "Send your bank details to receive your $5000 prize.",
     1, "prize_lure"),

    # Delivery / parcel scams
    ("Your DHL parcel could not be delivered. Pay the £2.99 redelivery fee at dhl-redelivery.info",
     1, "delivery_scam"),
    ("FedEx: Your package is being held at customs. Confirm details at fedex-customs-clear.com to release.",
     1, "delivery_scam"),

    # Brand impersonation
    ("Netflix: Your subscription payment failed. Update your card now: http://netflix-billing.support",
     1, "brand_impersonation"),
    ("Singpass alert: Suspicious login detected. Verify your identity at singpass-secure.gov-sg.co",
     1, "brand_impersonation"),

    # Benign: personal
    ("Hi mum, just letting you know I'll be home for dinner tonight. See you at 7.",
     0, "benign"),
    ("Hey, are we still on for coffee tomorrow at 10? Let me know if you need to reschedule.",
     0, "benign"),
    ("Thanks for sending over the notes from yesterday's lecture, really appreciate it!",
     0, "benign"),

    # Benign: transactional
    ("Your Amazon order #404-1234567-8901234 has shipped and will arrive on Friday.",
     0, "benign"),
    ("Your appointment with Dr. Lim is confirmed for 3 March at 2:30 PM.",
     0, "benign"),
    ("Your monthly bank statement is ready to view in the app.",
     0, "benign"),

    # Benign: legitimate marketing
    ("New collection just dropped! Browse the latest arrivals at our store this weekend.",
     0, "benign"),
    ("Don't forget — your local library has a free reading event this Saturday.",
     0, "benign"),

    # Benign: work / professional
    ("Please review the attached report by end of week and share your feedback.",
     0, "benign"),
    ("The team meeting has been moved to 11am tomorrow. Calendar invite to follow.",
     0, "benign"),

    # Adversarial benign: contain SOME scam-like keywords but are actually safe
    ("Just a reminder that your gym membership payment is due next week. "
     "You can manage it through the app or visit the front desk.",
     0, "benign_tricky"),
    ("Your password to the office Wi-Fi has been updated. Please ask IT if you need the new one.",
     0, "benign_tricky"),
]


def evaluate():
    print(f"[eval] Running evaluation on {len(TEST_CASES)} hand-crafted test cases")
    print("[eval] These messages are NOT from any public dataset — they were written")
    print("[eval] specifically for TrustLens to avoid data leakage with the pre-trained model.\n")

    rows = []
    for text, true_label, category in TEST_CASES:
        cls = classify_text(text)
        inds = extract_indicators(text)
        assessment = aggregate_risk(cls["phishing_prob"], inds)

        pred = 1 if assessment["level"] in ("Suspicious", "High Risk") else 0

        rows.append({
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "category": category,
            "true": true_label,
            "pred": pred,
            "phishing_prob": cls["phishing_prob"],
            "indicators": len(inds),
            "score": assessment["score"],
            "level": assessment["level"],
            "correct": pred == true_label,
        })

    df = pd.DataFrame(rows)

    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("=" * 62)
    print("=== OVERALL METRICS (hand-crafted test set) ===")
    print("=" * 62)
    print(f"  Test cases:         {len(df)}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1-score:           {f1:.4f}")
    print(f"  False-positive rate:{fpr:.4f}")
    print("  Confusion matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    print("\n" + "=" * 62)
    print("=== ACCURACY BY SCAM CATEGORY ===")
    print("=" * 62)
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        cat_acc = sub["correct"].mean()
        print(f"  {cat:<25} {int(sub['correct'].sum())}/{len(sub)} ({cat_acc:.0%})")

    df.to_csv("evaluation_results.csv", index=False)
    print("\n[eval] Per-row results saved to evaluation_results.csv")

    pd.DataFrame([{
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "fpr": fpr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test_cases": len(df),
    }]).to_csv("evaluation_metrics.csv", index=False)
    print("[eval] Metrics summary saved to evaluation_metrics.csv")

    failures = df[~df["correct"]]
    if len(failures) > 0:
        print("\n" + "=" * 62)
        print(f"=== FAILED CASES ({len(failures)}) ===")
        print("=" * 62)
        for _, row in failures.iterrows():
            print(f"\n  [{row['category']}] true={row['true']} pred={row['pred']} "
                  f"score={row['score']} level={row['level']}")
            print(f"    {row['text']}")
    else:
        print("\n[eval] All test cases classified correctly.")


if __name__ == "__main__":
    evaluate()
