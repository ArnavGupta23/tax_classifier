import joblib
import pandas as pd
from rules import rule_based_label

# Load the tuned pipeline
_pipeline = joblib.load("models/tax_deductible_clf.joblib")

def classify_text(text: str, merchant: str, use_ml: bool = True):
    """
    Classify a transaction given its text and merchant.
    """
    text = text.lower()
    merchant = merchant if isinstance(merchant, str) else ""

    if use_ml:
        # Build a one-row DataFrame so ColumnTransformer can do its job
        X = pd.DataFrame([{"text": text, "merchant": merchant}])
        probs = _pipeline.predict_proba(X)[0]
        p = probs[1]
        label = int(p >= 0.5)
        reason = f"ML (p={p:.2f})"
    else:
        label = rule_based_label(text)
        reason = "Rule-based"
    return label, reason
