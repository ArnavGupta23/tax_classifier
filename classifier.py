import joblib
from rules import rule_based_label

# Load the pre‑trained pipeline (assumes train_model.py has been run)
_pipeline = joblib.load("models/tax_deductible_clf.joblib")

def classify_text(text: str, use_ml: bool = True):
    """
    Classify a single transaction text.
    
    Args:
        text (str): Combined merchant + description, lowercase.
        use_ml (bool): If True, use the ML model; otherwise, fallback to rules.
    
    Returns:
        label (int): 1 for deductible, 0 for non‑deductible
        reason (str): Explanation or probability
    """
    if use_ml:
        # Model returns [ [prob_non_deductible, prob_deductible] ]
        probs = _pipeline.predict_proba([text])[0]
        prob_deductible = probs[1]
        label = int(prob_deductible >= 0.5)
        reason = f"ML (p={prob_deductible:.2f})"
    else:
        # Pure rule-based fallback
        label = rule_based_label(text)
        reason = "Rule-based"
    return label, reason
