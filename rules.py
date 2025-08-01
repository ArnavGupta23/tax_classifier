from rules_config import COMPILED_RULES


def rule_based_label(text: str):
    text = text.lower()
    for rule in COMPILED_RULES:
        if rule["regex"].search(text):
            # If label is None, you’ll defer to ML
            return rule["label"], rule["reason"]
    # Should never reach here because of fallback
