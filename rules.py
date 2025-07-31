def rule_based_label(text: str) -> int:
    """
    A simple rule‑based function to assign a weak label:
    1 for deductible, 0 for non‑deductible.
    
    We’ll use this to bootstrap our ML training data.
    """
    # Common keywords for deductible categories
    business_travel = ["flight", "airlines", "hotel", "uber", "lyft"]
    business_meal   = ["lunch", "dinner", "cafe", "restaurant", "meal"]
    software_svcs   = ["zoom", "slack", "aws", "github", "azure", "gcp"]
    
    # Negative keywords (non‑deductible)
    for kw in ["vacation", "holiday", "personal", "family", "trip"]:
        if kw in text:
            return 0

    # If any travel keyword appears, label as deductible
    for kw in business_travel:
        if kw in text:
            return 1

    # If any meal keyword appears, label as deductible
    for kw in business_meal:
        if kw in text:
            return 1

    # If any software/service keyword appears, label as deductible
    for kw in software_svcs:
        if kw in text:
            return 1

    # Otherwise, treat as non‑deductible
    return 0
