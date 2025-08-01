def rule_based_label(text: str) -> int:
    """
    A simple rule‑based function to assign a weak label:
    1 for deductible, 0 for non‑deductible.
    
    We’ll use this to bootstrap our ML training data.
    """
    
    """
    Weak-label overrides before ML training.
    """
    text = text.lower()

    # 1) Negative overrides
    for kw in ["vacation", "holiday", "personal", "family", "trip"]:
        if kw in text:
            return 0

    # 2) Positive overrides for business categories
    if "coworking" in text:
        return 1
    if any(kw in text for kw in ["course", "training"]) and "online" in text:
        return 1
    if any(kw in text for kw in ["macbook", "laptop", "computer"]) and "work" in text:
        return 1

    # 3) Existing rules
    for kw in ["flight", "airlines", "hotel", "uber", "lyft"]:
        if kw in text:
            return 1
    for kw in ["lunch", "dinner", "cafe", "restaurant", "meal"]:
        if kw in text:
            return 1
    for kw in ["zoom", "slack", "aws", "github", "azure", "gcp"]:
        if kw in text:
            return 1

    # Default: non-deductible
    return 0