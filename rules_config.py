# rules_config.py

import re

# Ordered list of tax deductibility rules (first match wins)
RULES = [
    # 0) Personal Zoom calls with family → non-deductible
    {
        "pattern": r"\bzoom\b.*\b(mom|dad|mother|father|brother|sister|family|parent)\b",
        "label": 0,
        "reason": "Personal Zoom call"
    },

    # 1) Groceries and household supplies → non-deductible
    {
        "pattern": r"\b(grocery|groceries|household supplies|walmart|target|costco)\b",
        "label": 0,
        "reason": "Personal groceries or household supplies"
    },

    # 2) General personal meals → non-deductible
    {
        "pattern": r"\b(lunch|dinner|breakfast|brunch|meal|restaurant|cafe)\b",
        "label": 0,
        "reason": "Personal meal"
    },

    # 3) Clearly personal expenses (vacation, gifts, etc.) → non-deductible
    {
        "pattern": r"\b(vacation|holiday|gift|birthday|anniversary|party|celebration)\b",
        "label": 0,
        "reason": "Personal occasion"
    },

    # 4) Office supplies from known vendors → deductible
    {
        "pattern": r"\b(staples|office depot|office max|office supplies)\b",
        "label": 1,
        "reason": "Office supplies"
    },

    # 5) Business-related equipment purchases → deductible
    {
        "pattern": r"\b(macbook|laptop|computer|monitor|printer|keyboard|mouse|webcam|headset)\b.*\b(work|business|freelance|remote job|job)\b",
        "label": 1,
        "reason": "Business equipment purchase"
    },

    # 6) Coworking and shared office spaces → deductible
    {
        "pattern": r"\b(wework|regus|coworking|shared office|office space rental)\b",
        "label": 1,
        "reason": "Coworking membership"
    },

    # 7) Online courses or training → deductible
    {
        "pattern": r"\b(online course|webinar|training|certification|bootcamp|coursera|udemy|edx)\b",
        "label": 1,
        "reason": "Professional development"
    },

    # 8) Transportation for business (gas, mileage, etc.) → deductible
    {
        "pattern": r"\b(gas|fuel|mileage|toll|parking)\b.*\b(business|client|meeting|work)\b",
        "label": 1,
        "reason": "Business transportation"
    },

    # 9) Business meals (with team/client context) → deductible
    {
        "pattern": (
            r"\b(lunch|dinner|breakfast|meal)\b.*\b(client|team|coworker|meeting|business|work)\b|"
            r"\b(client|team|coworker|meeting|business|work)\b.*\b(lunch|dinner|breakfast|meal)\b"
        ),
        "label": 1,
        "reason": "Business meal"
    },

    # 10) Business travel (flight, hotel, rideshare) → deductible
    {
        "pattern": (
            r"\b(flight|airfare|airlines|hotel|lyft|uber|taxi|train|rail|airbnb|car rental|ride)\b"
            r".*\b(work|business|client|meeting|office|job|site)\b"
        ),
        "label": 1,
        "reason": "Business travel"
    },

    # 11) Work-related software & subscriptions → deductible
    {
        "pattern": r"\b(zoom|slack|aws|github|jira|notion|dropbox|adobe|microsoft 365|gcp|azure|quickbooks|intuit)\b",
        "label": 1,
        "reason": "Business software subscription"
    },

    # 12) Amazon + work-related keywords → deductible
    {
        "pattern": r"\b(amazon)\b.*\b(office|supplies|business|equipment|work)\b",
        "label": 1,
        "reason": "Office-related Amazon purchase"
    },

    # 13) Fallback: no rule match → defer to ML
    {
        "pattern": r".*",
        "label": None,
        "reason": "No rule matched — defer to ML"
    }
]

# Precompiled regex rules for performance
COMPILED_RULES = [
    {
        **rule,
        "regex": re.compile(rule["pattern"], re.IGNORECASE)
    }
    for rule in RULES
]
