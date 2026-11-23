"""
DeepGuard Temporal Information Extractor
Extracts event dates, temporal anchors, date ranges, and classifies whether
a claim asserts a current event, historical report, or timeless statement.
Uses standard Python datetime, pytz, and dateutil.
"""

import re
from datetime import datetime, timezone
import dateutil.parser as dparser

# Relative expressions anchored to the current moment
CURRENT_EVENT_PATTERNS = [
    re.compile(r'\b(today|right now|currently|at present|this morning|this afternoon|this evening|tonight)\b', re.IGNORECASE),
    re.compile(r'\b(breaking(\s+news)?|just in|happening now|urgent|alert|developing story)\b', re.IGNORECASE),
    re.compile(r'\b(this week|past few hours|few hours ago|minutes ago|moments ago)\b', re.IGNORECASE),
]

RECENT_PAST_PATTERNS = [
    re.compile(r'\b(yesterday|last night|earlier today)\b', re.IGNORECASE),
    re.compile(r'\b(last week|past week|past few days|recently|in recent days)\b', re.IGNORECASE),
    re.compile(r'\b(\d+)\s+(hours?|days?|mins?|minutes?)\s+ago\b', re.IGNORECASE),
]

# Year patterns: 1900 to 2099
YEAR_PATTERN = re.compile(r'\b(19\d\d|20\d\d)\b')

# Explicit month names
MONTH_NAMES = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)'

# Matches formats like: "November 15, 2026", "15 Nov 2026", "2026-11-15", "15th of October 2022"
EXPLICIT_DATE_PATTERNS = [
    re.compile(rf'\b({MONTH_NAMES}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}})\b', re.IGNORECASE),
    re.compile(rf'\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH_NAMES}(?:,)?\s+\d{{4}})\b', re.IGNORECASE),
    re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'),
    re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'),
]

# Past-event indicator verbs
HISTORICAL_INDICATORS = [
    re.compile(r'\b(occurred in|happened in|struck in|took place in|back in|during)\s+(19\d\d|20\d\d)\b', re.IGNORECASE),
    re.compile(r'\b(anniversary|remembering|memorial|years ago|decade ago|historical|archived)\b', re.IGNORECASE),
]


def extract_temporal_info(text, reference_datetime=None):
    """
    Extracts temporal expressions, claimed event dates, and temporal context from text.

    Args:
        text (str): Input claim or news headline/body.
        reference_datetime (datetime, optional): Time against which relative dates
            ('today', 'yesterday') are evaluated. Defaults to current UTC time.

    Returns:
        dict: {
            "has_temporal_expression": bool,
            "claim_type": "CURRENT_EVENT" | "HISTORICAL_EVENT" | "SPECIFIC_DATE_EVENT" | "TIMELESS",
            "extracted_date": str (ISO YYYY-MM-DD) or None,
            "extracted_year": int or None,
            "temporal_expressions": list of str,
            "is_current_claim": bool,
            "date_range": tuple (start_iso, end_iso) or None,
            "explanation": str
        }
    """
    if not text or not isinstance(text, str):
        return {
            "has_temporal_expression": False,
            "claim_type": "TIMELESS",
            "extracted_date": None,
            "extracted_year": None,
            "temporal_expressions": [],
            "is_current_claim": False,
            "date_range": None,
            "explanation": "No text provided for temporal extraction."
        }

    clean_text = text.strip()
    ref_dt = reference_datetime or datetime.now(timezone.utc)
    found_expressions = []
    extracted_date = None
    extracted_year = None
    claim_type = "TIMELESS"
    is_current_claim = False

    # 1. Check for historical phrasing (e.g., "Flooding occurred in Lahore in 2022")
    for pat in HISTORICAL_INDICATORS:
        m = pat.search(clean_text)
        if m:
            found_expressions.append(m.group(0))
            claim_type = "HISTORICAL_EVENT"
            # Try to grab year from group
            for grp in m.groups():
                if grp and grp.isdigit():
                    extracted_year = int(grp)
                    break

    # 2. Check for explicit full calendar dates (e.g. "November 15, 2026")
    for pat in EXPLICIT_DATE_PATTERNS:
        matches = pat.findall(clean_text)
        for match in matches:
            date_str = match if isinstance(match, str) else match[0]
            try:
                # Strip ordinals like 15th -> 15
                clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str, flags=re.IGNORECASE)
                parsed = dparser.parse(clean_date_str, fuzzy=True)
                extracted_date = parsed.strftime("%Y-%m-%d")
                extracted_year = parsed.year
                found_expressions.append(date_str)
                if claim_type != "HISTORICAL_EVENT":
                    claim_type = "SPECIFIC_DATE_EVENT"
                break
            except Exception:
                continue
        if extracted_date:
            break

    # 3. Check for standalone explicit years if not yet found (e.g., "in 2022")
    if not extracted_year:
        years = YEAR_PATTERN.findall(clean_text)
        if years:
            extracted_year = int(years[0])
            found_expressions.append(years[0])
            if claim_type == "TIMELESS":
                # If year is in the past, treat as historical unless explicitly claimed as current
                if extracted_year < ref_dt.year:
                    claim_type = "HISTORICAL_EVENT"
                else:
                    claim_type = "SPECIFIC_DATE_EVENT"

    # 4. Check for current-event phrases (e.g. "today", "breaking", "happening now")
    for pat in CURRENT_EVENT_PATTERNS:
        matches = pat.findall(clean_text)
        if matches:
            for m in matches:
                expr = m if isinstance(m, str) else m[0]
                if expr and expr not in found_expressions:
                    found_expressions.append(expr)
            is_current_claim = True
            claim_type = "CURRENT_EVENT"
            extracted_date = ref_dt.strftime("%Y-%m-%d")
            extracted_year = ref_dt.year

    # 5. Check for recent past expressions (e.g., "yesterday", "last week")
    if not is_current_claim:
        for pat in RECENT_PAST_PATTERNS:
            matches = pat.findall(clean_text)
            if matches:
                for m in matches:
                    expr = m if isinstance(m, str) else m[0]
                    if expr and expr not in found_expressions:
                        found_expressions.append(expr)
                is_current_claim = True  # Recent timeframe
                claim_type = "CURRENT_EVENT"
                if not extracted_date:
                    extracted_date = ref_dt.strftime("%Y-%m-%d")
                    extracted_year = ref_dt.year

    has_temporal = len(found_expressions) > 0 or extracted_date is not None or extracted_year is not None

    # Construct descriptive summary
    if claim_type == "CURRENT_EVENT":
        explanation = f"Text explicitly asserts a current/breaking event timeframe ('{', '.join(found_expressions)}'). Evaluated against reference date {ref_dt.strftime('%Y-%m-%d')}."
    elif claim_type == "HISTORICAL_EVENT":
        explanation = f"Text references a historical event occurring in {extracted_year or 'a past period'} ('{', '.join(found_expressions)}')."
    elif claim_type == "SPECIFIC_DATE_EVENT":
        explanation = f"Text specifies an event date on {extracted_date or extracted_year}."
    else:
        explanation = "No explicit temporal expressions or dates detected in the text."

    return {
        "has_temporal_expression": has_temporal,
        "claim_type": claim_type,
        "extracted_date": extracted_date,
        "extracted_year": extracted_year,
        "temporal_expressions": list(dict.fromkeys(found_expressions)),  # unique list
        "is_current_claim": is_current_claim,
        "reference_date": ref_dt.strftime("%Y-%m-%d"),
        "explanation": explanation
    }
