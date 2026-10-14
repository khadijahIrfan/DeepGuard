import re


# =====================================
# Viral Pattern Analysis
# =====================================

def analyze_viral_patterns(text):
    """
    Detect observable language and formatting
    patterns commonly associated with highly
    shareable or attention-driven content.

    This analysis does NOT determine whether
    content is REAL or FAKE.
    """

    if not isinstance(text, str):
        return {
            "error": "Invalid text input."
        }

    text = text.strip()

    if not text:
        return {
            "error": "Text is required."
        }

    # =====================================
    # Basic Statistics
    # =====================================

    words = text.split()
    word_count = len(words)

    if word_count > 0:
        uppercase_words = [
            word for word in words
            if word.isalpha() and word.isupper() and len(word) >= 2
        ]

        uppercase_ratio = (
            len(uppercase_words) / word_count
        ) * 100
    else:
        uppercase_words = []
        uppercase_ratio = 0

    # =====================================
    # 1. Urgency / Breaking Language
    # =====================================

    urgency_patterns = [
        r"\bbreaking\b",
        r"\burgent\b",
        r"\balert\b",
        r"\bjust in\b",
        r"\bdeveloping\b",
        r"\bwarning\b",
        r"\bimmediately\b",
        r"\bshocking\b"
    ]

    urgency_matches = []

    for pattern in urgency_patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )
        urgency_matches.extend(matches)

    urgency_detected = len(urgency_matches) > 0

    # =====================================
    # 2. Emotional / Sensational Language
    # =====================================

    emotional_words = [
        "shocking",
        "terrifying",
        "horrifying",
        "outrage",
        "outrageous",
        "unbelievable",
        "incredible",
        "terrible",
        "disaster",
        "catastrophe",
        "miracle",
        "scandal",
        "explosive",
        "dangerous",
        "deadly",
        "horrific",
        "stunning",
        "devastating"
    ]

    emotional_matches = []

    for word in emotional_words:
        if re.search(
            rf"\b{re.escape(word)}\b",
            text,
            flags=re.IGNORECASE
        ):
            emotional_matches.append(word)

    emotional_detected = len(
        emotional_matches
    ) > 0

    # =====================================
    # 3. Excessive Punctuation
    # =====================================

    exclamation_count = text.count("!")
    question_count = text.count("?")

    excessive_exclamation = bool(
        re.search(r"!{2,}", text)
    )

    excessive_question = bool(
        re.search(r"\?{2,}", text)
    )

    excessive_punctuation = (
        excessive_exclamation
        or excessive_question
    )

    # =====================================
    # 4. ALL-CAPS Usage
    # =====================================

    all_caps_detected = uppercase_ratio >= 10

    # =====================================
    # 5. Call-to-Action Language
    # =====================================

    call_to_action_patterns = [
        r"\bshare this\b",
        r"\bshare now\b",
        r"\bshare with\b",
        r"\bspread the word\b",
        r"\btell everyone\b",
        r"\bplease share\b",
        r"\bshare before\b",
        r"\bdo not ignore\b",
        r"\bdon't ignore\b",
        r"\bmake this viral\b"
    ]

    cta_matches = []

    for pattern in call_to_action_patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )
        cta_matches.extend(matches)

    call_to_action_detected = len(
        cta_matches
    ) > 0

    # =====================================
    # 6. Hashtag Analysis
    # =====================================

    hashtags = re.findall(
        r"#[A-Za-z0-9_]+",
        text
    )

    hashtag_count = len(hashtags)

    hashtag_density = (
        hashtag_count / word_count * 100
        if word_count > 0
        else 0
    )

    hashtag_usage = (
        hashtag_count > 0
    )

    high_hashtag_density = (
        hashtag_density >= 10
    )

    # =====================================
    # 7. Repeated Words
    # =====================================

    normalized_words = [
        re.sub(
            r"[^a-zA-Z0-9]",
            "",
            word.lower()
        )
        for word in words
    ]

    normalized_words = [
        word
        for word in normalized_words
        if word
    ]

    word_frequency = {}

    for word in normalized_words:
        word_frequency[word] = (
            word_frequency.get(word, 0) + 1
        )

    repeated_words = [
        word
        for word, count in word_frequency.items()
        if count >= 3 and len(word) > 2
    ]

    repeated_word_detected = (
        len(repeated_words) > 0
    )

    # =====================================
    # Pattern Count
    # =====================================

    detected_patterns = 0

    if urgency_detected:
        detected_patterns += 1

    if emotional_detected:
        detected_patterns += 1

    if excessive_punctuation:
        detected_patterns += 1

    if all_caps_detected:
        detected_patterns += 1

    if call_to_action_detected:
        detected_patterns += 1

    if hashtag_usage:
        detected_patterns += 1

    if repeated_word_detected:
        detected_patterns += 1

    # =====================================
    # Overall Pattern Description
    # =====================================

    if detected_patterns == 0:
        overall = (
            "No notable viral-style patterns detected."
        )
    elif detected_patterns <= 2:
        overall = (
            "A small number of viral-style patterns "
            "were detected."
        )
    elif detected_patterns <= 4:
        overall = (
            "Several viral-style patterns "
            "were detected."
        )
    else:
        overall = (
            "Multiple viral-style patterns "
            "were detected."
        )

    # =====================================
    # Return Results
    # =====================================

    return {
        "overall": overall,

        "patterns_detected": detected_patterns,

        "urgency": {
            "detected": urgency_detected,
            "matches": urgency_matches
        },

        "emotional_language": {
            "detected": emotional_detected,
            "matches": emotional_matches
        },

        "excessive_punctuation": {
            "detected": excessive_punctuation,
            "exclamation_marks": exclamation_count,
            "question_marks": question_count
        },

        "all_caps": {
            "detected": all_caps_detected,
            "uppercase_ratio": round(
                uppercase_ratio,
                2
            )
        },

        "call_to_action": {
            "detected": call_to_action_detected,
            "matches": cta_matches
        },

        "hashtags": {
            "detected": hashtag_usage,
            "count": hashtag_count,
            "density": round(
                hashtag_density,
                2
            ),
            "high_density": high_hashtag_density
        },

        "repeated_words": {
            "detected": repeated_word_detected,
            "words": repeated_words
        }
    }

