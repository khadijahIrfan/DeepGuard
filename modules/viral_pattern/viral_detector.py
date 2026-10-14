import re
import logging

logger = logging.getLogger(__name__)

# Pattern Rules Database categorized by virus/manipulation taxonomy
VIRAL_PATTERNS = [
    # 1. Chain Message Threats & Forward Mandates (High Risk)
    {
        "category": "Chain Message / Forward Mandate",
        "weight": 25,
        "patterns": [
            r"forward\s+to\s+\d+\s+(groups|people|friends|contacts)",
            r"share\s+with\s+\d+\s+(groups|people|friends)",
            r"send\s+this\s+to\s+\d+",
            r"if\s+you\s+(ignore|don'?t\s+share)\s+(bad\s+luck|curse|disaster)",
            r"share\s+before\s+it\s+gets\s+(deleted|banned|removed)",
            r"don'?t\s+keep\s+this\s+to\s+yourself",
            r"pass\s+it\s+on\s+to\s+everyone",
            r"forwarded\s+as\s+received",
            r"bhejo\s+sab\s+ko",
            r"doston\s+ko\s+bhejo",
        ]
    },
    
    # 2. Pakistani WhatsApp Roman Urdu Viral Phrases (High Risk)
    {
        "category": "Pakistani WhatsApp Manipulation",
        "weight": 20,
        "patterns": [
            r"\babhi\s+share\s+kar(o|ein)\b",
            r"\bsab\s+ko\s+bata(o|ein)\b",
            r"\bfauran\s+share\s+kar(o|ein)\b",
            r"\bsachi\s+khabar\b",
            r"\bmazaq\s+mat\s+samjh(o|ein)\b",
            r"\bkhuda\s+ka\s+wasta\b",
            r"\basl\s+haqeeqat\b",
            r"\bsab\s+pakistani\s+sun(o|ein)\b",
            r"\bziyada\s+se\s+ziyada\s+share\b",
            r"\bhar\s+group\s+mai\s+bhejo\b",
        ]
    },

    # 3. False Urgency & Panic Triggers (Medium-High Risk)
    {
        "category": "False Urgency",
        "weight": 15,
        "patterns": [
            r"\bbreaking\s+news\b",
            r"\burgent\s+attention\b",
            r"\bshare\s+immediately\b",
            r"\bdon'?t\s+ignore\b",
            r"\bimportant\s+alert\b",
            r"\balert\s+alert\b",
            r"\bwarning\s+warning\b",
            r"\bemergency\s+update\b",
            r"\bread\s+before\s+it'?s\s+too\s+late\b",
        ]
    },

    # 4. Authority & Government Fake Endorsements (High Risk)
    {
        "category": "Authority Manipulation",
        "weight": 20,
        "patterns": [
            r"\bgovt\s+(has\s+)?declared\b",
            r"\bgovernment\s+announced\b",
            r"\bwho\s+(has\s+)?confirmed\b",
            r"\bunesco\s+(has\s+)?declared\b",
            r"\bpak\s+army\s+(issued|stated)\b",
            r"\bsupreme\s+court\s+ordered\b",
            r"\bhealth\s+ministry\s+warning\b",
            r"\bprime\s+minister\s+announced\b",
            r"\bofficial\s+notification\b",
        ]
    },

    # 5. Conspiracy Framing & Secret Hiding (Medium Risk)
    {
        "category": "Conspiracy Framing",
        "weight": 15,
        "patterns": [
            r"\bmedia\s+won'?t\s+show\s+you\b",
            r"\bhidden\s+truth\b",
            r"\bthey\s+don'?t\s+want\s+you\s+to\s+know\b",
            r"\bgovernment\s+is\s+hiding\b",
            r"\bsecret\s+formula\b",
            r"\bcure\s+that\s+doctors\s+hide\b",
            r"\bshocking\s+revelation\b",
        ]
    },

    # 6. Clickbait / Financial Scams & Free Giveaways (Medium Risk)
    {
        "category": "Financial / Scam Bait",
        "weight": 15,
        "patterns": [
            r"\bfree\s+recharge\b",
            r"\bgovt\s+giving\s+free\b",
            r"\bclick\s+this\s+link\s+to\s+claim\b",
            r"\bwin\s+free\s+iphone\b",
            r"\behsass?\s+programme?\s+money\b",
            r"\b8171\s+free\s+cash\b",
            r"\bget\s+rs\.?\s*\d+\b",
        ]
    }
]

class ViralPatternDetector:
    """
    Module 5 — Viral Pattern Detector for DeepGuard.
    Analyzes raw post text using compiled regular expressions to detect
    viral propagation tactics, chain message threats, false urgency, and regional (Pakistani) viral triggers.
    """
    def __init__(self):
        self.compiled_rules = []
        for group in VIRAL_PATTERNS:
            cat = group["category"]
            weight = group["weight"]
            for pat in group["patterns"]:
                regex = re.compile(pat, re.IGNORECASE)
                self.compiled_rules.append({
                    "category": cat,
                    "weight": weight,
                    "regex": regex,
                    "pattern_str": pat
                })

    def analyze(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "viral_score": 0.0,
                "is_viral_pattern": False,
                "patterns_detected": [],
                "matched_count": 0,
                "threat_level": "LOW",
                "status": "empty_input"
            }

        detected = []
        total_score = 0
        seen_matches = set()

        for rule in self.compiled_rules:
            match = rule["regex"].search(text)
            if match:
                matched_phrase = match.group(0)
                # Avoid duplicate identical string matches
                key = (rule["category"], matched_phrase.lower())
                if key not in seen_matches:
                    seen_matches.add(key)
                    detected.append({
                        "category": rule["category"],
                        "matched_phrase": matched_phrase,
                        "weight": rule["weight"]
                    })
                    total_score += rule["weight"]

        # Cap maximum viral score at 100%
        viral_score = min(100.0, float(total_score))
        is_viral = viral_score >= 30.0

        if viral_score >= 60.0:
            threat_level = "CRITICAL"
        elif viral_score >= 30.0:
            threat_level = "HIGH"
        elif viral_score > 0.0:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        return {
            "viral_score": round(viral_score, 2),
            "is_viral_pattern": is_viral,
            "patterns_detected": detected,
            "matched_count": len(detected),
            "threat_level": threat_level,
            "status": "success"
        }

# Global detector instance
_viral_detector = None

def detect_viral_patterns(text):
    """
    Standard interface function for DeepGuard backend.
    """
    global _viral_detector
    if _viral_detector is None:
        _viral_detector = ViralPatternDetector()
    return _viral_detector.analyze(text)


if __name__ == "__main__":
    test_sample = "BREAKING NEWS: Govt declared emergency! Forward to 10 groups immediately. Abhi share karo subko batao!"
    print("Testing Viral Pattern Detector on:\n", test_sample)
    result = detect_viral_patterns(test_sample)
    print("\nResult:")
    import json
    print(json.dumps(result, indent=2))
