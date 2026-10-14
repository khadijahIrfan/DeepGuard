"""
DeepGuard Module 6: Sentiment Analysis Engine
Evaluates emotional intensity, tone polarities, and sentiment markers using VADER.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def analyze_sentiment(text):
    """
    Computes sentiment polarity and returns formatted label with emoji and detailed scores.
    """
    if not text or not text.strip():
        return {
            "label": "Neutral 😐",
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0
        }

    scores = get_analyzer().polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive 😊"
    elif compound <= -0.05:
        label = "Negative 😠"
    else:
        label = "Neutral 😐"

    return {
        "label": label,
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"]
    }
