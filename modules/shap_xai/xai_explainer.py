"""
DeepGuard Module 7: Explainable AI (XAI) & Feature Attribution Engine
Identifies top contributing tokens, linguistic triggers, and explains predictions.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from xai_analysis import get_important_words
except Exception:
    def get_important_words(text):
        return []

def explain_text_prediction(text):
    """Generates explainability token attributions for the input text."""
    return get_important_words(text)
