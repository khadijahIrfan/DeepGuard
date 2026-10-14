import pathlib
import logging
import torch

try:
    import joblib
    from transformers import AutoTokenizer, DistilBertForSequenceClassification
    HAS_TEXT_DEPS = True
except ImportError:
    HAS_TEXT_DEPS = False

logger = logging.getLogger(__name__)

MODULE_DIR = pathlib.Path(__file__).parent.resolve()
BASE_DIR = MODULE_DIR.parent.parent

# Paths to trained text models
DISTILBERT_PATH = BASE_DIR / "models" / "final_distilbert_2026" / "FINAL_MODEL"
SVM_PATH = BASE_DIR / "models" / "FINAL ALL DATSETS" / "deepguard_final_svm.pkl"
VEC_PATH = BASE_DIR / "models" / "FINAL ALL DATSETS" / "deepguard_final_vectorizer.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextAnalyzer:
    """
    Module 1 — Text Analysis Module for DeepGuard.
    Uses DistilBERT (98% accuracy) as primary classifier and LinearSVC TF-IDF (96% accuracy) as fallback.
    """
    def __init__(self):
        self.distilbert_model = None
        self.tokenizer = None
        self.svm_model = None
        self.vectorizer = None

    def _load_distilbert(self):
        if self.distilbert_model is not None and self.tokenizer is not None:
            return True

        if not DISTILBERT_PATH.exists():
            logger.warning(f"DistilBERT model path not found: {DISTILBERT_PATH}")
            return False

        try:
            logger.info("Loading DistilBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_PATH))
            self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_PATH)).to(DEVICE)
            self.distilbert_model.eval()
            return True
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            return False

    def _load_svm(self):
        if self.svm_model is not None and self.vectorizer is not None:
            return True

        if not SVM_PATH.exists() or not VEC_PATH.exists():
            logger.warning(f"SVM or Vectorizer model files missing.")
            return False

        try:
            logger.info("Loading SVM fallback model...")
            self.svm_model = joblib.load(SVM_PATH)
            self.vectorizer = joblib.load(VEC_PATH)
            return True
        except Exception as e:
            logger.error(f"Failed to load SVM model: {e}")
            return False

    def predict(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "prediction": "UNKNOWN",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "model_used": "None",
                "status": "empty_input"
            }

        text = text.strip()

        # Enforce minimum character limit — fewer than 20 characters is not enough context
        MIN_CHARS = 20
        if len(text) < MIN_CHARS:
            return {
                "prediction": "UNKNOWN",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "model_used": "None",
                "status": "insufficient_text",
                "char_count": len(text),
                "message": f"Text too short ({len(text)} characters). Minimum {MIN_CHARS} characters required for analysis."
            }

        # Enforce max character limit
        MAX_CHARS = 1200
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]  # Truncate safely at char limit

        if self._load_distilbert():
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = self.distilbert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]

                # Label 0: REAL, Label 1: FAKE
                real_prob = float(probs[0].item())
                fake_prob = float(probs[1].item())

                is_fake = fake_prob >= real_prob
                prediction = "FAKE" if is_fake else "REAL"
                confidence = round((fake_prob if is_fake else real_prob) * 100, 2)
                fake_percent = round(fake_prob * 100, 2)

                return {
                    "prediction": prediction,
                    "is_fake": is_fake,
                    "fake_probability": fake_percent,
                    "confidence": confidence,
                    "model_used": "DistilBERT (98% Acc)",
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"DistilBERT prediction failed, switching to SVM: {e}")

        # Fallback to SVM
        if self._load_svm():
            try:
                features = self.vectorizer.transform([text])
                pred_label = self.svm_model.predict(features)[0]
                
                # Check decision function for confidence approximation
                if hasattr(self.svm_model, "decision_function"):
                    df = self.svm_model.decision_function(features)[0]
                    # Sigmoid for pseudo-probability
                    import math
                    fake_prob = 1.0 / (1.0 + math.exp(-df))
                else:
                    fake_prob = 1.0 if pred_label == 1 or str(pred_label).upper() in ["FAKE", "1"] else 0.0

                is_fake = pred_label == 1 or str(pred_label).upper() in ["FAKE", "1"] or fake_prob >= 0.5
                prediction = "FAKE" if is_fake else "REAL"
                fake_percent = round(fake_prob * 100, 2)
                confidence = round(max(fake_prob, 1.0 - fake_prob) * 100, 2)

                return {
                    "prediction": prediction,
                    "is_fake": is_fake,
                    "fake_probability": fake_percent,
                    "confidence": confidence,
                    "model_used": "SVM LinearSVC (96% Acc)",
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"SVM prediction failed: {e}")

        return {
            "prediction": "UNKNOWN",
            "fake_probability": 50.0,
            "confidence": 50.0,
            "model_used": "Fallback",
            "status": "failed"
        }

_text_analyzer = None

def analyze_text_fake(text):
    """
    Standard interface function for DeepGuard Backend.
    """
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer.predict(text)

if __name__ == "__main__":
    sample = "Government declared a national emergency effective immediately!"
    print("Testing Text Analyzer on:\n", sample)
    res = analyze_text_fake(sample)
    import json
    print(json.dumps(res, indent=2))
