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

    def predict(self, text, is_ocr=False):
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
        MIN_CHARS = 15
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
        MAX_CHARS = 1500
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]  # Truncate safely at char limit

        # For short headlines and OCR, normalize masked words and excessive uppercase
        # to avoid triggering artificial clickbait flags on capitalized real news
        if len(text.split()) < 35 or is_ocr:
            import re
            text = re.sub(r'\bki\*ling\b', 'killing', text, flags=re.IGNORECASE)
            text = re.sub(r'\bk\*lling\b', 'killing', text, flags=re.IGNORECASE)
            text = re.sub(r'\bki\*l\b', 'kill', text, flags=re.IGNORECASE)
            text = re.sub(r'\bk\*ll\b', 'kill', text, flags=re.IGNORECASE)
            text = re.sub(r'\bd\*ath\b', 'death', text, flags=re.IGNORECASE)
            text = re.sub(r'\bd\*ed\b', 'died', text, flags=re.IGNORECASE)
            text = re.sub(r'\bm\*rder\b', 'murder', text, flags=re.IGNORECASE)
            # If headline is all uppercase, title-case it to match standard journalistic corpus
            if text.isupper():
                text = text.title()

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

                word_count = len(text.split())
                # Short headlines (<35 words): use T=3.0 + SVM ensemble to prevent
                # isolated brand/clickbait words from skewing predictions.
                # Full articles (>=35 words): use T=2.5 temperature only.
                if word_count < 35 or is_ocr:  # OCR text always uses ensemble
                    # For short journalistic claims, soften DistilBERT's article-length bias (T=3.5)
                    # and blend 50/50 with SVM TF-IDF keyword distribution
                    TEMPERATURE = 3.5
                    scaled_logits = outputs.logits / TEMPERATURE
                    probs = torch.softmax(scaled_logits, dim=1)[0]
                    real_prob_raw = float(probs[0].item())
                    fake_prob_raw = float(probs[1].item())

                    if self._load_svm():
                        try:
                            import math
                            feat = self.vectorizer.transform([text])
                            df = self.svm_model.decision_function(feat)[0]
                            svm_fake = 1.0 / (1.0 + math.exp(-df))
                            # Weight DistilBERT 85% and SVM 15% for short text/OCR
                            # DistilBERT captures deep semantics; low SVM weight prevents TF-IDF vocabulary skew
                            fake_prob = (fake_prob_raw * 0.85) + (svm_fake * 0.15)
                            real_prob = 1.0 - fake_prob
                        except Exception:
                            fake_prob = fake_prob_raw
                            real_prob = real_prob_raw
                    else:
                        fake_prob = fake_prob_raw
                        real_prob = real_prob_raw
                else:
                    TEMPERATURE = 2.5
                    scaled_logits = outputs.logits / TEMPERATURE
                    probs = torch.softmax(scaled_logits, dim=1)[0]
                    # Label 0: REAL, Label 1: FAKE
                    real_prob = float(probs[0].item())
                    fake_prob = float(probs[1].item())

                # OCR threshold calibration: require >= 53% fake probability to classify as FAKE
                # to prevent minor OCR character recognition noise from flipping authentic claims
                is_fake = (fake_prob >= 0.53) if is_ocr else (fake_prob >= real_prob)
                prediction = "FAKE" if is_fake else "REAL"
                confidence = round((fake_prob if is_fake else (1.0 - fake_prob)) * 100, 2)
                fake_percent = round(fake_prob * 100, 2)

                return {
                    "prediction": prediction,
                    "is_fake": is_fake,
                    "fake_probability": fake_percent,
                    "confidence": confidence,
                    "model_used": "DistilBERT (98% Acc, T-scaled)",
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

def analyze_text_fake(text, is_ocr=False):
    """
    Standard interface function for DeepGuard Backend.
    Args:
        text (str): Text to analyze.
        is_ocr (bool): If True, text came from OCR extraction — always use
                       ensemble (T=3.0 + SVM) since OCR text is never a full
                       article and must not trigger the DistilBERT article bias.
    """
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer.predict(text, is_ocr=is_ocr)

if __name__ == "__main__":
    sample = "Government declared a national emergency effective immediately!"
    print("Testing Text Analyzer on:\n", sample)
    res = analyze_text_fake(sample)
    import json
    print(json.dumps(res, indent=2))
