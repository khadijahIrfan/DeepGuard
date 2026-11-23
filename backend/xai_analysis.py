import re
import torch
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification
)

# =====================================
# Load DistilBERT Model
# =====================================

MODEL_PATH = r"E:\DeepGuard\models\final_distilbert_2026\FINAL_MODEL"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# =====================================
# Prediction Function
# =====================================

def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    texts = [str(text) for text in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    return probabilities.detach().cpu().numpy()

# =====================================
# High-Speed Explainability (LOO Occlusion)
# =====================================

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'is', 'are', 'was', 'were', 'it', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'as', 'from', 'at', 'my', 'your'
}

def get_important_words(text, top_k=5):
    """
    Ultra-fast Leave-One-Out (Occlusion Sensitivity) explainability.
    Evaluates word contributions via a single batched DistilBERT pass (<200ms latency),
    providing accurate feature attribution without the 40-second CPU delay of SHAP.
    """
    if not text or not str(text).strip():
        return []

    text_clean = str(text).strip()

    # Get baseline prediction
    base_probs = predict_proba([text_clean])[0]
    pred_class = int(base_probs.argmax())
    base_score = float(base_probs[pred_class])
    target_label = "FAKE" if pred_class == 1 else "REAL"

    # Tokenize words while preserving case for matching
    raw_words = re.findall(r'\b[A-Za-z0-9\'-]{2,}\b', text_clean)
    if not raw_words:
        return []

    # Filter unique meaningful words
    unique_words = []
    seen = set()
    for w in raw_words:
        w_lower = w.lower()
        if w_lower not in seen and len(w) > 1:
            seen.add(w_lower)
            unique_words.append(w)

    # Prioritize substantive words
    content_words = [w for w in unique_words if w.lower() not in STOP_WORDS]
    eval_words = content_words[:25] if len(content_words) >= 5 else unique_words[:25]

    if not eval_words:
        return []

    # Build batched occlusion variations
    occluded_texts = []
    for w in eval_words:
        pattern = r'\b' + re.escape(w) + r'\b'
        masked = re.sub(pattern, '', text_clean, count=1).strip()
        occluded_texts.append(masked if masked else "[EMPTY]")

    # Single batched forward pass
    batch_probs = predict_proba(occluded_texts)

    important_words = []
    for w, occl_prob in zip(eval_words, batch_probs):
        occl_score = float(occl_prob[pred_class])
        # Delta: how much confidence dropped when word was removed
        delta = base_score - occl_score
        
        # If removing w dropped confidence in target class (delta >= 0), w supports it
        direction = f"Supports {target_label}" if delta >= 0 else f"Opposes {target_label}"
        
        important_words.append({
            "word": w,
            "contribution": round(delta, 4),
            "direction": direction
        })

    # Sort by absolute magnitude of contribution
    important_words.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return important_words[:top_k]

def explain_prediction(text):
    """
    Fallback explanation if SHAP object interface is requested.
    """
    return get_important_words(text, top_k=5)
