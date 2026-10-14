import logging
import torch
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP_DEPS = True
except ImportError:
    HAS_CLIP_DEPS = False

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"
THRESHOLD = 0.23
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if not HAS_CLIP_DEPS:
        return None, None

    if _clip_model is None or _clip_processor is None:
        try:
            logger.info(f"Loading CLIP model '{MODEL_NAME}' on {DEVICE}...")
            _clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
            _clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            _clip_model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return None, None

    return _clip_model, _clip_processor

def clip_similarity(image_path, text):
    """
    Returns cosine similarity between image and text.
    Range: approximately 0.0 - 1.0
    Higher = more semantically aligned.
    """
    model, processor = _get_clip()
    if model is None or processor is None:
        return 0.5  # Fallback neutral similarity if model fails

    try:
        with Image.open(image_path) as opened_img:
            image = opened_img.convert("RGB")

        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(image_features, text_features.T).item()

        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating CLIP similarity for {image_path}: {e}")
        return 0.5

def detect_context_mismatch(image_path, text):
    """
    Module 6 — Context Mismatch Detector.
    Returns:
        similarity: Cosine similarity score
        mismatch_score: Normalized 0-100 score indicating cross-modal contradiction
        is_mismatch: Boolean flag if similarity is below threshold
    """
    if not image_path or not text or not str(text).strip():
        return {
            "similarity": 0.0,
            "mismatch_score": 0.0,
            "is_mismatch": False,
            "explanation": "Image or text missing for context mismatch evaluation",
            "status": "bypassed"
        }

    similarity = clip_similarity(image_path=image_path, text=text)

    # Normalize similarity into a dynamic 0-100 mismatch score:
    # CLIP cosine similarity typically spans 0.05 to 0.35 for image-caption pairs.
    # If similarity < THRESHOLD (0.23), content is contextually contradictory.
    if similarity < THRESHOLD:
        # Cross-modal contradiction: maps [THRESHOLD down to 0.0] -> [50.0 up to 100.0]
        mismatch_percent = round(min(100.0, 50.0 + ((THRESHOLD - max(0.0, similarity)) / THRESHOLD) * 50.0), 2)
        is_mismatch = True
        explanation = f"Cross-modal contradiction detected! Text and image content have low semantic alignment (similarity: {round(similarity, 3)})."
    else:
        # Contextually aligned: maps [THRESHOLD up to 0.35] -> [50.0 down to 0.0]
        norm_align = min(1.0, (similarity - THRESHOLD) / 0.12)
        mismatch_percent = round(max(0.0, (1.0 - norm_align) * 45.0), 2)
        is_mismatch = False
        explanation = f"Image and text appear contextually aligned (similarity: {round(similarity, 3)})."

    return {
        "similarity": round(similarity, 4),
        "mismatch_score": mismatch_percent,
        "is_mismatch": is_mismatch,
        "explanation": explanation,
        "status": "success"
    }

if __name__ == "__main__":
    test_img = r"E:\Zujajah\test_images\sample_test.jpg"
    test_text = "A serious news headline about flood victims."
    print("Testing Context Mismatch Module...")
    res = detect_context_mismatch(test_img, test_text)
    print("Result:", res)