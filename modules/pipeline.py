import time
import logging
import pathlib
import sys

# Ensure root project path is in sys.path
MODULES_DIR = pathlib.Path(__file__).parent.resolve()
BASE_DIR = MODULES_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.text_analysis import analyze_text_fake
from modules.image_detection import predict_image_authenticity
from modules.viral_pattern import detect_viral_patterns
from modules.context_mismatch import detect_context_mismatch
from modules.fusion_engine import fuse_predictions
from modules.ocr_extractor import extract_grounded_text, extract_visual_roi
from modules.temporal_verification import TemporalConsistencyEngine

logger = logging.getLogger(__name__)

class DeepGuardPipeline:
    """
    Unified Pipeline Coordinator for DeepGuard.
    Orchestrates execution across all AI modules, records performance metrics,
    and returns a structured, defense-ready response.
    Supports all 4 analysis modes:
      1. Text Only Analysis (DistilBERT + Viral + VADER)
      2. Image Only Authenticity (EfficientNetV2-S + Forensic)
      3. Multimodal Analysis (Independent Text + Image + CLIP Mismatch)
      4. Grounded Text / Infographic Analysis (EasyOCR + DistilBERT + Visual ROI + CLIP)
    """
    def __init__(self, provenance_provider=None):
        self.temporal_engine = TemporalConsistencyEngine(provenance_provider=provenance_provider)

    def analyze(self, text=None, image_path=None, is_grounded=False, user_id=None):
        start_time = time.time()
        
        has_text = bool(text and isinstance(text, str) and text.strip())
        has_image = bool(image_path and pathlib.Path(image_path).is_file())

        if not has_text and not has_image:
            return {
                "error": "At least post text or an image file must be provided.",
                "status": "invalid_request"
            }

        text = text.strip() if has_text else ""
        module_outputs = {}
        fusion_inputs = {}
        eval_image_path = image_path
        _text_came_from_ocr = False  # Track if text was OCR-extracted (not user-typed)

        # 0. Grounded Text / Infographic OCR Extraction:
        # Activated when is_grounded=True or automatically when an image is provided
        # without accompanying text (e.g. social media news card / banner / meme with embedded claims)
        if has_image and (is_grounded or not has_text):
            try:
                ocr_result = extract_grounded_text(image_path)
                if ocr_result.get("has_grounded_text"):
                    module_outputs["grounded_text_ocr"] = ocr_result
                    # If user did not provide manual text, adopt the OCR grounded text
                    if not has_text:
                        text = ocr_result["cleaned_text"]
                        has_text = True
                        _text_came_from_ocr = True  # Mark as OCR-extracted
                    else:
                        # User provided text + grounded mode: also flag OCR-origin
                        # so ensemble applies (the image has OCR text grounded on it)
                        _text_came_from_ocr = True
                        # If user typed headline without agency attribution, but OCR card has full attribution:
                        ocr_txt = ocr_result.get("cleaned_text", "")
                        if ocr_txt and len(ocr_txt) > len(text):
                            first_words = [w for w in text.lower().split()[:3] if len(w) > 2]
                            if any(w in ocr_txt.lower() for w in first_words):
                                text = ocr_txt

                    # Extract visual ROI to isolate photo from sharp banner/text overlays
                    roi_path = extract_visual_roi(image_path, boxes=ocr_result.get("boxes"))
                    if roi_path and pathlib.Path(roi_path).is_file():
                        eval_image_path = roi_path
                else:
                    # OCR found no text, but is_grounded=True was explicitly requested:
                    # still mark text as OCR-calibrated so short user-provided headlines
                    # use the gentler T=3.5 + SVM ensemble (not full-article DistilBERT)
                    if is_grounded and has_text:
                        _text_came_from_ocr = True
            except Exception as e:
                logger.warning(f"OCR grounded text extraction failed or skipped: {e}")
                # Even on failure, honor the grounded flag for text calibration
                if is_grounded and has_text:
                    _text_came_from_ocr = True

        # 1. Text Analysis Module (DistilBERT)
        if has_text:
            # Pass is_ocr=True when text came from OCR extraction so the
            # ensemble (T=3.0 + SVM 30%) is used instead of pure DistilBERT.
            # OCR text from social cards is never a full article.
            text_res = analyze_text_fake(text, is_ocr=_text_came_from_ocr)
            module_outputs["text_analysis"] = text_res
            fusion_inputs["text_score"] = text_res.get("fake_probability", 50.0)

        # 2. Image Detection Module (Dual-Pass Authenticity Check)
        # Evaluates the primary photographic region and full image context.
        if has_image:
            is_grounded_mode = is_grounded or ("grounded_text_ocr" in module_outputs)
            image_res = predict_image_authenticity(eval_image_path, is_grounded=is_grounded_mode)
            if eval_image_path != image_path:
                full_image_res = predict_image_authenticity(image_path, is_grounded=is_grounded_mode)
                full_score = full_image_res.get("manipulation_score", 50.0)
                roi_score  = image_res.get("manipulation_score", 50.0)
                full_engine = full_image_res.get("engine", "")

                # C2PA-verified AI generation on the full image: definitive fake signal — keep it.
                if full_engine == "c2pa_verified":
                    image_res = full_image_res

                # Full image is authentic (< 50% fake): the ROI high score is a crop artefact.
                # Always trust the full-image authenticity reading for news card photographs.
                elif full_score < 50.0:
                    image_res = full_image_res

                # Both indicate manipulation — take the cleaner (lower) of the two readings
                # to avoid compounding any artefact introduced by the crop boundary.
                else:
                    image_res = full_image_res if full_score <= roi_score else image_res
            module_outputs["image_detection"] = image_res
            fusion_inputs["image_score"] = image_res.get("manipulation_score", 0.0)
            # Pass the image engine type so the fusion engine can distinguish
            # cryptographic C2PA evidence from visual CNN estimates
            fusion_inputs["image_engine"] = image_res.get("engine", "")

        # 3. Viral Pattern Detection Module
        if has_text:
            viral_res = detect_viral_patterns(text)
            module_outputs["viral_pattern"] = viral_res
            fusion_inputs["viral_score"] = viral_res.get("viral_score", 0.0)

        # 4. Context Mismatch Module (Cross-modal CLIP Visual-Semantic Alignment)
        if has_text and has_image:
            mismatch_res = detect_context_mismatch(image_path, text)
            module_outputs["context_mismatch"] = mismatch_res
            fusion_inputs["context_mismatch_score"] = mismatch_res.get("mismatch_score", 0.0)

        # 4b. Temporal Consistency & Image Provenance Verification Module
        # Evaluates chronological consistency between text event timelines and image history/EXIF
        if has_text and has_image:
            meta_info = None
            if "image_detection" in module_outputs:
                meta_info = module_outputs["image_detection"].get("metadata_forensics", {}).get("details")
            clip_res = module_outputs.get("context_mismatch")
            temporal_res = self.temporal_engine.evaluate(
                text=text,
                image_path=image_path,
                metadata_info=meta_info,
                clip_result=clip_res
            )
            module_outputs["temporal_verification"] = temporal_res
            fusion_inputs["temporal_verification"] = temporal_res

        # 5. Continuous Fusion Engine Execution (No static 72% clamping)
        # Pass grounded mode to fusion engine when OCR was active — this boosts
        # image_score weight to correctly handle AI-generated image + real-text scenario
        fusion_mode = "grounded" if "grounded_text_ocr" in module_outputs else "standard"
        final_verdict = fuse_predictions(fusion_inputs, mode=fusion_mode)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "prediction": final_verdict["verdict"],  # Alias for compatibility
            "verdict": final_verdict["verdict"],
            "is_fake": final_verdict["is_fake"],
            "final_score": final_verdict["final_score"],
            "confidence": final_verdict["confidence"],
            "risk_level": final_verdict["risk_level"],
            "risk_description": final_verdict["risk_description"],
            "overall_multimodal_assessment": final_verdict.get("overall_multimodal_assessment", "NO_DETECTED_INCONSISTENCY"),
            "temporal_warning": final_verdict.get("temporal_warning"),
            "temporal_verification": module_outputs.get("temporal_verification"),
            "modules_executed": list(module_outputs.keys()),
            "module_details": module_outputs,
            "fusion_breakdown": final_verdict["module_breakdown"],
            "analyzed_text": text if has_text else None,
            "is_grounded_text": "grounded_text_ocr" in module_outputs,
            "performance": {
                "execution_time_ms": elapsed_ms,
                "device": "CPU / GPU Auto-accelerated"
            },
            "grounded_text": module_outputs.get("grounded_text_ocr", {}).get("cleaned_text") if "grounded_text_ocr" in module_outputs else None,
            "status": "success"
        }

_pipeline_instance = None

def run_deepguard_analysis(text=None, image_path=None, is_grounded=False, user_id=None):
    """
    Main entry point for Flask backend route /api/analyze and /predict
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DeepGuardPipeline()
    return _pipeline_instance.analyze(text=text, image_path=image_path, is_grounded=is_grounded, user_id=user_id)


if __name__ == "__main__":
    sample_txt = "BREAKING: Govt declared emergency! Forward to 10 groups immediately. Abhi share karo!"
    sample_img = r"E:\DEEPGUARD\modules\context_mismatch\test_image.jpg"
    print("=== TESTING UNIFIED DEEPGUARD PIPELINE ===")
    res = run_deepguard_analysis(text=sample_txt, image_path=sample_img)
    import json
    print(json.dumps(res, indent=2))
