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

logger = logging.getLogger(__name__)

class DeepGuardPipeline:
    """
    Unified Pipeline Coordinator for DeepGuard.
    Orchestrates execution across all 7 AI modules, records performance metrics,
    and returns a structured, defense-ready response.
    """
    def __init__(self):
        pass

    def analyze(self, text=None, image_path=None, user_id=None):
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

        # 1. Text Analysis Module
        if has_text:
            text_res = analyze_text_fake(text)
            module_outputs["text_analysis"] = text_res
            fusion_inputs["text_score"] = text_res.get("fake_probability", 50.0)

        # 2. Image Detection Module
        if has_image:
            image_res = predict_image_authenticity(image_path)
            module_outputs["image_detection"] = image_res
            fusion_inputs["image_score"] = image_res.get("manipulation_score", 0.0)

        # 3. Viral Pattern Detection Module
        if has_text:
            viral_res = detect_viral_patterns(text)
            module_outputs["viral_pattern"] = viral_res
            fusion_inputs["viral_score"] = viral_res.get("viral_score", 0.0)

        # 4. Context Mismatch Module (Cross-modal)
        if has_text and has_image:
            mismatch_res = detect_context_mismatch(image_path, text)
            module_outputs["context_mismatch"] = mismatch_res
            fusion_inputs["context_mismatch_score"] = mismatch_res.get("mismatch_score", 0.0)

        # 5. Fusion Engine Execution
        final_verdict = fuse_predictions(fusion_inputs)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "verdict": final_verdict["verdict"],
            "is_fake": final_verdict["is_fake"],
            "final_score": final_verdict["final_score"],
            "confidence": final_verdict["confidence"],
            "risk_level": final_verdict["risk_level"],
            "risk_description": final_verdict["risk_description"],
            "modules_executed": list(module_outputs.keys()),
            "module_details": module_outputs,
            "fusion_breakdown": final_verdict["module_breakdown"],
            "performance": {
                "execution_time_ms": elapsed_ms,
                "device": "CPU / GPU Auto-accelerated"
            },
            "status": "success"
        }

_pipeline_instance = None

def run_deepguard_analysis(text=None, image_path=None, user_id=None):
    """
    Main entry point for Flask backend route /api/analyze
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DeepGuardPipeline()
    return _pipeline_instance.analyze(text=text, image_path=image_path, user_id=user_id)


if __name__ == "__main__":
    sample_txt = "BREAKING: Govt declared emergency! Forward to 10 groups immediately. Abhi share karo!"
    sample_img = r"E:\Zujajah\test_images\sample_test.jpg"
    print("=== TESTING UNIFIED DEEPGUARD PIPELINE ===")
    res = run_deepguard_analysis(text=sample_txt, image_path=sample_img)
    import json
    print(json.dumps(res, indent=2))
