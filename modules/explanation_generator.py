import logging

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """
    Human-Readable Explanation Generator for DeepGuard.
    Translates complex multi-modal AI model probabilities, XAI highlights,
    and cross-modal mismatch scores into clear, plain-English media literacy explanations.
    """

    def generate_explanation(self, pipeline_result):
        if not pipeline_result or pipeline_result.get("status") != "success":
            return {
                "summary": "Analysis could not be completed.",
                "key_reasons": ["Insufficient input data or processing error."],
                "plain_text": "Unable to verify content due to invalid inputs."
            }

        verdict = pipeline_result.get("verdict", "UNKNOWN")
        is_fake = pipeline_result.get("is_fake", False)
        confidence = pipeline_result.get("confidence", 0.0)
        risk_level = pipeline_result.get("risk_level", "LOW")
        details = pipeline_result.get("module_details", {})

        reasons = []

        # 1. Text Analysis Explanation
        text_mod = details.get("text_analysis")
        if text_mod and text_mod.get("status") == "success":
            if text_mod.get("is_fake"):
                fake_prob = text_mod.get("fake_probability", 0.0)
                reasons.append(
                    f"📰 **Linguistic Analysis ({fake_prob}% Fake)**: The writing style exhibits linguistic patterns commonly found in fabricated news or sensationalized headlines."
                )
            else:
                reasons.append(
                    "📰 **Linguistic Analysis (Authentic)**: The text vocabulary and structure match patterns of credible news reporting."
                )

        # 2. Viral Pattern Explanation
        viral_mod = details.get("viral_pattern")
        if viral_mod and viral_mod.get("status") == "success":
            patterns = viral_mod.get("patterns_detected", [])
            if patterns:
                phrases = [f"'{p['matched_phrase']}'" for p in patterns[:3]]
                reasons.append(
                    f"🚨 **Viral Manipulation Triggers**: Detected pressure phrases ({', '.join(phrases)}) designed to trick users into forwarding claims without verification."
                )

        # 3. Context Mismatch Explanation (Secret Weapon)
        mismatch_mod = details.get("context_mismatch")
        if mismatch_mod and mismatch_mod.get("status") == "success":
            if mismatch_mod.get("is_mismatch"):
                sim = mismatch_mod.get("similarity", 0.0)
                reasons.append(
                    f"⚠️ **Image & Text Contradiction**: The uploaded image does not contextually match the text story (Semantic Similarity: {round(sim * 100, 1)}%). This indicates a real image paired with a false caption."
                )

        # 4. Image Detection Explanation
        image_mod = details.get("image_detection")
        if image_mod and image_mod.get("status") == "success":
            if image_mod.get("is_fake"):
                score = image_mod.get("manipulation_score", 0.0)
                reasons.append(
                    f"🖼️ **Digital Image Editing ({score}% Manipulated)**: Neural vision analysis detected pixel distortion or synthetic AI generation artifacts."
                )
            else:
                reasons.append(
                    "🖼️ **Image Integrity**: No digital forgery or deepfake manipulation patterns were found in the photo structure."
                )

        # Summary sentence for normal people
        if is_fake:
            summary_sentence = (
                f"DeepGuard flagged this post as **FAKE** with **{confidence}% confidence** ({risk_level} Risk). "
                f"Multiple specialized AI modules detected deceptive tactics."
            )
        else:
            summary_sentence = (
                f"DeepGuard verified this content as **REAL** with **{confidence}% confidence** ({risk_level} Risk). "
                f"The post text and media appear authentic and aligned."
            )

        if not reasons:
            reasons.append("No suspicious indicators detected across evaluated features.")

        return {
            "summary_sentence": summary_sentence,
            "verdict_badge": verdict,
            "risk_level": risk_level,
            "reasons": reasons,
            "plain_text": f"{summary_sentence}\n\nKey Findings:\n- " + "\n- ".join(reasons)
        }

_explainer_instance = None

def generate_human_explanation(pipeline_result):
    """
    Standard interface function for DeepGuard Backend.
    """
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = ExplanationGenerator()
    return _explainer_instance.generate_explanation(pipeline_result)


if __name__ == "__main__":
    mock_pipeline = {
        "verdict": "FAKE",
        "is_fake": True,
        "confidence": 88.5,
        "risk_level": "HIGH",
        "module_details": {
            "text_analysis": {"is_fake": True, "fake_probability": 92.4, "status": "success"},
            "viral_pattern": {"patterns_detected": [{"matched_phrase": "Forward to 10 groups"}, {"matched_phrase": "Abhi share karo"}], "status": "success"},
            "context_mismatch": {"is_mismatch": True, "similarity": 0.14, "status": "success"},
            "image_detection": {"is_fake": False, "manipulation_score": 5.2, "status": "success"}
        },
        "status": "success"
    }
    print("Testing Plain Explanation Generator...")
    exp = generate_human_explanation(mock_pipeline)
    print("SUMMARY:", exp["summary_sentence"].encode("ascii", "ignore").decode())
    print("\nREASONS:")
    for r in exp["reasons"]:
        print("-", r.encode("ascii", "ignore").decode())

