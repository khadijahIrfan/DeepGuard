import logging

logger = logging.getLogger(__name__)

# Standard weights for Multimodal / Text-only / Image-only modes
BASE_WEIGHTS = {
    "text_score":              0.30,
    "image_score":             0.25,
    "context_mismatch_score":  0.25,
    "viral_score":             0.10,
    "sentiment_score":         0.05,
    "metadata_score":          0.05,
}

# Grounded Infographic mode weights (OCR text extracted from image).
# C2PA handles AI-generation detection at 100% before the visual model even runs.
# For real news cards, ConvNeXt can produce false positives on dense scenes
# (jerseys with numbers, protest crowds, low-light sports photography).
# Text + context alignment are the primary signals for grounded news card analysis.
GROUNDED_WEIGHTS = {
    "text_score":              0.35,   # text claims are primary for news cards
    "image_score":             0.20,   # C2PA handles AI gen; ConvNeXt is secondary
    "context_mismatch_score":  0.30,   # CLIP cross-modal contradiction still key
    "viral_score":             0.12,
    "sentiment_score":         0.03,
    "metadata_score":          0.00,   # no EXIF on infographics (re-shared screenshots)
}

class FusionEngine:
    """
    DeepGuard Decision Fusion Engine.
    Combines individual module probabilities into a unified confidence score,
    risk level assessment, and final FAKE / REAL verdict.
    Uses continuous dynamic weighting without static clamps.

    Modes:
      - "standard"  : default multimodal / text-only / image-only
      - "grounded"  : OCR grounded infographic — image_score weight boosted
    """
    def __init__(self, weights=None):
        self.weights = weights  # None = auto-select based on mode

    def calculate_verdict(self, module_results, mode="standard"):
        """
        Calculates final combined score based on provided module outputs.
        module_results is a dict containing:
          - text_score (0-100)
          - image_score (0-100, optional)
          - sentiment_score (0-100)
          - viral_score (0-100)
          - metadata_score (0-100, optional)
          - context_mismatch_score (0-100, optional)
          - image_engine (str, optional): "c2pa_verified" | "convnext_ensemble" | ...
        """
        active_scores = {}
        active_weights = {}

        # Identify whether the image result came from C2PA (cryptographic, definitive)
        # or from the visual CNN (can false-positive on dense/complex photographic scenes)
        image_engine = module_results.get("image_engine", "")
        image_is_c2pa = (image_engine == "c2pa_verified")

        # Select weight table based on analysis mode
        weight_table = self.weights if self.weights else (
            GROUNDED_WEIGHTS if mode == "grounded" else BASE_WEIGHTS
        )

        for key, weight in weight_table.items():
            if key in module_results and module_results[key] is not None:
                try:
                    score = float(module_results[key])
                    # Clamp score 0-100
                    active_scores[key] = max(0.0, min(100.0, score))
                    active_weights[key] = weight
                except (ValueError, TypeError):
                    continue

        if not active_weights:
            return {
                "verdict": "UNKNOWN",
                "final_score": 0.0,
                "confidence": 0.0,
                "risk_level": "UNKNOWN",
                "explanation": "No valid module scores provided."
            }

        # Normalize weights so they sum to 1.0 even if some modules (like image) are missing
        total_weight_sum = sum(active_weights.values())
        normalized_weights = {k: v / total_weight_sum for k, v in active_weights.items()}

        # Continuous weighted sum calculation
        final_score = sum(normalized_weights[k] * active_scores[k] for k in active_scores)

        # In Text-Only mode: viral_score is an additive behavioral penalty, NOT a discount.
        # Absence of viral clickbait phrases must never dilute or flip a fake text prediction to real.
        if "text_score" in active_scores and "image_score" not in active_scores:
            base_text = active_scores["text_score"]
            v_score = active_scores.get("viral_score", 0.0)
            if v_score > 0:
                final_score = (base_text * 0.85) + (v_score * 0.15)
            else:
                final_score = base_text

        # Dynamic Multi-Modal Deception Calibration:
        # In multimodal misinformation (e.g. false text caption placed next to a genuine photo,
        # or out-of-context mismatch between text and image), authentic visual features must not
        # suppress high-confidence textual or cross-modal deception.
        # DeepGuard smoothly elevates the score proportionally to the highest verified deceit signal.
        text_prob = active_scores.get("text_score", 0.0)
        img_prob  = active_scores.get("image_score", 0.0)
        mismatch  = active_scores.get("context_mismatch_score", 0.0)

        critical_signals = []
        # Any specialized module indicating deception (>50% probability) triggers elevation
        if text_prob >= 50.0:
            critical_signals.append(text_prob)
        # Image model is a critical signal ONLY when:
        #   (a) C2PA cryptographically proved AI-generation, OR
        #   (b) Not in grounded mode (standard / text-with-image multimodal)
        # In grounded mode without C2PA, the ConvNeXt visual model can false-positive on
        # genuine news photographs (sports jerseys, protest crowds, smog, low-light scenes).
        image_is_critical = image_is_c2pa or (mode != "grounded")
        if img_prob >= 50.0 and image_is_critical:
            critical_signals.append(img_prob)
        if mismatch >= 50.0:
            critical_signals.append(mismatch)

        if critical_signals:
            max_signal = max(critical_signals)

            # Grounded & Multimodal counter-evidence calibration:
            # If ONLY the text module shows a borderline score (50.0 <= text_prob < 75.0)
            # due to short headline length, while the image is clearly authentic (< 25.0),
            # context is semantically aligned (< 45.0), and no viral manipulation patterns exist (< 20.0):
            # The authentic image and contextual alignment prevent false alarms on genuine news cards.
            image_clearly_real = img_prob < 25.0 and "image_score" in active_scores
            no_mismatch = mismatch < 45.0 and "context_mismatch_score" in active_scores
            viral_prob = active_scores.get("viral_score", 0.0)
            no_viral_manipulation = viral_prob < 20.0
            borderline_text_only = (len(critical_signals) == 1 and text_prob < 75.0 and "text_score" in active_scores)

            if borderline_text_only and image_clearly_real and no_mismatch and no_viral_manipulation:
                # Keep the balanced weighted score (the weighted blend already accounts for all signals)
                pass
            else:
                # Confirmed deception: C2PA-verified AI image, cross-modal contradiction,
                # or high-confidence fake text (>= 65%). Elevate to reflect the verified deception.
                elevated_score = (final_score * 0.30) + (max_signal * 0.70)
                final_score = max(final_score, elevated_score)
                if max_signal >= 50.0 and final_score < 50.0:
                    final_score = max_signal

        final_score = round(final_score, 2)

        # Verdict Determination: 50.0 is the calibrated decision boundary
        is_fake = final_score >= 50.0
        verdict = "FAKE" if is_fake else "REAL"

        # Continuous Confidence Score (distance from boundary scaled 50-100%)
        confidence = round(50.0 + abs(final_score - 50.0), 2)

        # Dynamic Risk Level Assessment
        if final_score >= 75.0:
            risk_level = "CRITICAL"
            risk_description = "High probability of deceptive misinformation or synthetic manipulation."
        elif final_score >= 50.0:
            risk_level = "HIGH"
            risk_description = "Content flagged as deceptive by specialized AI modules."
        elif final_score >= 35.0:
            risk_level = "MODERATE"
            risk_description = "Contains sensational or disputed patterns; verify with primary sources."
        else:
            risk_level = "LOW"
            risk_description = "Content exhibits authentic linguistic, visual, and contextual patterns."

        # 6. Cross-Modal Mismatch & Temporal Inconsistency Handling
        # If cross-modal contradiction or temporal mismatch is detected (e.g. an authentic photo
        # reused with an unrelated narrative or "happening today"), alert the user that the media
        # is out-of-context, without crudely classifying the photo itself as synthetic fake.
        temporal_info = module_results.get("temporal_verification")
        overall_assessment = "NO_DETECTED_INCONSISTENCY"
        temporal_warning = None

        if mismatch >= 50.0:
            overall_assessment = "CROSS_MODAL_CONTRADICTION"
            if risk_level == "LOW":
                risk_level = "HIGH"
                risk_description = "Cross-modal contradiction detected: authentic media paired with an unrelated news narrative."

        if temporal_info and isinstance(temporal_info, dict):
            t_status = temporal_info.get("status")
            if t_status == "POSSIBLE_TEMPORAL_MISMATCH":
                overall_assessment = "POTENTIALLY_MISLEADING_CONTEXT"
                temporal_warning = "Image is authentic but temporally outdated for the current event claim."
                if risk_level in ["LOW", "MODERATE"]:
                    risk_level = "MODERATE"
                    risk_description = "The image appears authentic but evidence indicates it originated from a previous event/timeframe."
            elif t_status == "CONTRADICTORY_DATE_EVIDENCE":
                overall_assessment = "POTENTIALLY_INCONSISTENT"
                temporal_warning = "Discrepancy detected between claimed event date and image records."
            elif t_status == "TEMPORAL_STATUS_UNVERIFIED":
                if mismatch < 50.0:
                    overall_assessment = "INSUFFICIENT_TEMPORAL_EVIDENCE"
            elif t_status == "TEMPORALLY_CONSISTENT":
                if mismatch < 50.0:
                    overall_assessment = "TEMPORALLY_CONSISTENT"

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "final_score": final_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_description": risk_description,
            "overall_multimodal_assessment": overall_assessment,
            "temporal_warning": temporal_warning,
            "active_modules": list(active_scores.keys()),
            "module_breakdown": {
                k: {
                    "raw_score": round(active_scores[k], 2),
                    "weight_percentage": round(normalized_weights[k] * 100, 1)
                } for k in active_scores
            }
        }

_fusion_instance = None

def fuse_predictions(module_results, mode="standard"):
    """
    Standard interface function for DeepGuard Flask Backend.
    
    Args:
        module_results: dict of scored module outputs
        mode: "standard" (default) or "grounded" (OCR infographic mode,
              boosts image_score weight to better handle AI image + real text scenario)
    """
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = FusionEngine()
    return _fusion_instance.calculate_verdict(module_results, mode=mode)


if __name__ == "__main__":
    sample_inputs = {
        "text_score": 85.0,
        "image_score": 6.5,
        "sentiment_score": 70.0,
        "viral_score": 80.0,
        "metadata_score": 10.0,
        "context_mismatch_score": 65.0
    }
    print("Testing Fusion Engine...")
    res = fuse_predictions(sample_inputs)
    import json
    print(json.dumps(res, indent=2))
