import logging

logger = logging.getLogger(__name__)

# Base Weights specified in FYP SRS & Summary:
# Context Mismatch is DeepGuard's headline feature (cross-modal deception detection)
BASE_WEIGHTS = {
    "text_score": 0.30,
    "image_score": 0.25,
    "context_mismatch_score": 0.25,
    "viral_score": 0.10,
    "sentiment_score": 0.05,
    "metadata_score": 0.05
}

class FusionEngine:
    """
    DeepGuard Decision Fusion Engine.
    Combines individual module probabilities into a unified confidence score,
    risk level assessment, and final FAKE / REAL verdict.
    """
    def __init__(self, weights=None):
        self.weights = weights or BASE_WEIGHTS

    def calculate_verdict(self, module_results):
        """
        Calculates final combined score based on provided module outputs.
        module_results is a dict containing:
          - text_score (0-100)
          - image_score (0-100, optional)
          - sentiment_score (0-100)
          - viral_score (0-100)
          - metadata_score (0-100, optional)
          - context_mismatch_score (0-100, optional)
        """
        active_scores = {}
        active_weights = {}

        for key, weight in self.weights.items():
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

        # Weighted sum calculation
        final_score = sum(normalized_weights[k] * active_scores[k] for k in active_scores)
        
        # Context-Aware Cross-Modal Decision Rule:
        # If severe cross-modal mismatch is detected (mismatch_score >= 60.0), this represents
        # recycled/falsified media (out-of-context misinformation).
        if active_scores.get("context_mismatch_score", 0.0) >= 60.0:
            final_score = max(final_score, 52.0)

        final_score = round(final_score, 2)

        # Verdict Determination
        is_fake = final_score >= 50.0
        verdict = "FAKE" if is_fake else "REAL"

        # Confidence Calculation
        # Distance from 50% decision boundary mapped to 50-100% confidence
        confidence = round(50.0 + abs(final_score - 50.0), 2)

        # Risk Level Assessment
        if final_score >= 75.0:
            risk_level = "CRITICAL"
            risk_description = "High probability of coordinated misinformation or synthetic manipulation."
        elif final_score >= 50.0:
            risk_level = "HIGH"
            risk_description = "Content flagged as deceptive/fake by multiple specialized AI modules."
        elif final_score >= 30.0:
            risk_level = "MODERATE"
            risk_description = "Contains suspicious language or sensational patterns; verify source."
        else:
            risk_level = "LOW"
            risk_description = "Content exhibits authentic linguistic and structural patterns."

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "final_score": final_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_description": risk_description,
            "active_modules": list(active_scores.keys()),
            "module_breakdown": {
                k: {
                    "raw_score": active_scores[k],
                    "weight_percentage": round(normalized_weights[k] * 100, 1)
                } for k in active_scores
            }
        }

_fusion_instance = None

def fuse_predictions(module_results):
    """
    Standard interface function for DeepGuard Flask Backend.
    """
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = FusionEngine()
    return _fusion_instance.calculate_verdict(module_results)


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
