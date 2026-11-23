
import os
import sys
import torch
import timm

from PIL import Image
from torchvision import transforms


# ============================================================
# DEEPGUARD FINAL IMAGE INFERENCE MODULE
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

def _find_checkpoint(candidates):
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return str(p)
    return None

A_CHECKPOINT = _find_checkpoint([
    BASE_DIR / "v11a_convnext_tiny_best.pt",
    BASE_DIR / "V11A_ArtiFact_Defactify" / "v11a_convnext_tiny_best.pt",
])

B_CHECKPOINT = _find_checkpoint([
    BASE_DIR / "v11b_convnext_tiny_last.pt",
    BASE_DIR / "v11b_convnext_tiny_best.pt",
    BASE_DIR / "V11B_Quality_Augmentation" / "v11b_convnext_tiny_best.pt",
    BASE_DIR / "V11B_Quality_Augmentation" / "v11b_convnext_tiny_last.pt",
])

B2_CHECKPOINT = _find_checkpoint([
    BASE_DIR / "v11b2_convnext_tiny_best.pt",
    BASE_DIR / "V11B2_Controlled_Augmentation" / "v11b2_convnext_tiny_best.pt",
])

# Make the local C2PA module importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deepguard_c2pa import analyze_c2pa


# ============================================================
# CONFIGURATION
# ============================================================

VISUAL_THRESHOLD = 0.510

WEIGHT_A = 0.60
WEIGHT_B = 0.20
WEIGHT_B2 = 0.20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=1
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


class DeepGuardImageDetector:

    def __init__(self):

        print("Loading DeepGuard image models...")
        print("Device:", DEVICE)

        self.models = {}
        self.weights = {}

        if A_CHECKPOINT:
            try:
                self.models["v11a"] = load_model(A_CHECKPOINT)
                self.weights["v11a"] = WEIGHT_A
                print(f"V11-A loaded from: {A_CHECKPOINT}")
            except Exception as e:
                print(f"Failed to load V11-A: {e}")

        if B_CHECKPOINT:
            try:
                self.models["v11b"] = load_model(B_CHECKPOINT)
                self.weights["v11b"] = WEIGHT_B
                print(f"V11-B loaded from: {B_CHECKPOINT}")
            except Exception as e:
                print(f"Failed to load V11-B: {e}")

        if B2_CHECKPOINT:
            try:
                self.models["v11b2"] = load_model(B2_CHECKPOINT)
                self.weights["v11b2"] = WEIGHT_B2
                print(f"V11-B2 loaded from: {B2_CHECKPOINT}")
            except Exception as e:
                print(f"Failed to load V11-B2: {e}")

        # Normalize weights across available models
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.normalized_weights = {k: v / total_w for k, v in self.weights.items()}
        else:
            self.normalized_weights = {}

        print(f"Active ensemble models: {list(self.models.keys())} with normalized weights: {self.normalized_weights}")
        print("DeepGuard image detector is ready.")

    @staticmethod
    def _to_bool(value):

        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in (
            "true", "1", "yes"
        )

    def _visual_probability(self, model, image_tensor):

        with torch.inference_mode():

            logits = model(image_tensor)
            probability = torch.sigmoid(logits.reshape(-1))[0]

        return float(probability.item())

    def predict(self, image_path):

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        # ----------------------------------------------------
        # 1. BACKEND PROVENANCE CHECK (C2PA)
        # ----------------------------------------------------

        try:
            provenance = analyze_c2pa(image_path)
        except Exception:
            # If provenance analysis fails, use visual models.
            provenance = {}

        evidence_valid = (
            self._to_bool(provenance.get("c2pa_present"))
            and self._to_bool(
                provenance.get("ai_generation_declared")
            )
            and self._to_bool(
                provenance.get("signature_validated")
            )
            and self._to_bool(
                provenance.get("data_hash_validated")
            )
        )

        # ----------------------------------------------------
        # 2. EARLY EVIDENCE-BASED DECISION
        # ----------------------------------------------------

        if evidence_valid:

            return {
                "prediction": "FAKE",
                "score": 100.0,
                "score_type": "c2pa_provenance_verified",
                "is_fake": True,
                "provenance": provenance
            }

        # ----------------------------------------------------
        # 3. VISUAL MODEL FALLBACK
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        model_probabilities = {}
        for name, model in self.models.items():
            model_probabilities[name] = self._visual_probability(model, image_tensor)

        # ----------------------------------------------------
        # 4. ENSEMBLE COMBINATION
        # ----------------------------------------------------

        if self.normalized_weights:
            ensemble_probability = sum(
                self.normalized_weights[name] * model_probabilities[name]
                for name in self.models
            )
        elif model_probabilities:
            ensemble_probability = sum(model_probabilities.values()) / len(model_probabilities)
        else:
            ensemble_probability = 0.50

        prediction = (
            "FAKE"
            if ensemble_probability >= VISUAL_THRESHOLD
            else "REAL"
        )

        return {
            "prediction": prediction,
            "score": ensemble_probability,
            "score_type": "fake_probability",
            "is_fake": ensemble_probability >= VISUAL_THRESHOLD,
            "model_probabilities": model_probabilities,
            "ensemble_weights": self.normalized_weights,
            "provenance": provenance
        }
