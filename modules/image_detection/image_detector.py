import os
import pathlib
import logging
import numpy as np

try:
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    import torch.nn as nn
    HAS_IMAGE_DEPS = True
except ImportError:
    HAS_IMAGE_DEPS = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# Import C2PA provenance analysis if available
try:
    from .deepguard_c2pa import analyze_c2pa
    HAS_C2PA = True
except Exception:
    try:
        import sys
        _cand_c2pa = pathlib.Path(__file__).parent.parent.parent / "image_related_neu"
        if str(_cand_c2pa) not in sys.path:
            sys.path.insert(0, str(_cand_c2pa))
        from deepguard_c2pa import analyze_c2pa
        HAS_C2PA = True
    except Exception:
        analyze_c2pa = None
        HAS_C2PA = False

logger = logging.getLogger(__name__)

# Base directories
MODULE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = MODULE_DIR.parent.parent
IMAGE_RELATED_NEU_DIR = PROJECT_ROOT / "image_related_neu"
LOCAL_MODELS_DIR = MODULE_DIR / "models"

# Checkpoint candidate locations
def _resolve_checkpoint(filenames, search_dirs=None):
    if search_dirs is None:
        search_dirs = [IMAGE_RELATED_NEU_DIR, LOCAL_MODELS_DIR]
    for d in search_dirs:
        for fn in filenames:
            p = d / fn
            if p.is_file():
                return p
    return None

A_CHECKPOINT = _resolve_checkpoint([
    "v11a_convnext_tiny_best.pt",
    "V11A_ArtiFact_Defactify/v11a_convnext_tiny_best.pt"
])

B_CHECKPOINT = _resolve_checkpoint([
    "v11b_convnext_tiny_last.pt",
    "v11b_convnext_tiny_best.pt",
    "V11B_Quality_Augmentation/v11b_convnext_tiny_best.pt",
    "V11B_Quality_Augmentation/v11b_convnext_tiny_last.pt"
])

B2_CHECKPOINT = _resolve_checkpoint([
    "v11b2_convnext_tiny_best.pt",
    "V11B2_Controlled_Augmentation/v11b2_convnext_tiny_best.pt"
])

# Legacy model paths for fallback
ONNX_MODEL_PATH = LOCAL_MODELS_DIR / "deepguard_image_model.onnx"
PT_MODEL_PATH = LOCAL_MODELS_DIR / "deepguard_image_model_best.pt"
HYBRID_PATH = LOCAL_MODELS_DIR / "best_hybrid_multimodal_falsified_news_model.pt"

# Configuration thresholds and weights
VISUAL_THRESHOLD = 0.510
WEIGHT_A = 0.60
WEIGHT_B = 0.20
WEIGHT_B2 = 0.20


class DeepGuardImageNet(nn.Module):
    """
    Legacy PyTorch Architecture for DeepGuard EfficientNetV2-S.
    Maintained for backward compatibility.
    """
    def __init__(self, backbone_name='tf_efficientnetv2_s.in21k_ft_in1k'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0) if HAS_TIMM else None
        self.head = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


class HybridMultimodalFalsifiedNewsClassifier(nn.Module):
    """
    Maintained for backward compatibility with hybrid multimodal classifier.
    """
    def __init__(self, embedding_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, text):
        image_features = self.image_projection(image)
        text_features = self.text_projection(text)
        difference = torch.abs(image_features - text_features)
        interaction = image_features * text_features
        similarity = torch.sum(image * text, dim=1, keepdim=True)
        fused = torch.cat([image_features, text_features, difference, interaction, similarity], dim=1)
        return self.classifier(fused).squeeze(1)


def _load_convnext_checkpoint(checkpoint_path, device="cpu"):
    if not checkpoint_path or not pathlib.Path(checkpoint_path).is_file():
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=1
    )

    sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


class ImageDetector:
    """
    DeepGuard SOTA Image Authenticity Detector.
    Integrates ConvNeXt-Tiny neural ensemble (V11-A, V11-B, V11-B2)
    with C2PA cryptographic provenance verification and EXIF forensics.
    """
    def __init__(self, model_path=None, device="cpu"):
        self.device = torch.device(device)
        self.backend_engine = None  # 'convnext_ensemble', 'onnx', or 'pytorch'
        
        # ConvNeXt models and normalized weights
        self.convnext_models = {}
        self.model_weights = {}
        self.normalized_weights = {}

        # Legacy fallback containers
        self.session = None
        self.pt_model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) if HAS_IMAGE_DEPS else None

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("true", "1", "yes")

    def _load_model(self):
        if self.backend_engine is not None:
            return True

        # Priority 1: ConvNeXt-Tiny Ensemble from image_related_neu
        if HAS_TIMM and HAS_IMAGE_DEPS:
            # Load V11-A
            if A_CHECKPOINT:
                try:
                    mA = _load_convnext_checkpoint(A_CHECKPOINT, device=self.device)
                    if mA is not None:
                        self.convnext_models["v11a"] = mA
                        self.model_weights["v11a"] = WEIGHT_A
                        logger.info(f"Loaded ConvNeXt-Tiny V11-A from {A_CHECKPOINT.name}")
                except Exception as e:
                    logger.warning(f"Could not load ConvNeXt V11-A: {e}")

            # Load V11-B
            if B_CHECKPOINT:
                try:
                    mB = _load_convnext_checkpoint(B_CHECKPOINT, device=self.device)
                    if mB is not None:
                        self.convnext_models["v11b"] = mB
                        self.model_weights["v11b"] = WEIGHT_B
                        logger.info(f"Loaded ConvNeXt-Tiny V11-B from {B_CHECKPOINT.name}")
                except Exception as e:
                    logger.warning(f"Could not load ConvNeXt V11-B: {e}")

            # Load V11-B2 if present
            if B2_CHECKPOINT:
                try:
                    mB2 = _load_convnext_checkpoint(B2_CHECKPOINT, device=self.device)
                    if mB2 is not None:
                        self.convnext_models["v11b2"] = mB2
                        self.model_weights["v11b2"] = WEIGHT_B2
                        logger.info(f"Loaded ConvNeXt-Tiny V11-B2 from {B2_CHECKPOINT.name}")
                except Exception as e:
                    logger.warning(f"Could not load ConvNeXt V11-B2: {e}")

            if self.convnext_models:
                tot_w = sum(self.model_weights.values())
                self.normalized_weights = {k: v / tot_w for k, v in self.model_weights.items()}
                self.backend_engine = 'convnext_ensemble'
                logger.info(f"ConvNeXt Ensemble initialized with models {list(self.convnext_models.keys())} and weights {self.normalized_weights}")
                return True

        # Priority 2: High-performance ONNX Engine (Legacy fallback)
        if HAS_ONNX and ONNX_MODEL_PATH.is_file():
            try:
                self.session = ort.InferenceSession(str(ONNX_MODEL_PATH))
                self.backend_engine = 'onnx'
                logger.info(f"Loaded DeepGuard Image Model via ONNX runtime: {ONNX_MODEL_PATH.name}")
                return True
            except Exception as e:
                logger.warning(f"ONNX initialization failed: {e}")

        # Priority 3: Native PyTorch EfficientNetV2-S Engine (Legacy fallback)
        if HAS_TIMM and PT_MODEL_PATH.is_file():
            try:
                checkpoint = torch.load(str(PT_MODEL_PATH), map_location=torch.device('cpu'))
                backbone_name = checkpoint.get('backbone', 'tf_efficientnetv2_s.in21k_ft_in1k')
                model = DeepGuardImageNet(backbone_name=backbone_name)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.pt_model = model
                self.backend_engine = 'pytorch'
                logger.info(f"Loaded DeepGuard Image Model via PyTorch: {PT_MODEL_PATH.name}")
                return True
            except Exception as e:
                logger.error(f"PyTorch model initialization failed: {e}")

        logger.error("No valid DeepGuard Image model could be initialized.")
        return False

    def predict(self, image_path, is_grounded=False):
        if not HAS_IMAGE_DEPS:
            return {
                "error": "Required image dependencies not installed",
                "prediction": "UNKNOWN",
                "manipulation_score": 0.0,
                "confidence": 0.0
            }

        if not self._load_model():
            return {
                "error": "Model could not be loaded",
                "prediction": "UNKNOWN",
                "manipulation_score": 0.0,
                "confidence": 0.0
            }

        img_path = pathlib.Path(image_path)
        if not img_path.is_file():
            return {
                "error": f"Image file not found: {image_path}",
                "prediction": "UNKNOWN",
                "manipulation_score": 0.0,
                "confidence": 0.0
            }

        # Check maximum file size: 10 MB
        MAX_SIZE_BYTES = 10 * 1024 * 1024
        if img_path.stat().st_size > MAX_SIZE_BYTES:
            size_mb = round(img_path.stat().st_size / (1024 * 1024), 2)
            return {
                "error": f"Image file size ({size_mb} MB) exceeds maximum allowed limit of 10 MB",
                "prediction": "UNKNOWN",
                "manipulation_score": 0.0,
                "confidence": 0.0,
                "status": "file_too_large"
            }

        # Integrated Forensic Metadata Extraction (EXIF)
        metadata_info = {"has_metadata": False, "details": {}}
        try:
            import sys
            backend_dir = str(MODULE_DIR.parent.parent / "backend")
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from metadata_analysis import extract_important_metadata
            meta = extract_important_metadata(str(img_path))
            metadata_info["has_metadata"] = meta.get("has_metadata", False)
            metadata_info["details"] = meta
        except Exception as meta_err:
            logger.debug(f"Metadata extraction skipped: {meta_err}")

        # Integrated C2PA Cryptographic Provenance Verification
        c2pa_result = {}
        c2pa_evidence_valid = False
        if analyze_c2pa is not None:
            try:
                c2pa_result = analyze_c2pa(str(img_path))
                c2pa_evidence_valid = (
                    self._to_bool(c2pa_result.get("c2pa_present"))
                    and self._to_bool(c2pa_result.get("ai_generation_declared"))
                    and self._to_bool(c2pa_result.get("signature_validated"))
                    and self._to_bool(c2pa_result.get("data_hash_validated"))
                )
            except Exception as c2pa_err:
                logger.debug(f"C2PA analysis skipped: {c2pa_err}")
                c2pa_result = {"error": str(c2pa_err)}

        # For standalone image analyses (such as Gemini generated fake images), C2PA cryptographically verifies AI generation:
        # For grounded news infographics where users test authentic news cards that were formatted/exported via AI tools,
        # visual neural forensics (ConvNeXt) evaluates the photographic authenticity of the scene.
        if c2pa_evidence_valid and not is_grounded:
            logger.info(f"C2PA Cryptographic Provenance verified AI-generation for {img_path.name}")
            return {
                "prediction": "FAKE",
                "is_fake": True,
                "manipulation_score": 100.0,
                "confidence": 100.0,
                "score": 100.0,
                "fake_probability": 1.0,
                "real_probability": 0.0,
                "metadata_forensics": metadata_info,
                "c2pa_provenance": c2pa_result,
                "model_architecture": "C2PA Verified Cryptographic Provenance + ConvNeXt Ensemble",
                "engine": "c2pa_verified",
                "status": "success"
            }

        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")

            tensor = self.transform(img).unsqueeze(0).to(self.device)

            model_probs = {}
            if self.backend_engine == 'convnext_ensemble':
                with torch.inference_mode():
                    for name, m in self.convnext_models.items():
                        logits = m(tensor)
                        prob = float(torch.sigmoid(logits.reshape(-1))[0].item())
                        model_probs[name] = prob

                # Weighted ensemble probability
                if self.normalized_weights:
                    fake_prob = sum(self.normalized_weights[k] * model_probs[k] for k in self.convnext_models)
                else:
                    fake_prob = sum(model_probs.values()) / max(len(model_probs), 1)

                architecture_name = f"ConvNeXt-Tiny Ensemble ({' + '.join([k.upper() for k in self.convnext_models.keys()])}) + C2PA"
            elif self.backend_engine == 'onnx':
                out = self.session.run(None, {'input_image': tensor.cpu().numpy()})
                logit = float(out[0][0])
                fake_prob = float(1.0 / (1.0 + np.exp(-logit)))
                architecture_name = "EfficientNetV2-S (ONNX)"
            else:
                with torch.no_grad():
                    logit_tensor = self.pt_model(tensor)
                    logit = float(logit_tensor.item())
                fake_prob = float(1.0 / (1.0 + np.exp(-logit)))
                architecture_name = "EfficientNetV2-S (PyTorch)"

            real_prob = float(1.0 - fake_prob)
            is_fake = fake_prob >= VISUAL_THRESHOLD
            prediction = "FAKE" if is_fake else "REAL"
            confidence = round((fake_prob if is_fake else real_prob) * 100, 2)
            manipulation_score = round(fake_prob * 100, 2)

            return {
                "prediction": prediction,
                "is_fake": is_fake,
                "manipulation_score": manipulation_score,
                "confidence": confidence,
                "score": manipulation_score,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "metadata_forensics": metadata_info,
                "c2pa_provenance": c2pa_result,
                "ensemble_details": {
                    "models": list(self.convnext_models.keys()),
                    "weights": self.normalized_weights,
                    "probabilities": {k: round(v, 4) for k, v in model_probs.items()},
                    "threshold": VISUAL_THRESHOLD
                } if model_probs else None,
                "model_architecture": architecture_name,
                "engine": self.backend_engine,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "error": str(e),
                "prediction": "UNKNOWN",
                "manipulation_score": 0.0,
                "confidence": 0.0,
                "status": "failed"
            }


# Global instance container for clean Flask integration
_detector_instance = None

def predict_image_authenticity(image_path, model_path=None, is_grounded=False):
    """
    Standard interface function for DeepGuard backend integration.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ImageDetector(model_path=model_path)
    return _detector_instance.predict(image_path, is_grounded=is_grounded)


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else r"E:\DEEPGUARD\Testing Samples\Real Images\real.jpg"
    print("Testing Image Detector on:", test_img)
    res = predict_image_authenticity(test_img)
    print("Result:", res)
