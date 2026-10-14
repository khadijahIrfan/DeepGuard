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

logger = logging.getLogger(__name__)

# Relative paths based on module location
MODULE_DIR = pathlib.Path(__file__).parent.resolve()
ONNX_MODEL_PATH = MODULE_DIR / "models" / "deepguard_image_model.onnx"
PT_MODEL_PATH = MODULE_DIR / "models" / "deepguard_image_model_best.pt"
HYBRID_PATH = MODULE_DIR / "models" / "best_hybrid_multimodal_falsified_news_model.pt"

class DeepGuardImageNet(nn.Module):
    """
    Native PyTorch Architecture for DeepGuard EfficientNetV2-S.
    Used as high-fidelity fallback when ONNX runtime is unavailable.
    """
    def __init__(self, backbone_name='tf_efficientnetv2_s.in21k_ft_in1k'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
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


class ImageDetector:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.onnx_path = pathlib.Path(model_path) if model_path and str(model_path).endswith('.onnx') else ONNX_MODEL_PATH
        self.pt_path = pathlib.Path(model_path) if model_path and str(model_path).endswith('.pt') else PT_MODEL_PATH
        self.session = None
        self.pt_model = None
        self.backend_engine = None  # 'onnx' or 'pytorch'
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        if self.backend_engine is not None:
            return True

        # Priority 1: High-performance ONNX Engine
        if HAS_ONNX and self.onnx_path.is_file():
            try:
                self.session = ort.InferenceSession(str(self.onnx_path))
                self.backend_engine = 'onnx'
                logger.info(f"Loaded DeepGuard Image Model via ONNX runtime: {self.onnx_path.name}")
                return True
            except Exception as e:
                logger.warning(f"ONNX initialization failed, falling back to PyTorch: {e}")

        # Priority 2: Native PyTorch EfficientNetV2-S Engine
        if HAS_TIMM and self.pt_path.is_file():
            try:
                checkpoint = torch.load(str(self.pt_path), map_location=torch.device('cpu'))
                backbone_name = checkpoint.get('backbone', 'tf_efficientnetv2_s.in21k_ft_in1k')
                model = DeepGuardImageNet(backbone_name=backbone_name)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.pt_model = model
                self.backend_engine = 'pytorch'
                logger.info(f"Loaded DeepGuard Image Model via PyTorch: {self.pt_path.name}")
                return True
            except Exception as e:
                logger.error(f"PyTorch model initialization failed: {e}")

        logger.error("No valid DeepGuard Image model could be initialized.")
        return False

    def predict(self, image_path):
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

        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")

            tensor = self.transform(img).unsqueeze(0)

            # Execute Inference
            if self.backend_engine == 'onnx':
                out = self.session.run(None, {'input_image': tensor.numpy()})
                logit = float(out[0][0])
            else:
                with torch.no_grad():
                    logit_tensor = self.pt_model(tensor)
                    logit = float(logit_tensor.item())

            # Binary Sigmoid: 0 = Real, 1 = AI-Generated
            fake_prob = float(1.0 / (1.0 + np.exp(-logit)))
            real_prob = float(1.0 - fake_prob)

            is_fake = fake_prob >= 0.50
            prediction = "FAKE" if is_fake else "REAL"
            confidence = round((fake_prob if is_fake else real_prob) * 100, 2)
            manipulation_score = round(fake_prob * 100, 2)

            # Integrated Forensic Metadata Extraction
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

            return {
                "prediction": prediction,
                "is_fake": is_fake,
                "manipulation_score": manipulation_score,
                "confidence": confidence,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "metadata_forensics": metadata_info,
                "model_architecture": "EfficientNetV2-S (Fine-Tuned)",
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

def predict_image_authenticity(image_path, model_path=None):
    """
    Standard interface function for DeepGuard backend integration.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ImageDetector(model_path=model_path)
    return _detector_instance.predict(image_path)


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else r"E:\DEEPGUARD\backend\test_images\test.jpg"
    print("Testing Image Detector on:", test_img)
    res = predict_image_authenticity(test_img)
    print("Result:", res)
