# DeepGuard Trained Models & Weights

This directory documents the machine learning and deep learning models developed across the DeepGuard project lifecycle.

## Active Production Models

| Subsystem | Architecture | Engine / Format | Location |
|---|---|---|---|
| **Visual Fake Detection** | EfficientNetV2-S (Fine-Tuned) | ONNX Runtime (`float32`) | `modules/image_detection/models/deepguard_image_model.onnx` |
| **Contextual Text Analysis** | DistilBERT (`distilbert-base-uncased`) | PyTorch / HuggingFace | `backend/deepguard_distilbert/` & `models/final_distilbert_2026/FINAL_MODEL` |
| **Cross-Modal Mismatch** | CLIP ViT-B/32 | PyTorch (Zero-Shot Cross-Modal) | `modules/context_mismatch/mismatch.py` |
| **Sentiment Analysis** | VADER Lexicon | Rule-based Lexicon | `modules/sentiment/sentiment_analyzer.py` |

---

## Historical & Prototype Models

* **`deepguard_svm_model.pkl` & `deepguard_vectorizer.pkl`**: Early Linear SVM baseline model with TF-IDF n-gram vectorizer used during the initial prototype phase before upgrading to DistilBERT.
* **`FINAL ALL DATSETS/`**: Checkpoint directory containing composite training artifacts.
