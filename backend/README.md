# DeepGuard Backend Engine

The backend of DeepGuard is powered by Flask, SQLAlchemy, PyTorch, and HuggingFace Transformers, delivering real-time multimodal forensic analysis.

## Core Modules & Services

| File | Type | Description |
|---|---|---|
| [`app.py`](app.py) | Application Core | Flask entry point, API route controllers, session authentication, and error handlers. |
| [`database.py`](database.py) | Database Init | SQLAlchemy instance initialization and configuration binding. |
| [`models.py`](models.py) | ORM Schema | Database models: `User`, `Prediction`, `Notification`, and `PasswordReset`. |
| [`metadata_analysis.py`](metadata_analysis.py) | Forensics Service | Image EXIF metadata parsing, camera make/model detection, GPS, and timestamp verification. |
| [`viral_analysis.py`](viral_analysis.py) | Heuristics Engine | Viral pattern detection, sensationalism markers, urgency triggers, and engagement scores. |
| [`xai_analysis.py`](xai_analysis.py) | Explainability Service | Word-level attribution, sentiment intensity, and keyword highlighting for user explanations. |
| [`migrate_db.py`](migrate_db.py) | Maintenance | Database migration and table synchronization utility. |

---

## Directory Structure

* **`instance/`**: Contains the active SQLite database file (`deepguard.db`).
* **`templates/`**: 13 Jinja2 HTML templates providing the user interface.
* **`static/`**: CSS design tokens, modern themes, client-side JavaScript, and brand assets.
* **`deepguard_distilbert/`**: Fine-tuned DistilBERT transformer weights and configuration.
* **`test_images/`**: Verified image assets for testing and demonstration.
* **`archive_legacy_svm/`**: Preserved historical artifacts from the early prototype phase (SVM model, TF-IDF vectorizer, and prototype scripts).

---

## Running the Server
```bash
# From the project root or backend folder:
python backend/app.py
```
* **Port:** `http://127.0.0.1:5000`
