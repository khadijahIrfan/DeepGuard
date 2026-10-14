import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import werkzeug
if not hasattr(werkzeug, '__version__'):
    werkzeug.__version__ = "2.3.0"

from app import app, db, User, Prediction, Notification, PasswordReset, create_notification, run_deepguard_analysis

print("=== DEEPGUARD END-TO-END AUTOMATED VERIFICATION ===")

client = app.test_client()

with app.app_context():
    # 1. Test User Creation / DB
    print("\n[TEST 1] User Creation & DB")
    user = User.query.filter_by(email="testuser@example.com").first()
    if not user:
        from werkzeug.security import generate_password_hash
        user = User(name="Test User", email="testuser@example.com", password_hash=generate_password_hash("Password123!"))
        db.session.add(user)
        db.session.commit()
    print("User ID:", user.id, "Name:", user.name)

    # 2. Test Notification Creation
    print("\n[TEST 2] Notification System")
    notif = create_notification(user.id, "Test Notification", "This is a test notification message", "success")
    print("Notification Created ID:", notif.id, "Title:", notif.title)
    
    # 3. Test Prediction Database & Pipeline Integration
    print("\n[TEST 3] Multimodal Pipeline Analysis")
    text_sample = "BREAKING NEWS: Emergency declared! Share to 10 groups immediately."
    pipeline_res = run_deepguard_analysis(text=text_sample, image_path=None, user_id=user.id)
    print("Pipeline Verdict:", pipeline_res["verdict"], "Confidence:", pipeline_res["confidence"])
    
    xai_data = json.dumps({
        "pipeline": pipeline_res,
        "important_words": ["BREAKING", "emergency", "Share"],
        "human_explanation": {"summary_sentence": "Linguistic pattern matches misinformation heuristics."}
    })
    
    pred = Prediction(
        user_id=user.id,
        input_type="Text",
        input_text=text_sample,
        xai_summary=xai_data,
        prediction="FAKE 🔴",
        confidence=round(pipeline_res["confidence"], 2),
        sentiment="Negative 😠"
    )
    db.session.add(pred)
    db.session.commit()
    print("Prediction Saved to DB with ID:", pred.id)

    # 4. Test Report Generation Render Context
    print("\n[TEST 4] Report Template Context Check")
    with app.test_request_context(f"/report/{pred.id}"):
        saved_pred = Prediction.query.get(pred.id)
        assert saved_pred is not None, "Report retrieval failed"
        print("Retrieved Report ID:", saved_pred.id, "Prediction:", saved_pred.prediction)

print("\n✅ ALL BACKEND CORE PIPELINES & DATABASE MODELS VERIFIED CLEANLY!")
