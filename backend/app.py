import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.pipeline import run_deepguard_analysis
from modules.explanation_generator import generate_human_explanation

from transformers import AutoTokenizer, DistilBertForSequenceClassification
from viral_analysis import analyze_viral_patterns

import json
import os
import uuid
import secrets
import hashlib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from functools import wraps

import torch
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    flash
)

from database import db
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from metadata_analysis import extract_important_metadata
from xai_analysis import get_important_words
from models import User, Prediction, Notification, PasswordReset

# =====================================
# Flask App Configuration
# =====================================

app = Flask(__name__, instance_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance"))
app.secret_key = os.environ.get("SECRET_KEY", "deepguard-secret-key-2026")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///deepguard.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB Maximum upload limit per SRS NFR

db.init_app(app)

@app.errorhandler(413)
def request_entity_too_large(error):
    if request.is_json or request.path.startswith("/predict") or request.path.startswith("/api/"):
        return jsonify({"error": "File size exceeds the 10 MB limit. Please upload an image smaller than 10 MB."}), 413
    flash("File size exceeds the 10 MB limit. Please upload an image smaller than 10 MB.", "danger")
    return redirect(request.url), 413

with app.app_context():
    db.create_all()

# =====================================
# Sentiment Analyzer Initialization
# =====================================


analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Neutral 😐"
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive 😊"
    elif compound <= -0.05:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# =====================================
# Helper: Notification Creator
# =====================================

def create_notification(user_id, title, message, notif_type="info"):
    try:
        notif = Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notif_type,
            is_read=False,
            created_at=datetime.utcnow()
        )
        db.session.add(notif)
        db.session.commit()
        return notif
    except Exception as e:
        db.session.rollback()
        print("Error creating notification:", e)
        return None

# =====================================
# Helper: Normalize Metadata (backward compat)
# Converts old-format dicts {Camera Brand, Camera Model, ...}
# into the new structured format {has_metadata, camera, date_time, gps, ...}
# =====================================

def normalize_metadata(meta):
    """Ensure metadata dict always has the new structured format.
    Old predictions stored keys like 'Camera Brand', 'GPS Info' etc.
    New ones use 'camera', 'date_time', 'gps', 'has_metadata' etc.
    This function handles both transparently.
    """
    if not meta or not isinstance(meta, dict):
        return {"has_metadata": False, "status": "No image analyzed"}

    # Already in new format (has the has_metadata key)
    if "has_metadata" in meta:
        return meta

    # Old format — detect and remap
    old_camera_brand = meta.get("Camera Brand") or meta.get("camera_make")
    old_camera_model = meta.get("Camera Model") or meta.get("camera_model")
    old_date = meta.get("Date Taken") or meta.get("date_time") or meta.get("DateTime")
    old_gps = meta.get("GPS Info") or meta.get("gps")
    old_software = meta.get("Software") or meta.get("software")
    old_status = meta.get("Status") or meta.get("status", "")
    old_reason = meta.get("Reason") or meta.get("reason") or meta.get("note")
    old_width = meta.get("Image Width") or meta.get("width")
    old_height = meta.get("Image Height") or meta.get("height")

    # If old format had 'No Metadata Found' status
    if "no metadata" in str(old_status).lower() or "not found" in str(old_status).lower():
        return {
            "has_metadata": False,
            "status": old_status,
            "note": old_reason or "No EXIF metadata was found in this image."
        }

    # Build new format from old fields
    camera = f"{old_camera_brand or ''} {old_camera_model or ''}".strip() or None
    resolution = f"{old_width} × {old_height}" if (old_width and old_height) else None

    if not camera and not old_date:
        return {
            "has_metadata": False,
            "status": "No EXIF metadata found (legacy record)",
            "note": "This record was analyzed before the metadata update. Re-analyze the image to see full EXIF details."
        }

    return {
        "has_metadata": True,
        "status": "Metadata extracted (legacy format)",
        "camera": camera,
        "camera_make": old_camera_brand,
        "camera_model": old_camera_model,
        "date_time": old_date,
        "software": old_software,
        "resolution": resolution,
        "focal_length": meta.get("focal_length") or meta.get("FocalLength"),
        "aperture": meta.get("aperture") or meta.get("FNumber"),
        "iso": meta.get("iso") or meta.get("ISOSpeedRatings"),
        "exposure": meta.get("exposure") or meta.get("ExposureTime"),
        "flash": meta.get("flash") or meta.get("Flash"),
        "gps": old_gps,
        "authenticity_flags": [],
        "metadata_fields_count": sum(1 for v in [camera, old_date, old_gps, old_software] if v)
    }

# =====================================
# Helper: Send Email (SMTP with Fallback)
# =====================================

def send_reset_email(to_email, reset_url):
    smtp_server = os.environ.get("MAIL_SERVER")
    smtp_port = int(os.environ.get("MAIL_PORT", 587))
    smtp_user = os.environ.get("MAIL_USERNAME")
    smtp_pass = os.environ.get("MAIL_PASSWORD")

    if smtp_server and smtp_user and smtp_pass:
        try:
            msg = MIMEText(f"Hello,\n\nYou requested a password reset for your DeepGuard account.\nClick the link below to reset your password (valid for 30 minutes):\n\n{reset_url}\n\nIf you did not request this, please ignore this email.")
            msg['Subject'] = "DeepGuard — Password Reset Link"
            msg['From'] = smtp_user
            msg['To'] = to_email

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            return True
        except Exception as e:
            print("SMTP Email sending failed:", e)

    # Console fallback
    print(f"\n========================================================")
    print(f"[CONSOLE DEMO] Password Reset Link for {to_email}:")
    print(f"{reset_url}")
    print(f"========================================================\n")
    return False

# =====================================
# Auth Decorator
# =====================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# =====================================
# Authentication Routes
# =====================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            return render_template("register.html", error="All fields are required.", entered_name=name, entered_email=email)

        if len(name) < 2:
            return render_template("register.html", error="Name must contain at least 2 characters.", entered_name=name, entered_email=email)

        if "@" not in email or "." not in email.split("@")[-1]:
            return render_template("register.html", error="Please enter a valid email address.", entered_name=name, entered_email=email)

        if len(password) < 8:
            return render_template("register.html", error="Password must be at least 8 characters long.", entered_name=name, entered_email=email)

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.", entered_name=name, entered_email=email)

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template("register.html", error="An account with this email already exists.", entered_name=name, entered_email=email)

        password_hash = generate_password_hash(password)
        new_user = User(name=name, email=email, password_hash=password_hash)
        db.session.add(new_user)
        db.session.commit()

        create_notification(
            user_id=new_user.id,
            title="Welcome to DeepGuard!",
            message="Your account has been created successfully. Start verifying claims and images on your dashboard.",
            notif_type="success"
        )

        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template("login.html", error="Email and password are required.", entered_email=email)

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return render_template("login.html", error="Invalid email or password.", entered_email=email)

        session["user_id"] = user.id
        session["user_name"] = user.name
        session["user_email"] = user.email

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =====================================
# Password Reset Routes (Task 6)
# =====================================

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                raw_token = secrets.token_urlsafe(32)
                token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
                expires_at = datetime.utcnow() + timedelta(minutes=30)

                reset_rec = PasswordReset(
                    user_id=user.id,
                    token_hash=token_hash,
                    expires_at=expires_at,
                    used=False
                )
                db.session.add(reset_rec)
                db.session.commit()

                reset_url = url_for("reset_password", token=raw_token, _external=True)
                sent = send_reset_email(user.email, reset_url)

                return render_template(
                    "forgot_password.html",
                    reset_url=reset_url,
                    email=user.email,
                    email_sent=sent
                )
            else:
                flash("No account registered with that email address. Please check your email or register.", "error")
                return render_template("forgot_password.html", entered_email=email)

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    reset_rec = PasswordReset.query.filter_by(token_hash=token_hash, used=False).first()

    if not reset_rec or reset_rec.expires_at < datetime.utcnow():
        return render_template("reset_password.html", error="This password reset link is invalid or has expired.")

    user = User.query.get(reset_rec.user_id)
    if not user:
        return render_template("reset_password.html", error="User account not found.")

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 8:
            return render_template("reset_password.html", error="Password must be at least 8 characters long.")

        if password != confirm_password:
            return render_template("reset_password.html", error="Passwords do not match.")

        user.password_hash = generate_password_hash(password)
        reset_rec.used = True
        db.session.commit()

        create_notification(
            user_id=user.id,
            title="Password Reset Successful",
            message="Your account password was reset successfully. You can now log in with your new password.",
            notif_type="success"
        )

        flash("Your password has been reset successfully. Please sign in with your new password.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")

# =====================================
# Dashboard & History Routes (Task 4)
# =====================================

@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    user_preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()

    total_count = len(user_preds)
    fake_count = sum(1 for p in user_preds if "FAKE" in p.prediction.upper())
    real_count = sum(1 for p in user_preds if "REAL" in p.prediction.upper())
    limited_count = sum(1 for p in user_preds if p.input_type != "Multimodal")
    recent_predictions = user_preds[:5]

    return render_template(
        "dashboard.html",
        total_count=total_count,
        fake_count=fake_count,
        real_count=real_count,
        limited_count=limited_count,
        recent_predictions=recent_predictions
    )


@app.route("/history")
@login_required
def history():
    user_preds = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=user_preds)


@app.route("/history/delete/<int:id>", methods=["POST"])
@login_required
def delete_history(id):
    pred = Prediction.query.filter_by(id=id, user_id=session["user_id"]).first()
    if pred:
        db.session.delete(pred)
        db.session.commit()
        flash("Prediction record deleted successfully.", "success")
    else:
        flash("Prediction record not found or access denied.", "error")
    return redirect(url_for("history"))

# =====================================
# Analyze & Predict Routes (Task 2 & Task 3)
# =====================================

@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    text = ""
    temp_image_path = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
    else:
        text = request.form.get("text", "").strip()
        if "image" in request.files:
            file = request.files["image"]
            if file and file.filename != "":
                upload_folder = os.path.join(app.root_path, "temp_uploads")
                os.makedirs(upload_folder, exist_ok=True)
                ext = os.path.splitext(secure_filename(file.filename))[1]
                temp_filename = f"{uuid.uuid4()}{ext}"
                temp_image_path = os.path.join(upload_folder, temp_filename)
                file.save(temp_image_path)

                # Server-side verification: file must not exceed 10 MB
                if os.path.exists(temp_image_path):
                    saved_size = os.path.getsize(temp_image_path)
                    if saved_size > 10 * 1024 * 1024:
                        try:
                            os.remove(temp_image_path)
                        except Exception:
                            pass
                        actual_mb = round(saved_size / (1024 * 1024), 2)
                        return jsonify({"error": f"Image file size ({actual_mb} MB) exceeds maximum allowed limit of 10 MB. Please upload an image smaller than 10 MB."}), 413

    if not text and not temp_image_path:
        return jsonify({"error": "Please provide text content, an image, or both for analysis."}), 400

    # Validate text length limits (server-side safety net)
    MIN_CHARS = 20
    MAX_CHARS = 1200
    if text:
        if len(text) > MAX_CHARS:
            return jsonify({"error": f"Text exceeds the maximum limit of {MAX_CHARS} characters. Please shorten your input."}), 400
        if len(text) < MIN_CHARS:
            return jsonify({"error": f"Text is too short ({len(text)} character{'s' if len(text) != 1 else ''} entered). Please enter at least {MIN_CHARS} characters for accurate analysis."}), 400

    user_id = session.get("user_id")

    # 1. Extract Metadata BEFORE pipeline run (so we can pass it to fusion)
    metadata = {}
    metadata_score = None  # None = not applicable
    if temp_image_path:
        try:
            metadata = extract_important_metadata(temp_image_path)
            if metadata.get("has_metadata"):
                # Real camera images with EXIF: LOW fake score from metadata perspective
                # Camera make, GPS, timestamp present → likely authentic
                fields = metadata.get("metadata_fields_count", 0)
                auth_flags = len(metadata.get("authenticity_flags", []))
                # More fields = more authentic, editing flags push score up
                metadata_score = max(0.0, min(40.0, (4 - fields) * 10 + auth_flags * 15))
            else:
                # No EXIF at all = slightly suspicious (could be stripped/AI-generated)
                metadata_score = 55.0
        except Exception as e:
            print("Metadata extraction error:", e)
            metadata = {}

    # 2. Run Unified Multi-Modal AI Pipeline
    pipeline_res = run_deepguard_analysis(text=text, image_path=temp_image_path, user_id=user_id)
    if pipeline_res.get("status") != "success":
        create_notification(user_id, "Analysis Failed", "DeepGuard encountered an error while processing your request.", "danger")
        return jsonify({"error": pipeline_res.get("error", "Analysis pipeline execution failed.")}), 500

    # Inject metadata_score into the fusion breakdown if available
    if metadata_score is not None and "fusion_breakdown" in pipeline_res:
        pipeline_res["fusion_breakdown"]["metadata_score"] = {
            "raw_score": metadata_score,
            "note": "Derived from EXIF authenticity analysis"
        }

    # 3. Plain English Explanation & Heuristics
    human_explanation = generate_human_explanation(pipeline_res)
    verdict_str = "FAKE" if pipeline_res["is_fake"] else "REAL"
    confidence = pipeline_res["confidence"]
    sentiment = analyze_sentiment(text) if text else "Neutral"
    important_words = get_important_words(text, top_k=5) if text else []
    viral_patterns = analyze_viral_patterns(text) if text else {"viral_score": 0, "patterns_detected": []}

    input_type = "Multimodal" if (text and temp_image_path) else ("Image" if temp_image_path else "Text")

    # Pack comprehensive XAI summary JSON
    xai_data = json.dumps({
        "important_words": important_words,
        "human_explanation": human_explanation,
        "pipeline": pipeline_res,
        "metadata": metadata,
        "viral_patterns": viral_patterns
    })

    # Save Prediction to Database
    new_prediction = Prediction(
        user_id=user_id,
        input_type=input_type,
        input_text=text or "Image Upload",
        xai_summary=xai_data,
        prediction=verdict_str,
        confidence=round(confidence, 2),
        sentiment=sentiment
    )
    db.session.add(new_prediction)
    db.session.commit()

    # Create Notification (Task 5)
    create_notification(
        user_id=user_id,
        title="Analysis Complete",
        message=f"DeepGuard verified your {input_type.lower()} input as {verdict_str} ({round(confidence, 2)}% confidence).",
        notif_type="success"
    )

    # Clean up temporary uploaded file if created
    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception:
            pass

    return jsonify({
        "status": "success",
        "prediction_id": new_prediction.id,
        "prediction": verdict_str,
        "verdict": pipeline_res["verdict"],
        "confidence": round(confidence, 2)
    })


@app.route("/report/<int:id>")
@login_required
def view_report(id):
    prediction = Prediction.query.filter_by(id=id, user_id=session["user_id"]).first_or_404()

    xai_words = []
    human_exp = {}
    pipeline_result = {}
    metadata_info = {}
    viral_patterns = {}

    if prediction.xai_summary:
        try:
            data = json.loads(prediction.xai_summary)
            xai_words = data.get("important_words", [])
            human_exp = data.get("human_explanation", {})
            pipeline_result = data.get("pipeline", {})
            metadata_info = normalize_metadata(data.get("metadata", {}))
            viral_patterns = data.get("viral_patterns", {})
        except Exception as e:
            print("Error parsing prediction xai_summary JSON:", e)

    return render_template(
        "result.html",
        prediction=prediction,
        xai_words=xai_words,
        human_exp=human_exp,
        pipeline_result=pipeline_result,
        metadata_info=metadata_info,
        viral_patterns=viral_patterns
    )

# =====================================
# Notification Routes (Task 5)
# =====================================

@app.route("/notifications", methods=["GET"])
@login_required
def notifications():
    user_notifs = Notification.query.filter_by(user_id=session["user_id"]).order_by(Notification.created_at.desc()).limit(10).all()
    unread_count = Notification.query.filter_by(user_id=session["user_id"], is_read=False).count()

    return jsonify({
        "notifications": [
            {
                "id": n.id,
                "title": n.title,
                "message": n.message,
                "type": n.type,
                "is_read": n.is_read,
                "created_at": n.created_at.strftime("%b %d, %H:%M")
            }
            for n in user_notifs
        ],
        "unread_count": unread_count
    })


@app.route("/notifications/<int:id>/read", methods=["POST"])
@login_required
def mark_notification_read(id):
    notif = Notification.query.filter_by(id=id, user_id=session["user_id"]).first()
    if notif:
        notif.is_read = True
        db.session.commit()
    return jsonify({"success": True})


@app.route("/notifications/read-all", methods=["POST"])
@login_required
def mark_all_notifications_read():
    Notification.query.filter_by(user_id=session["user_id"], is_read=False).update({"is_read": True})
    db.session.commit()
    return jsonify({"success": True})

# =====================================
# Profile Update Routes
# =====================================

@app.route("/profile")
@login_required
def profile():
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))
    return render_template("profile.html", user=user)


@app.route("/update-profile", methods=["POST"])
@login_required
def update_profile():
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()

    if not name or len(name) < 2:
        flash("Full name must contain at least 2 characters.", "error")
        return redirect(url_for("profile"))

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("profile"))

    existing_user = User.query.filter(User.email == email, User.id != user.id).first()
    if existing_user:
        flash("An account with this email already exists.", "error")
        return redirect(url_for("profile"))

    user.name = name
    user.email = email
    db.session.commit()

    session["user_name"] = user.name
    session["user_email"] = user.email

    create_notification(user.id, "Profile Updated", "Your profile details have been updated successfully.", "info")
    flash("Your profile has been updated successfully.", "success")
    return redirect(url_for("profile"))


@app.route("/change-password", methods=["POST"])
@login_required
def change_password():
    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not check_password_hash(user.password_hash, current_password):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("profile"))

    if len(new_password) < 8:
        flash("New password must be at least 8 characters long.", "error")
        return redirect(url_for("profile"))

    if new_password != confirm_password:
        flash("New passwords do not match.", "error")
        return redirect(url_for("profile"))

    user.password_hash = generate_password_hash(new_password)
    db.session.commit()

    create_notification(user.id, "Password Changed", "Your account password was updated successfully.", "success")
    flash("Password updated successfully!", "success")
    return redirect(url_for("profile"))

# =====================================
# Modules & About Info Pages
# =====================================

@app.route("/modules")
def modules():
    return render_template("modules.html")


@app.route("/about")
def about():
    return render_template("about.html")

# =====================================
# Main Runner
# =====================================

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)