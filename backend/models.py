from database import db
from datetime import datetime

# =====================================
# User Model
# =====================================

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    theme = db.Column(db.String(20), default="dark")
    role = db.Column(db.String(20), default="user")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    predictions = db.relationship("Prediction", backref="user", lazy=True, cascade="all, delete-orphan")
    notifications = db.relationship("Notification", backref="user", lazy=True, cascade="all, delete-orphan")
    password_resets = db.relationship("PasswordReset", backref="user", lazy=True, cascade="all, delete-orphan")
    feedback_messages = db.relationship("FeedbackMessage", backref="user", lazy=True, cascade="all, delete-orphan")
    saved_reports = db.relationship("SavedReport", backref="user", lazy=True, cascade="all, delete-orphan")
    assistant_messages = db.relationship("AssistantMessage", backref="user", lazy=True, cascade="all, delete-orphan")


# =====================================
# Prediction Model
# =====================================

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    input_type = db.Column(db.String(20), nullable=False) # Text / Image / Multimodal
    input_text = db.Column(db.Text, nullable=True)
    xai_summary = db.Column(db.Text, nullable=True) # Full pipeline JSON output
    prediction = db.Column(db.String(20), nullable=False) # REAL / FAKE / UNCERTAIN
    confidence = db.Column(db.Float, nullable=False)
    sentiment = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================
# Notification Model
# =====================================

class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), default="info", nullable=False) # success, warning, danger, info
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================
# Password Reset Model
# =====================================

class PasswordReset(db.Model):
    __tablename__ = "password_resets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    token_hash = db.Column(db.String(64), nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================
# Feedback Message Model
# =====================================

class FeedbackMessage(db.Model):
    __tablename__ = "feedback_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    type = db.Column(db.String(50), nullable=False) # Bug / Suggestion / Other
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="new", nullable=False) # new / reviewed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================
# Saved Report Model
# =====================================

class SavedReport(db.Model):
    __tablename__ = "saved_reports"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    analysis_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    tag = db.Column(db.String(100), nullable=True) # e.g. "To verify later"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    prediction = db.relationship("Prediction", backref=db.backref("saved_records", cascade="all, delete-orphan"))

    __table_args__ = (
        db.UniqueConstraint("user_id", "analysis_id", name="uq_user_analysis_saved"),
    )


# =====================================
# Assistant Message Model
# =====================================

class AssistantMessage(db.Model):
    __tablename__ = "assistant_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    analysis_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=True)
    question = db.Column(db.String(300), nullable=False)
    answer = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    prediction = db.relationship("Prediction", backref=db.backref("assistant_records", cascade="all, delete-orphan"))
