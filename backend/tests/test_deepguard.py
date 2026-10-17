"""
DeepGuard Automated Test Suite
================================
Full-fledged unit + integration tests covering:
  - Authentication (register, login, logout)
  - Role-based access control (login_required, admin_required)
  - Admin features (dashboard, user management, role toggle, feedback review)
  - Quiz logic (unique sampling, session cycling, save result)
  - Feedback submission and validation
  - Session management (persistence, clear on logout)
  - Sentiment analysis helper
  - Password reset token generation
  - Notification creation helper
  - Metadata normalizer (both old and new format)

Run from the backend directory:
    python -m pytest tests/test_deepguard.py -v
or:
    python tests/test_deepguard.py
"""

import sys
import os
import json
import unittest
import hashlib
import tempfile

# ── Path bootstrap ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR    = os.path.dirname(BACKEND_DIR)

for path in [BACKEND_DIR, ROOT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Stub out heavy ML imports before importing app ───────────────
import unittest.mock as mock

# Stub transformers
transformers_stub = mock.MagicMock()
sys.modules.setdefault("transformers", transformers_stub)

# Stub torch
torch_stub = mock.MagicMock()
sys.modules.setdefault("torch", torch_stub)

# Stub pipeline module
pipeline_stub = mock.MagicMock()
pipeline_stub.run_deepguard_analysis.return_value = {
    "prediction": "FAKE",
    "confidence": 0.87,
    "text_score": 0.9,
}
sys.modules.setdefault("modules.pipeline", pipeline_stub)

# Stub explanation generator
exp_stub = mock.MagicMock()
exp_stub.generate_human_explanation.return_value = "Stub explanation."
sys.modules.setdefault("modules.explanation_generator", exp_stub)

# Stub viral_analysis
viral_stub = mock.MagicMock()
viral_stub.analyze_viral_patterns.return_value = {}
sys.modules.setdefault("viral_analysis", viral_stub)

# Stub metadata_analysis
meta_stub = mock.MagicMock()
meta_stub.extract_important_metadata.return_value = {}
sys.modules.setdefault("metadata_analysis", meta_stub)

# Stub xai_analysis
xai_stub = mock.MagicMock()
xai_stub.get_important_words.return_value = []
sys.modules.setdefault("xai_analysis", xai_stub)

# Stub rag_service
rag_stub = mock.MagicMock()
sys.modules.setdefault("rag_service", rag_stub)

# Now import the real app
from app import app, db, create_notification, analyze_sentiment, normalize_metadata
from models import User, Prediction, Notification, FeedbackMessage, QuizResult
from werkzeug.security import generate_password_hash


# ════════════════════════════════════════════════════════════════
#  Base Test Setup
# ════════════════════════════════════════════════════════════════

class DeepGuardTestBase(unittest.TestCase):
    """Sets up an in-memory SQLite DB and Flask test client for each test."""

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["SERVER_NAME"] = None
        self.client = app.test_client()
        self.app_ctx = app.app_context()
        self.app_ctx.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_ctx.pop()

    # ── Helpers ──────────────────────────────────────────────────

    def _create_user(self, name="Test User", email="test@example.com",
                     password="TestPass123", role="user"):
        """Create a user directly in the DB and return the User object."""
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
            theme="dark"
        )
        db.session.add(user)
        db.session.commit()
        return user

    def _login(self, email, password):
        """POST login form and return response."""
        return self.client.post("/login", data={
            "email": email,
            "password": password
        }, follow_redirects=True)

    def _login_as_user(self):
        """Create a standard user and log them in."""
        user = self._create_user()
        self._login("test@example.com", "TestPass123")
        return user

    def _login_as_admin(self):
        """Create an admin user and log them in."""
        admin = self._create_user(
            name="Admin User",
            email="admin@deepguard.com",
            password="AdminPass123",
            role="admin"
        )
        self._login("admin@deepguard.com", "AdminPass123")
        return admin


# ════════════════════════════════════════════════════════════════
#  1. Authentication Tests
# ════════════════════════════════════════════════════════════════

class TestAuthentication(DeepGuardTestBase):

    def test_register_get_shows_form(self):
        """GET /register should return 200 with the registration form."""
        resp = self.client.get("/register")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Register", resp.data)

    def test_register_user_success(self):
        """Valid user registration creates account and redirects to login."""
        resp = self.client.post("/register", data={
            "name": "Jane Doe",
            "email": "jane@example.com",
            "password": "SecurePass1",
            "confirm_password": "SecurePass1",
            "role": "user",
            "admin_key": ""
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        user = User.query.filter_by(email="jane@example.com").first()
        self.assertIsNotNone(user)
        self.assertEqual(user.role, "user")

    def test_register_admin_with_correct_key(self):
        """Admin registration with correct secret key sets role='admin'."""
        with mock.patch.dict(os.environ, {"ADMIN_SECRET_KEY": "DEEPGUARD-ADMIN-2026"}):
            resp = self.client.post("/register", data={
                "name": "Super Admin",
                "email": "superadmin@deepguard.com",
                "password": "AdminPass1",
                "confirm_password": "AdminPass1",
                "role": "admin",
                "admin_key": "DEEPGUARD-ADMIN-2026"
            }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        user = User.query.filter_by(email="superadmin@deepguard.com").first()
        self.assertIsNotNone(user)
        self.assertEqual(user.role, "admin")

    def test_register_admin_with_wrong_key_fails(self):
        """Admin registration with wrong key shows an error."""
        resp = self.client.post("/register", data={
            "name": "Fake Admin",
            "email": "fakeadmin@example.com",
            "password": "AdminPass1",
            "confirm_password": "AdminPass1",
            "role": "admin",
            "admin_key": "WRONG-KEY"
        }, follow_redirects=False)
        # Should stay on register page (not redirect to login)
        self.assertEqual(resp.status_code, 200)
        user = User.query.filter_by(email="fakeadmin@example.com").first()
        self.assertIsNone(user)

    def test_register_duplicate_email_fails(self):
        """Registering with an existing email returns an error."""
        self._create_user(email="dup@example.com")
        resp = self.client.post("/register", data={
            "name": "Duplicate",
            "email": "dup@example.com",
            "password": "TestPass123",
            "confirm_password": "TestPass123",
            "role": "user",
            "admin_key": ""
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"already exists", resp.data)

    def test_register_short_password_fails(self):
        """Password shorter than 8 characters is rejected."""
        resp = self.client.post("/register", data={
            "name": "Short",
            "email": "short@example.com",
            "password": "abc",
            "confirm_password": "abc",
            "role": "user",
            "admin_key": ""
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"8 characters", resp.data)

    def test_register_mismatched_passwords_fails(self):
        """Mismatched passwords are rejected."""
        resp = self.client.post("/register", data={
            "name": "Mismatch",
            "email": "mismatch@example.com",
            "password": "TestPass123",
            "confirm_password": "Different123",
            "role": "user",
            "admin_key": ""
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"do not match", resp.data)

    def test_login_get_shows_form(self):
        """GET /login should return 200."""
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)

    def test_login_user_success_redirects_to_dashboard(self):
        """Valid user login redirects to /dashboard."""
        self._create_user()
        resp = self.client.post("/login", data={
            "email": "test@example.com",
            "password": "TestPass123"
        })
        # Should redirect to dashboard (302 → /dashboard)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/dashboard", resp.headers.get("Location", ""))

    def test_login_admin_success_redirects_to_admin_dashboard(self):
        """Valid admin login redirects to /admin."""
        self._create_user(email="admin@deepguard.com", password="AdminPass123", role="admin")
        resp = self.client.post("/login", data={
            "email": "admin@deepguard.com",
            "password": "AdminPass123"
        })
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin", resp.headers.get("Location", ""))

    def test_login_wrong_password_fails(self):
        """Wrong password shows error, no redirect."""
        self._create_user()
        resp = self.client.post("/login", data={
            "email": "test@example.com",
            "password": "WrongPassword"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid email or password", resp.data)

    def test_login_nonexistent_user_fails(self):
        """Non-existent email shows error."""
        resp = self.client.post("/login", data={
            "email": "nobody@example.com",
            "password": "TestPass123"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid email or password", resp.data)

    def test_logout_clears_session_and_redirects(self):
        """Logout clears session and redirects to /login."""
        self._login_as_user()
        resp = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers.get("Location", ""))
        # After logout, hitting a protected page should redirect to login
        resp2 = self.client.get("/dashboard", follow_redirects=False)
        self.assertIn("/login", resp2.headers.get("Location", ""))


# ════════════════════════════════════════════════════════════════
#  2. Role-Based Access Control Tests
# ════════════════════════════════════════════════════════════════

class TestAccessControl(DeepGuardTestBase):

    def test_unauthenticated_dashboard_redirects_to_login(self):
        """/dashboard requires login."""
        resp = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_unauthenticated_quiz_redirects_to_login(self):
        """/quiz requires login."""
        resp = self.client.get("/quiz", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_unauthenticated_feedback_redirects_to_login(self):
        """/feedback requires login."""
        resp = self.client.get("/feedback", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_unauthenticated_admin_dashboard_redirects_to_login(self):
        """/admin requires login (not just any user)."""
        resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

    def test_regular_user_cannot_access_admin_dashboard(self):
        """A regular user hitting /admin gets 403 Forbidden."""
        self._login_as_user()
        resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_cannot_access_admin_users(self):
        """A regular user hitting /admin/users gets 403 Forbidden."""
        self._login_as_user()
        resp = self.client.get("/admin/users", follow_redirects=False)
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_cannot_access_admin_reports(self):
        """A regular user hitting /admin/reports gets 403 Forbidden."""
        self._login_as_user()
        resp = self.client.get("/admin/reports", follow_redirects=False)
        self.assertEqual(resp.status_code, 403)

    def test_admin_can_access_admin_dashboard(self):
        """An admin user successfully reaches /admin."""
        self._login_as_admin()
        resp = self.client.get("/admin", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_admin_can_access_admin_users(self):
        """An admin user successfully reaches /admin/users."""
        self._login_as_admin()
        resp = self.client.get("/admin/users", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_admin_predict_is_forbidden(self):
        """Admin account POSTing to /predict gets 403."""
        self._login_as_admin()
        resp = self.client.post("/predict", json={
            "text": "This is a test claim"
        })
        self.assertEqual(resp.status_code, 403)

    def test_admin_analyze_redirects_to_admin_dashboard(self):
        """Admin hitting /analyze is redirected back to admin dashboard."""
        self._login_as_admin()
        resp = self.client.get("/analyze", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin", resp.headers.get("Location", ""))


# ════════════════════════════════════════════════════════════════
#  3. Session Management Tests
# ════════════════════════════════════════════════════════════════

class TestSessionManagement(DeepGuardTestBase):

    def test_session_set_on_login(self):
        """After login, session contains user_id, role, user_name."""
        self._create_user()
        self.client.post("/login", data={
            "email": "test@example.com",
            "password": "TestPass123"
        })
        with self.client.session_transaction() as sess:
            self.assertIn("user_id", sess)
            self.assertEqual(sess.get("role"), "user")
            self.assertEqual(sess.get("user_name"), "Test User")

    def test_session_role_admin_on_admin_login(self):
        """Admin login sets session['role'] = 'admin'."""
        self._create_user(email="admin@deepguard.com", password="AdminPass123", role="admin")
        self.client.post("/login", data={
            "email": "admin@deepguard.com",
            "password": "AdminPass123"
        })
        with self.client.session_transaction() as sess:
            self.assertEqual(sess.get("role"), "admin")

    def test_session_cleared_on_logout(self):
        """Session is empty after logout."""
        self._login_as_user()
        self.client.get("/logout")
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)
            self.assertNotIn("role", sess)

    def test_session_permanent_on_login(self):
        """Session is marked permanent (24h lifetime) after login."""
        self._create_user()
        self.client.post("/login", data={
            "email": "test@example.com",
            "password": "TestPass123"
        })
        with self.client.session_transaction() as sess:
            self.assertTrue(sess.permanent)


# ════════════════════════════════════════════════════════════════
#  4. Quiz Tests
# ════════════════════════════════════════════════════════════════

class TestQuiz(DeepGuardTestBase):

    SAMPLE_QUIZ = [
        {"id": f"q{i}", "type": ["text", "image", "mismatch"][i % 3],
         "claim": f"Claim {i}", "options": ["A", "B", "C", "D"],
         "correct": "A", "explanation": "Test explanation"}
        for i in range(1, 37)  # 36 questions
    ]

    def _quiz_get(self, category="all"):
        return self.client.get(f"/quiz?category={category}", follow_redirects=True)

    def setUp(self):
        super().setUp()
        # Patch quiz.json loading to return our 36-question sample
        self._quiz_patcher = mock.patch("builtins.open", mock.mock_open(
            read_data=json.dumps(self.SAMPLE_QUIZ)
        ))
        # We'll patch at the route level by injecting data in session

    def test_quiz_requires_login(self):
        """GET /quiz redirects to login if not authenticated."""
        resp = self.client.get("/quiz", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_quiz_unique_sampling_no_repeat(self):
        """Each attempt should have no questions repeated from the previous attempt."""
        user = self._login_as_user()

        # We need to use the real quiz data path
        quiz_path = os.path.join(ROOT_DIR, "data", "quiz.json")
        if not os.path.exists(quiz_path):
            self.skipTest("quiz.json not found; skipping quiz uniqueness test")

        with open(quiz_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        total_questions = len(all_questions)
        if total_questions < 12:
            self.skipTest("Not enough questions in quiz.json for uniqueness test")

        seen_ids_across_attempts = []
        max_attempts = total_questions // 6  # Test as many attempts as we can without cycling

        for attempt in range(max_attempts):
            resp = self._quiz_get()
            self.assertEqual(resp.status_code, 200)
            with self.client.session_transaction() as sess:
                current_seen = sess.get("seen_quiz_ids", [])
            # IDs added in this attempt = last 6 in seen list
            new_ids = current_seen[len(seen_ids_across_attempts):]
            # Verify no overlap with previous seen IDs
            overlap = set(new_ids) & set(seen_ids_across_attempts)
            self.assertEqual(len(overlap), 0,
                f"Attempt {attempt + 1}: repeated question IDs {overlap}")
            seen_ids_across_attempts = current_seen[:]

    def test_quiz_cycles_when_pool_exhausted(self):
        """After seeing all questions, the pool resets and quiz still works."""
        self._login_as_user()

        quiz_path = os.path.join(ROOT_DIR, "data", "quiz.json")
        if not os.path.exists(quiz_path):
            self.skipTest("quiz.json not found")

        with open(quiz_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        all_ids = [q["id"] for q in all_questions]

        # Pre-populate session with ALL question IDs (simulate exhausted pool)
        with self.client.session_transaction() as sess:
            sess["seen_quiz_ids"] = all_ids

        # Next GET should still succeed and return a valid quiz
        resp = self._quiz_get()
        self.assertEqual(resp.status_code, 200)
        # Session seen list should have been partially reset and repopulated
        with self.client.session_transaction() as sess:
            new_seen = sess.get("seen_quiz_ids", [])
        # At least 6 IDs should be in seen (the new attempt's questions)
        self.assertGreaterEqual(len(new_seen), 6)

    def test_save_quiz_result(self):
        """POST /api/quiz/save stores a QuizResult record in DB."""
        self._login_as_user()
        resp = self.client.post("/api/quiz/save", json={
            "score": 5,
            "total": 6
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["score"], 5)
        self.assertAlmostEqual(data["percentage"], 83.3, places=0)

        # Verify DB record
        result = QuizResult.query.first()
        self.assertIsNotNone(result)
        self.assertEqual(result.score, 5)
        self.assertEqual(result.total_questions, 6)

    def test_save_quiz_result_requires_login(self):
        """Unauthenticated POST to /api/quiz/save redirects."""
        resp = self.client.post("/api/quiz/save", json={"score": 5, "total": 6},
                                follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

    def test_quiz_category_filter_text(self):
        """GET /quiz?category=text requests text-only quiz (route should load 200)."""
        self._login_as_user()
        quiz_path = os.path.join(ROOT_DIR, "data", "quiz.json")
        if not os.path.exists(quiz_path):
            self.skipTest("quiz.json not found")
        resp = self._quiz_get(category="text")
        self.assertEqual(resp.status_code, 200)

    def test_quiz_category_filter_image(self):
        """GET /quiz?category=image requests image-only quiz (route should load 200)."""
        self._login_as_user()
        quiz_path = os.path.join(ROOT_DIR, "data", "quiz.json")
        if not os.path.exists(quiz_path):
            self.skipTest("quiz.json not found")
        resp = self._quiz_get(category="image")
        self.assertEqual(resp.status_code, 200)

    def test_quiz_category_filter_mismatch(self):
        """GET /quiz?category=mismatch requests mismatch-only quiz (route should load 200)."""
        self._login_as_user()
        quiz_path = os.path.join(ROOT_DIR, "data", "quiz.json")
        if not os.path.exists(quiz_path):
            self.skipTest("quiz.json not found")
        resp = self._quiz_get(category="mismatch")
        self.assertEqual(resp.status_code, 200)


# ════════════════════════════════════════════════════════════════
#  5. Feedback Tests
# ════════════════════════════════════════════════════════════════

class TestFeedback(DeepGuardTestBase):

    def test_feedback_get_requires_login(self):
        """GET /feedback redirects unauthenticated user."""
        resp = self.client.get("/feedback", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_feedback_get_shows_form_for_logged_in_user(self):
        """Authenticated GET /feedback returns 200."""
        self._login_as_user()
        resp = self.client.get("/feedback")
        self.assertEqual(resp.status_code, 200)

    def test_feedback_submission_saves_to_db(self):
        """Valid feedback POST creates a FeedbackMessage record."""
        self._login_as_user()
        resp = self.client.post("/feedback", data={
            "type": "Bug",
            "subject": "Login button broken",
            "message": "When I click login nothing happens on Firefox."
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        msg = FeedbackMessage.query.first()
        self.assertIsNotNone(msg)
        self.assertEqual(msg.subject, "Login button broken")
        self.assertEqual(msg.type, "Bug")
        self.assertEqual(msg.status, "new")

    def test_feedback_empty_subject_fails(self):
        """Feedback with empty subject returns form error."""
        self._login_as_user()
        resp = self.client.post("/feedback", data={
            "type": "Bug",
            "subject": "",
            "message": "Some message here."
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Subject is required", resp.data)

    def test_feedback_long_subject_fails(self):
        """Feedback with subject > 100 chars returns form error."""
        self._login_as_user()
        resp = self.client.post("/feedback", data={
            "type": "Bug",
            "subject": "X" * 101,
            "message": "Some message here."
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Subject is required", resp.data)

    def test_feedback_empty_message_fails(self):
        """Feedback with empty message returns form error."""
        self._login_as_user()
        resp = self.client.post("/feedback", data={
            "type": "Suggestion",
            "subject": "Valid subject",
            "message": ""
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Message is required", resp.data)

    def test_feedback_creates_notification(self):
        """Successful feedback submission also creates a notification for the user."""
        user = self._login_as_user()
        self.client.post("/feedback", data={
            "type": "Suggestion",
            "subject": "Dark mode toggle",
            "message": "Please add a dark mode toggle in the header."
        }, follow_redirects=True)
        notif = Notification.query.filter_by(user_id=user.id).first()
        self.assertIsNotNone(notif)
        self.assertEqual(notif.title, "Feedback Received")


# ════════════════════════════════════════════════════════════════
#  6. Admin Feature Tests
# ════════════════════════════════════════════════════════════════

class TestAdminFeatures(DeepGuardTestBase):

    def test_admin_dashboard_shows_200(self):
        """Admin dashboard route returns 200 for admin user."""
        self._login_as_admin()
        resp = self.client.get("/admin", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_admin_users_list_shows_all_users(self):
        """Admin /users page lists all registered users."""
        admin = self._login_as_admin()
        self._create_user(name="Alice", email="alice@example.com")
        self._create_user(name="Bob", email="bob@example.com")
        resp = self.client.get("/admin/users", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Alice", resp.data)
        self.assertIn(b"Bob", resp.data)

    def test_admin_toggle_user_to_admin(self):
        """Admin can promote a user to admin via toggle route."""
        self._login_as_admin()
        target = self._create_user(name="Regular", email="regular@example.com", role="user")
        resp = self.client.post(
            f"/admin/users/{target.id}/toggle-role",
            follow_redirects=True
        )
        self.assertEqual(resp.status_code, 200)
        db.session.refresh(target)
        self.assertEqual(target.role, "admin")

    def test_admin_toggle_admin_to_user(self):
        """Admin can demote another admin to user."""
        self._login_as_admin()
        target = self._create_user(name="OtherAdmin", email="other@deepguard.com", role="admin")
        resp = self.client.post(
            f"/admin/users/{target.id}/toggle-role",
            follow_redirects=True
        )
        self.assertEqual(resp.status_code, 200)
        db.session.refresh(target)
        self.assertEqual(target.role, "user")

    def test_admin_cannot_demote_self(self):
        """Admin cannot demote their own account via toggle route."""
        admin = self._login_as_admin()
        resp = self.client.post(
            f"/admin/users/{admin.id}/toggle-role",
            follow_redirects=True
        )
        self.assertEqual(resp.status_code, 200)
        db.session.refresh(admin)
        # Role should remain 'admin'
        self.assertEqual(admin.role, "admin")

    def test_non_admin_toggle_role_is_forbidden(self):
        """Regular user cannot call the toggle-role route (403)."""
        user = self._login_as_user()
        target = self._create_user(name="Target", email="target@example.com", role="user")
        resp = self.client.post(
            f"/admin/users/{target.id}/toggle-role",
            follow_redirects=False
        )
        self.assertEqual(resp.status_code, 403)

    def test_admin_feedback_review(self):
        """Admin can mark a feedback message as reviewed."""
        admin = self._login_as_admin()
        # Create a user and feedback message
        user = self._create_user(name="Sender", email="sender@example.com")
        msg = FeedbackMessage(
            user_id=user.id,
            type="Bug",
            subject="Test bug",
            message="This is a test bug report.",
            status="new"
        )
        db.session.add(msg)
        db.session.commit()

        resp = self.client.post(f"/admin/feedback/{msg.id}/review")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "reviewed")

        db.session.refresh(msg)
        self.assertEqual(msg.status, "reviewed")

    def test_admin_statistics_page_loads(self):
        """Admin /statistics page returns 200."""
        self._login_as_admin()
        resp = self.client.get("/admin/statistics", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_admin_reports_page_loads(self):
        """Admin /reports page returns 200."""
        self._login_as_admin()
        resp = self.client.get("/admin/reports", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

    def test_admin_feedback_page_loads(self):
        """Admin /admin/feedback page returns 200."""
        self._login_as_admin()
        resp = self.client.get("/admin/feedback", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)


# ════════════════════════════════════════════════════════════════
#  7. Unit Tests — Helper Functions
# ════════════════════════════════════════════════════════════════

class TestHelperFunctions(DeepGuardTestBase):

    # ── analyze_sentiment ────────────────────────────────────────

    def test_sentiment_positive(self):
        """Highly positive text returns 'Positive'."""
        result = analyze_sentiment("I love this amazing breakthrough discovery!")
        self.assertIn("Positive", result)

    def test_sentiment_negative(self):
        """Clearly negative text returns 'Negative'."""
        result = analyze_sentiment("This is terrible, awful, and completely disastrous.")
        self.assertIn("Negative", result)

    def test_sentiment_neutral(self):
        """Neutral/empty text returns 'Neutral'."""
        result = analyze_sentiment("The document was filed on March 15.")
        self.assertIn("Neutral", result)

    def test_sentiment_empty_string(self):
        """Empty string should return Neutral (not crash)."""
        result = analyze_sentiment("")
        self.assertIn("Neutral", result)

    def test_sentiment_none(self):
        """None input should return Neutral (not crash)."""
        result = analyze_sentiment(None)
        self.assertIn("Neutral", result)

    # ── create_notification ──────────────────────────────────────

    def test_create_notification_saves_to_db(self):
        """create_notification saves a Notification record."""
        user = self._create_user()
        notif = create_notification(user.id, "Test Title", "Test body", "info")
        self.assertIsNotNone(notif)
        self.assertEqual(notif.title, "Test Title")
        self.assertEqual(notif.type, "info")
        self.assertFalse(notif.is_read)

    def test_create_notification_type_warning(self):
        """create_notification accepts 'warning' type."""
        user = self._create_user()
        notif = create_notification(user.id, "Warning", "Something wrong", "warning")
        self.assertEqual(notif.type, "warning")

    # ── normalize_metadata ───────────────────────────────────────

    def test_normalize_metadata_none_input(self):
        """normalize_metadata with None returns has_metadata=False."""
        result = normalize_metadata(None)
        self.assertFalse(result["has_metadata"])

    def test_normalize_metadata_new_format_passthrough(self):
        """normalize_metadata with already-new-format dict passes through unchanged."""
        new_fmt = {"has_metadata": True, "camera": "Canon EOS", "date_time": "2024:01:01"}
        result = normalize_metadata(new_fmt)
        self.assertEqual(result["camera"], "Canon EOS")
        self.assertTrue(result["has_metadata"])

    def test_normalize_metadata_old_format_conversion(self):
        """normalize_metadata converts old-format Camera Brand/Model keys to new format."""
        old_fmt = {
            "Camera Brand": "Nikon",
            "Camera Model": "D3500",
            "Date Taken": "2023:06:15",
        }
        result = normalize_metadata(old_fmt)
        self.assertTrue(result.get("has_metadata", False))
        self.assertIn("Nikon", result.get("camera", ""))

    def test_normalize_metadata_no_metadata_found_status(self):
        """Old format with 'No Metadata Found' status returns has_metadata=False."""
        old_fmt = {"Status": "No Metadata Found", "Reason": "JPEG has no EXIF."}
        result = normalize_metadata(old_fmt)
        self.assertFalse(result["has_metadata"])

    def test_normalize_metadata_empty_dict(self):
        """normalize_metadata with empty dict returns has_metadata=False."""
        result = normalize_metadata({})
        self.assertFalse(result["has_metadata"])


# ════════════════════════════════════════════════════════════════
#  8. Password Reset Token Tests
# ════════════════════════════════════════════════════════════════

class TestPasswordReset(DeepGuardTestBase):

    def test_forgot_password_get_shows_form(self):
        """GET /forgot-password returns 200."""
        resp = self.client.get("/forgot-password")
        self.assertEqual(resp.status_code, 200)

    def test_forgot_password_nonexistent_email_no_error(self):
        """POSTing unknown email to /forgot-password does NOT reveal user existence."""
        resp = self.client.post("/forgot-password", data={
            "email": "unknown@example.com"
        }, follow_redirects=True)
        # Should not show user-not-found error (security: no enumeration)
        self.assertEqual(resp.status_code, 200)

    def test_forgot_password_valid_email_creates_reset_record(self):
        """POSTing a valid email creates a PasswordReset record in DB."""
        from models import PasswordReset
        user = self._create_user()
        self.client.post("/forgot-password", data={
            "email": "test@example.com"
        }, follow_redirects=True)
        reset = PasswordReset.query.filter_by(user_id=user.id).first()
        self.assertIsNotNone(reset)
        self.assertFalse(reset.used)
        self.assertIsNotNone(reset.token_hash)

    def test_reset_password_invalid_token_returns_error(self):
        """GET /reset-password with a fake token shows error or 404."""
        resp = self.client.get("/reset-password/FAKEFAKETOKEN", follow_redirects=True)
        # Should either return 404 or show invalid token message
        self.assertIn(resp.status_code, [200, 404])


# ════════════════════════════════════════════════════════════════
#  9. Theme Toggle API Test
# ════════════════════════════════════════════════════════════════

class TestThemeAPI(DeepGuardTestBase):

    def test_theme_toggle_dark_to_light(self):
        """POST /api/user/theme switches theme to light."""
        self._login_as_user()
        resp = self.client.post("/api/user/theme", json={"theme": "light"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["theme"], "light")
        self.assertEqual(data["status"], "success")

    def test_theme_toggle_invalid_falls_back_to_dark(self):
        """Invalid theme value defaults to 'dark'."""
        self._login_as_user()
        resp = self.client.post("/api/user/theme", json={"theme": "rainbow"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["theme"], "dark")

    def test_theme_toggle_persists_to_db(self):
        """Theme change is saved to user record in DB."""
        user = self._login_as_user()
        self.client.post("/api/user/theme", json={"theme": "light"})
        db.session.refresh(user)
        self.assertEqual(user.theme, "light")


# ════════════════════════════════════════════════════════════════
#  10. Public Pages Tests
# ════════════════════════════════════════════════════════════════

class TestPublicPages(DeepGuardTestBase):

    def test_about_page_loads(self):
        """GET /about returns 200 without authentication."""
        resp = self.client.get("/about")
        self.assertEqual(resp.status_code, 200)

    def test_modules_page_loads(self):
        """GET /modules returns 200 without authentication."""
        resp = self.client.get("/modules")
        self.assertEqual(resp.status_code, 200)

    def test_root_redirects(self):
        """GET / returns a redirect or 200 (not an error)."""
        resp = self.client.get("/")
        self.assertIn(resp.status_code, [200, 301, 302])

    def test_library_page_loads_for_authenticated_user(self):
        """GET /library returns 200 for a logged-in user."""
        self._login_as_user()
        resp = self.client.get("/library", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)


# ════════════════════════════════════════════════════════════════
#  Entry Point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
