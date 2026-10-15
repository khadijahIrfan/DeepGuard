"""
Safe Database Migration for DeepGuard v2
Adds `theme` and `role` to users table if missing,
and creates `feedback_messages` and `saved_reports` tables.
Preserves all existing users, predictions, notifications, and passwords.
"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "instance" / "deepguard.db"

def run_migration():
    if not DB_PATH.exists():
        print(f"[!] Database file does not exist at {DB_PATH}")
        return

    print(f"[*] Running safe migration on: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 1. Check columns in users table
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in cursor.fetchall()]

    if "theme" not in user_cols:
        print("  -> Adding column 'theme' to users table...")
        cursor.execute("ALTER TABLE users ADD COLUMN theme VARCHAR(20) DEFAULT 'dark'")
    else:
        print("  -> Column 'theme' already exists on users table.")

    if "role" not in user_cols:
        print("  -> Adding column 'role' to users table...")
        cursor.execute("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'")
    else:
        print("  -> Column 'role' already exists on users table.")

    # 2. Create feedback_messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type VARCHAR(50) NOT NULL,
        subject VARCHAR(100) NOT NULL,
        message TEXT NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'new',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_messages(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_messages(status)")
    print("  -> Checked / created 'feedback_messages' table.")

    # 3. Create saved_reports table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS saved_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        analysis_id INTEGER NOT NULL,
        tag VARCHAR(100),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, analysis_id),
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(analysis_id) REFERENCES predictions(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_user ON saved_reports(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_analysis ON saved_reports(analysis_id)")
    print("  -> Checked / created 'saved_reports' table.")

    conn.commit()
    conn.close()
    print("[+] Migration completed safely with 100% data preservation.")

if __name__ == "__main__":
    run_migration()
