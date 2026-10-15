"""
Safe Database Migration for DeepGuard v3 (Task 9)
Creates `assistant_messages` table if missing.
Preserves all existing users, predictions, notifications, feedback, and saved items.
"""
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

    # Create assistant_messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assistant_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        analysis_id INTEGER NOT NULL,
        question VARCHAR(300) NOT NULL,
        answer TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(analysis_id) REFERENCES predictions(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_assistant_user ON assistant_messages(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_assistant_analysis ON assistant_messages(analysis_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_assistant_user_analysis ON assistant_messages(user_id, analysis_id)")
    conn.commit()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assistant_messages'")
    tbl = cursor.fetchone()
    if tbl:
        print("  -> Table 'assistant_messages' is ready.")

    conn.close()
    print("[+] Migration v3 completed successfully.")

if __name__ == "__main__":
    run_migration()
