import sqlite3
import os

db_path = r"E:\DEEPGUARD\backend\instance\deepguard.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(notifications);").fetchall()]
    if "type" not in cols:
        cursor.execute("ALTER TABLE notifications ADD COLUMN type VARCHAR(50) DEFAULT 'info' NOT NULL;")
        print("Added 'type' column to notifications table.")
        
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS password_resets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token_hash VARCHAR(64) NOT NULL,
        expires_at DATETIME NOT NULL,
        used BOOLEAN NOT NULL DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_password_resets_token_hash ON password_resets(token_hash);")
    conn.commit()
    conn.close()
    print("Database updated successfully.")
