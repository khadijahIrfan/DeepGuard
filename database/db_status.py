"""
DeepGuard Database Inspection Utility
Reports table health, record counts, and latest activity.
"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "backend" / "instance" / "deepguard.db"

def inspect_db():
    print("=" * 60)
    print("  DEEPGUARD DATABASE HEALTH & STATUS")
    print("=" * 60)
    print(f"Target DB Path: {DB_PATH}")

    if not DB_PATH.exists():
        print("[ERROR] Database file not found!")
        return

    size_kb = round(DB_PATH.stat().st_size / 1024, 2)
    print(f"File Size:      {size_kb} KB ({DB_PATH.stat().st_size} bytes)")
    print("-" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    tables = ["users", "predictions", "notifications", "password_resets"]
    for t in tables:
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  * Table '{t:<15}': {count:>5} records")
        except sqlite3.OperationalError:
            print(f"  * Table '{t:<15}': [NOT FOUND]")

    print("-" * 60)
    # Latest 3 predictions
    try:
        preds = cursor.execute("SELECT id, input_type, prediction, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 3").fetchall()
        print("Latest Predictions:")
        for p in preds:
            print(f"    ID: {p[0]} | Type: {p[1]:<10} | Verdict: {p[2]:<5} | Conf: {p[3]:.1f}% | Date: {p[4]}")
    except Exception as e:
        print(f"    (Could not fetch predictions: {e})")

    print("=" * 60)
    conn.close()

if __name__ == "__main__":
    inspect_db()
