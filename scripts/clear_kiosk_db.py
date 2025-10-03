"""Utility to wipe members and face_features tables from data/kiosk.db."""
from pathlib import Path
import sqlite3

DB_PATH = Path('data/kiosk.db')

if not DB_PATH.exists():
    raise SystemExit(f"Database {DB_PATH} does not exist")

conn = sqlite3.connect(DB_PATH)
with conn:
    conn.execute("DELETE FROM face_features")
    conn.execute("DELETE FROM members")

conn.close()
print("Cleared tables: members, face_features")
