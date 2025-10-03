from pathlib import Path
import sqlite3

from pi_kiosk import database

DB_PATH = Path('data/kiosk.db')

if not DB_PATH.exists():
    raise SystemExit(f"Database {DB_PATH} does not exist")

conn = database.connect(DB_PATH)
database.initialize_db(conn)
rows = conn.execute(
    "SELECT member_id, created_at, length(descriptor) AS descriptor_bytes, length(snapshot) AS snapshot_bytes FROM face_features ORDER BY created_at"
).fetchall()

if not rows:
    print("face_features table is empty")
else:
    for row in rows:
        print(
            f"member_id={row['member_id']}, created_at={row['created_at']}, "
            f"descriptor_bytes={row['descriptor_bytes']}, snapshot_bytes={row['snapshot_bytes']}"
        )

conn.close()
