# tests/init_db.py - optional helper to create users.db from data.sql (run locally)
import sqlite3, pathlib

def main():
    sql_path = pathlib.Path("data.sql")
    db_path = pathlib.Path("users.db")
    if not sql_path.exists():
        raise SystemExit("data.sql not found in repo root")
    sql = sql_path.read_text(encoding="utf-8")
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(sql)
        con.commit()
        print("users.db created with seed data.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
