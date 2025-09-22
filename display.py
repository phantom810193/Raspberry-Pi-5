# display.py - Minimal text output (>= 5 lines, < 5s)
# If users.db exists with a 'purchases' table, print 5 rows; otherwise print placeholders.

import sqlite3, os, time, sys

def print_rows_from_db(db_path="users.db"):
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]
        if "purchases" in tables:
            cur.execute("SELECT id, user, item, amount FROM purchases ORDER BY id LIMIT 5;")
            rows = cur.fetchall()
            for r in rows:
                print(f"[DB] id={r[0]} user={r[1]} item={r[2]} amount={r[3]}")
            return len(rows) >= 5
        return False
    except Exception:
        return False
    finally:
        con.close()

def main():
    # Try DB first; fallback to 5 placeholder lines
    ok = False
    if os.path.exists("users.db"):
        ok = print_rows_from_db("users.db")
    if not ok:
        for i in range(1, 6):
            print(f"[TEXT] line {i}: hello LCD/terminal")
    sys.exit(0)

if __name__ == "__main__":
    main()
