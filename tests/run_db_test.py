# tests/run_db_test.py
import argparse, sqlite3, os, re, time, sys
from pathlib import Path

def count_rows(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]
        counts = []
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {t};")
                n = int(cur.fetchone()[0])
            except Exception:
                n = 0
            counts.append((t, n))
        return counts
    finally:
        con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="users.db")
    ap.add_argument("--sql", default="data.sql")
    ap.add_argument("--log", default="data_test.log")
    ap.add_argument("--min_rows", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    exists_db = Path(args.db).exists()
    exists_sql = Path(args.sql).exists()

    table_counts = count_rows(args.db) if exists_db else []
    total_rows = sum(n for _, n in table_counts)

    insert_count = 0
    if exists_sql:
        try:
            txt = Path(args.sql).read_text(encoding="utf-8", errors="ignore")
            insert_count = len(re.findall(r"\binsert\b", txt, flags=re.I))
        except Exception:
            insert_count = 0

    passed = exists_db and (total_rows >= args.min_rows) and (not exists_sql or insert_count >= args.min_rows)

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"db_exists={exists_db}\n")
        for t, n in table_counts:
            f.write(f"table_{t}_rows={n}\n")
        f.write(f"total_rows={total_rows}\n")
        f.write(f"sql_exists={exists_sql}\n")
        if exists_sql:
            f.write(f"sql_insert_statements={insert_count}\n")
        f.write(f"criteria=users.db total_rows>={args.min_rows} and (data.sql INSERTs>={args.min_rows} if present)\n")
        f.write(f"result={'PASS' if passed else 'FAIL'}\n")
        f.write(f"elapsed_sec={time.time()-t0:.2f}\n")

    assert passed, "DB test failed: not enough rows or missing files"

if __name__ == "__main__":
    main()
