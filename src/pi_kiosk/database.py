"""SQLite data access utilities for the Raspberry Pi face advertising kiosk."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

Connection = sqlite3.Connection

MEMBER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS members (
    id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL
);
"""

TRANSACTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id TEXT NOT NULL,
    item TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(member_id) REFERENCES members(id)
);
"""

SAMPLE_TRANSACTIONS: Tuple[Tuple[str, str, float, str], ...] = (
    ("member-001", "有機牛奶", 85.0, "2024-05-02T09:30:00"),
    ("member-001", "手工優格", 120.0, "2024-05-15T14:15:00"),
    ("member-002", "燕麥片", 65.0, "2024-04-28T11:42:00"),
    ("member-003", "冷萃咖啡", 90.0, "2024-05-01T08:20:00"),
    ("member-003", "義式麵包", 55.0, "2024-05-12T18:45:00"),
)


class MemberNotFoundError(RuntimeError):
    """Raised when attempting to operate on a non-existent member."""


def connect(db_path: Path) -> Connection:
    """Open a connection to ``db_path`` ensuring that the parent directory exists."""
    db_path = Path(db_path)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(conn: Connection) -> None:
    """Create required tables if they are missing."""
    with conn:
        conn.execute(MEMBER_TABLE_SQL)
        conn.execute(TRANSACTION_TABLE_SQL)


def ensure_sample_transactions(conn: Connection, sample_transactions: Iterable[Tuple[str, str, float, str]] = SAMPLE_TRANSACTIONS) -> None:
    """Populate the ``transactions`` table with demo rows when empty."""
    cursor = conn.execute("SELECT COUNT(*) FROM transactions")
    (count,) = cursor.fetchone()
    if count:
        return

    with conn:
        conn.executemany(
            "INSERT INTO transactions (member_id, item, amount, timestamp) VALUES (?, ?, ?, ?)",
            list(sample_transactions),
        )


def register_member(conn: Connection, member_id: str, first_seen: str) -> None:
    """Insert the member if it does not exist."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO members (id, first_seen) VALUES (?, ?)",
            (member_id, first_seen),
        )


def get_member(conn: Connection, member_id: str) -> sqlite3.Row | None:
    """Return the member row if present."""
    cursor = conn.execute("SELECT id, first_seen FROM members WHERE id = ?", (member_id,))
    return cursor.fetchone()


def get_transactions(conn: Connection, member_id: str) -> List[sqlite3.Row]:
    """Return transactions for ``member_id`` ordered from newest to oldest."""
    cursor = conn.execute(
        "SELECT item, amount, timestamp FROM transactions WHERE member_id = ? ORDER BY timestamp DESC",
        (member_id,),
    )
    return list(cursor.fetchall())


def upsert_transactions(conn: Connection, member_id: str, records: Iterable[Tuple[str, float, str]]) -> None:
    """Overwrite demo transactions for ``member_id`` with custom ``records``."""
    with conn:
        conn.execute("DELETE FROM transactions WHERE member_id = ?", (member_id,))
        conn.executemany(
            "INSERT INTO transactions (member_id, item, amount, timestamp) VALUES (?, ?, ?, ?)",
            [(member_id, item, amount, timestamp) for item, amount, timestamp in records],
        )


def insert_transactions(conn: Connection, member_id: str, records: Iterable[Tuple[str, float, str]]) -> int:
    """Insert transactions for an existing member.

    Raises ``MemberNotFoundError`` if the member is missing.
    """

    cursor = conn.execute("SELECT 1 FROM members WHERE id = ?", (member_id,))
    if cursor.fetchone() is None:
        raise MemberNotFoundError(f"Member {member_id} not found")

    prepared: List[Tuple[str, float, str]] = [(item, float(amount), timestamp) for item, amount, timestamp in records]
    if not prepared:
        return 0

    with conn:
        conn.executemany(
            "INSERT INTO transactions (member_id, item, amount, timestamp) VALUES (?, ?, ?, ?)",
            [(member_id, item, amount, timestamp) for item, amount, timestamp in prepared],
        )
    return len(prepared)
