"""Generate template-based advertisement text for kiosk display."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional


@dataclass
class Transaction:
    item: str
    amount: float
    timestamp: str

    def formatted_time(self) -> str:
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return self.timestamp


def _pick_anchor_transaction(transactions: Iterable[Transaction]) -> Optional[Transaction]:
    for tx in transactions:
        return tx
    return None


def generate_message(member_id: str, transactions: Iterable[Transaction]) -> str:
    """Return an advertisement string based on ``transactions``."""
    anchor = _pick_anchor_transaction(transactions)
    member_alias = member_id.replace("member-", "會員ID-")
    if anchor is None:
        return f"{member_alias}，首次光臨！立即領取全館95折優惠。"

    discount = "9折" if anchor.amount >= 80 else "85折"
    return (
        f"{member_alias}，上次於{anchor.formatted_time()}購買「{anchor.item}」，"
        f"今日同品項{discount}，快來加購！"
    )
