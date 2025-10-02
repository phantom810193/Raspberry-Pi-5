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


@dataclass
class AdDisplay:
    """Structure describing how the front-end should render an advert."""

    template_id: str
    paragraphs: tuple[str, ...]
    cta_text: Optional[str] = None
    cta_href: Optional[str] = None


def build_idle_display() -> AdDisplay:
    """Return the display payload used while waiting for a detection."""

    return AdDisplay(
        template_id="ME0000",
        paragraphs=("等待辨識中，請將視線對準鏡頭以進行辨識。",),
    )


def build_display_payload(
    member_id: Optional[str],
    transactions: Iterable[Transaction],
    *,
    message: str,
) -> AdDisplay:
    """Create a front-end payload combining transaction history and copy."""

    if not member_id:
        return build_idle_display()

    tx_list = list(transactions)
    if not tx_list:
        return AdDisplay(
            template_id="AD0000",
            paragraphs=(
                "尚未綁定商場會員代號，歡迎至服務台完成綁定。",
                message,
            ),
            cta_text="馬上註冊開啟專屬特權！",
            cta_href="#register",
        )

    template_id = _select_member_template(tx_list)
    return AdDisplay(
        template_id=template_id,
        paragraphs=(message,),
    )


def _select_member_template(transactions: list[Transaction]) -> str:
    """Choose a template for members with purchase history."""

    dessert_keywords = ("蛋糕", "塔", "布丁", "慕斯", "鬆餅",
                        "捲", "派", "甜", "奶酪", "可麗餅")
    kids_keywords = ("幼兒", "親子", "園", "兒童", "才藝")
    fitness_keywords = ("健身", "運動", "蛋白", "補給", "瑜珈", "體驗", "牛奶", "豆漿", "麥片")

    def _count(keywords: tuple[str, ...]) -> int:
        return sum(1 for tx in transactions if any(keyword in tx.item for keyword in keywords))

    dessert_hits = _count(dessert_keywords)
    kids_hits = _count(kids_keywords)
    fitness_hits = _count(fitness_keywords)

    if dessert_hits > max(kids_hits, fitness_hits):
        return "ME0001"
    if kids_hits > max(dessert_hits, fitness_hits):
        return "ME0002"
    return "ME0003"
