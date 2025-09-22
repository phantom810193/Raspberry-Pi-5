"""Pluggable AI client for generating advertisement copy."""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from openai import OpenAI, OpenAIError

LOGGER = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "LLaMA_CPP"
DEFAULT_API_KEY = "sk-no-key-required"


SYSTEM_PROMPTS = [
    "你是一位經驗豐富的廣告文案專家。請為折扣促銷，撰寫簡潔、有力的繁體中文廣告詞，並控制在 50 字左右。",
    "你是一位擅長創意行銷的文案大師。請為促銷折扣活動，設計充滿吸引力與情感共鳴的繁體中文廣告文案，長度約 50 字。",
    "你是一位專注於顧客利益的行銷顧問。請將促銷折扣轉化為令人心動的價值，並以約 50 字的繁體中文廣告語呈現。",
    "你是一位專門鎖定年輕族群的行銷顧問。請為折扣促銷，撰寫符合流行語氣、能引發共鳴的繁體中文廣告文案，長度約 50 字回答。",
    "你是一位熱愛分享的社群小編。請用活潑、具備親和力的繁體中文口吻，為折扣活動撰寫互動式廣告貼文，字數控制在 50 字以內。",
    "你是一位擅長用故事打動人心的文案作者。請描寫一個顧客的日常困擾，並以促銷折扣作為解決方案，用繁體中文呈現，長度約 50 字。",
]

USER_PROMPTS = [
    "請為一款熱銷{商品}撰寫廣告文案，強調會員獨享、限時買一送一，鼓勵顧客立即行動。",
    "為{商品}撰寫促銷文案。強調買一箱享 7 折，並使用「囤貨就是省」等標語，激發顧客的購物慾。",
    "想像你正在為熱銷{商品}撰寫限時特價的文案。請用「即刻搶購」或「僅此一天」等詞語，讓顧客感受到不買可惜。",
    "為{商品}撰寫組合價文案。強調「{商品}搭配{搭配品項}，早餐輕鬆享」，並標示划算價格，吸引顧客。",
    "為指定{商品}撰寫促銷文案，強調「會員點數 10 倍送」，讓顧客覺得除了折扣外，還能獲得額外回饋。",
]


@dataclass
class AIConfig:
    provider: str = "local"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    timeout: float = 10.0
    cache_ttl: float = 60.0

    @classmethod
    def from_env(cls) -> "AIConfig":
        return cls(
            provider=os.getenv("AI_PROVIDER", "local"),
            base_url=os.getenv("AI_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("AI_MODEL", DEFAULT_MODEL),
            api_key=os.getenv("AI_API_KEY", DEFAULT_API_KEY),
            timeout=float(os.getenv("AI_TIMEOUT", "10")),
            cache_ttl=float(os.getenv("AI_CACHE_TTL", "60")),
        )


@dataclass
class CacheEntry:
    value: str
    timestamp: float


class AIClient:
    """Client capable of generating promotional copy via configured provider."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig.from_env()
        self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        self._cache: Dict[tuple[str, str], CacheEntry] = {}

    def _select_prompts(self, context: Dict[str, str]) -> tuple[str, str]:
        system_prompt = random.choice(SYSTEM_PROMPTS)
        template = random.choice(USER_PROMPTS)
        user_prompt = template.format(**context)
        return system_prompt, user_prompt

    def _cache_key(self, member_id: str, context: Dict[str, str]) -> tuple[str, str]:
        items = tuple(sorted(context.items()))
        return member_id, repr(items)

    def _get_cached(self, key: tuple[str, str]) -> Optional[str]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry.timestamp > self.config.cache_ttl:
            del self._cache[key]
            return None
        return entry.value

    def _store_cache(self, key: tuple[str, str], value: str) -> None:
        self._cache[key] = CacheEntry(value=value, timestamp=time.time())

    def generate(self, member_id: str, context: Dict[str, str]) -> Optional[str]:
        key = self._cache_key(member_id, context)
        cached = self._get_cached(key)
        if cached:
            return cached

        system_prompt, user_prompt = self._select_prompts(context)
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self.config.timeout,
            )
        except OpenAIError as exc:
            LOGGER.warning("AI generation failed for member %s: %s", member_id, exc)
            return None

        message = response.choices[0].message.get("content") if response.choices else None
        if message:
            self._store_cache(key, message)
        return message


def build_context_from_transactions(transactions: Iterable[Dict[str, str]]) -> Dict[str, str]:
    """Build prompt context from recent transactions."""
    context: Dict[str, str] = {}
    items = list(transactions)
    if not items:
        return context

    primary = items[0]
    context["商品"] = primary.get("item", "熱銷商品")
    context["價格"] = str(primary.get("amount", ""))

    if len(items) > 1:
        secondary = items[1]
        context["搭配品項"] = secondary.get("item", context["商品"])
    else:
        context["搭配品項"] = primary.get("item", "人氣組合")

    return context
