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
    "你是一位廣告文案專家。請只輸出一段不超過 50 字的繁體中文促銷文案，不要附加說明、標題或建議。",
    "你是資深行銷總監。僅回傳單句 50 字內的繁體中文折扣口號，禁止出現額外註解。",
    "你是專注零售促銷的文案顧問。請提供一段 50 字內的繁體中文文案，只要內容本身，不要任何補充。",
    "你是年輕族群行銷專家。輸出 50 字以內的流行語氣廣告文案，不得附上分析或建議。",
    "你是活潑親和的社群小編。僅回傳一段不超過 50 字的繁體中文促銷句子，禁止額外說明。",
    "你是善於情感溝通的文案作者。請給出單段 50 字內的繁體中文折扣文案，不要再寫其他內容。",
]

USER_PROMPTS = [
    "請為熱銷{商品}撰寫會員獨享、限時買一送一的促銷句，50 字內，僅回文案。",
    "為{商品}撰寫折扣文案，主打買一箱享 7 折，50 字內，不能加入任何說明。",
    "請輸出 {商品} 限時特價口號，包含「即刻搶購」或「僅此一天」，限 50 字。",
    "為 {商品} 與 {搭配品項} 組合撰寫優惠句，突出早餐優惠，50 字以內即可。",
    "請寫一段 {商品} 會員點數 10 倍送的促銷文案，字數不超過 50 字。",
]


@dataclass
class AIConfig:
    provider: str = "local"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    timeout: float = 10.0
    cache_ttl: float = 120.0

    @classmethod
    def from_env(cls) -> "AIConfig":
        provider = os.getenv("AI_PROVIDER", "local")
        provider_normalized = provider.strip().lower()
        default_timeout = 35.0 if provider_normalized in {
            "local", "llama", "llama_cpp"} else 20.0

        return cls(
            provider=provider,
            base_url=os.getenv("AI_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("AI_MODEL", DEFAULT_MODEL),
            api_key=os.getenv("AI_API_KEY", DEFAULT_API_KEY),
            timeout=float(os.getenv("AI_TIMEOUT", str(default_timeout))),
            cache_ttl=float(os.getenv("AI_CACHE_TTL", "120")),
        )


@dataclass
class CacheEntry:
    value: str
    timestamp: float


class AIClient:
    """Client capable of generating promotional copy via configured provider."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig.from_env()
        provider = (self.config.provider or "").strip().lower()
        self._use_remote = provider in {"remote", "remote_ollama", "ollama"}
        self._client: Optional[OpenAI] = None
        self._remote_client = None
        self._HumanMessage = None
        self._SystemMessage = None

        if self._use_remote:
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore
                from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Remote LLM provider requires 'langchain-community' and 'langchain-core' packages"
                ) from exc
            self._remote_client = ChatOllama(model=self.config.model, base_url=self.config.base_url)
            self._HumanMessage = HumanMessage
            self._SystemMessage = SystemMessage
        else:
            self._client = OpenAI(base_url=self.config.base_url,
                                  api_key=self.config.api_key)
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

        message: Optional[str] = None

        if self._use_remote and self._remote_client is not None:
            try:
                response = self._remote_client.invoke([
                    self._SystemMessage(content=system_prompt),
                    self._HumanMessage(content=user_prompt),
                ])
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Remote AI generation failed for member %s: %s", member_id, exc)
                response = None
            if response is not None:
                message = getattr(response, "content", None)
        else:
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    timeout=self.config.timeout,
                    stop=["<end_of_turn>", "文案重點說明", "---", "<|eot_id|>"],
                )
            except OpenAIError as exc:
                LOGGER.warning(
                    "AI generation failed for member %s: %s", member_id, exc)
                return None

            choice = response.choices[0] if response.choices else None
            if choice is not None:
                chat_message = getattr(choice, "message", None)
                if chat_message is not None:
                    message = getattr(chat_message, "content", None)
        if message:
            message = message.replace("<end_of_turn>", "").strip()
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
