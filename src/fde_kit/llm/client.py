"""LLM 客户端：OpenAI 兼容协议，支持流式。

模型连接失败不抛到上层打断业务——统一抛 LLMUnavailable，由 agent 降级到
“抽取式回答 + 证据摘录”。现场没网、客户没批预算，系统也得能演示。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import httpx

from ..config import Settings


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMClient:
    def __init__(self, settings: Settings, timeout: float = 90.0) -> None:
        self.settings = settings
        self.timeout = timeout
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key

    @property
    def available(self) -> bool:
        return self.settings.has_llm

    @property
    def name(self) -> str:
        return self.model if self.available else "offline-extractive"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise LLMUnavailable("未配置 LLM（FDE_LLM_BASE_URL / FDE_LLM_API_KEY / FDE_LLM_MODEL）")
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"模型调用失败：{exc}") from exc
        except (KeyError, ValueError) as exc:
            raise LLMUnavailable(f"模型返回格式异常：{exc}") from exc
        return (data["choices"][0]["message"]["content"] or "").strip()

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Iterator[str]:
        if not self.available:
            raise LLMUnavailable("未配置 LLM")
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            yield piece
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"模型流式调用失败：{exc}") from exc
