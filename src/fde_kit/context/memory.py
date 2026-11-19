"""会话记忆：跨轮次保留“已确认的事实”和“已做的决定”。

为什么不用向量记忆：现场对话里真正要跨轮带走的只有两类东西——
客户确认过的口径（“这批是 2024 年出货，已过保”）、以及已经排除的路径。
把它们压成短句存库，比把历史对话再检索一遍便宜且可控。
"""

from __future__ import annotations

from ..store import Store


class MemoryStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    def remember(self, session_id: str, kind: str, text: str, source: str = "") -> None:
        text = " ".join(text.split()).strip()
        if not text:
            return
        # 同一条不要重复写
        for item in self.store.list_memory(session_id, limit=20):
            if item["text"] == text:
                return
        self.store.add_memory(session_id, kind, text[:400], source)

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        return self.store.list_memory(session_id, limit=limit)

    def render(self, session_id: str, limit: int = 6) -> str:
        items = self.store.list_memory(session_id, limit=limit)
        if not items:
            return ""
        lines = []
        for item in items:
            tag = {"fact": "事实", "decision": "决定", "gap": "缺口"}.get(item["kind"], item["kind"])
            lines.append(f"- [{tag}] {item['text']}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self.store._exec("DELETE FROM memory WHERE session_id = ?", (session_id,))
