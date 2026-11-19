"""领域模型：全部用 dataclass，保持 core 层不依赖 web 框架。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


def stable_id(prefix: str, *parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


@dataclass
class Asset:
    """接入层产物：一份被纳入知识库的原始资料。"""

    id: str
    source: str
    uri: str
    media_type: str
    title: str
    sha256: str
    size: int
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.meta.get("kind", self.media_type)

    def to_row(self) -> dict:
        import json

        return {
            "id": self.id,
            "source": self.source,
            "uri": self.uri,
            "media_type": self.media_type,
            "title": self.title,
            "sha256": self.sha256,
            "size": self.size,
            "text": self.text,
            "meta": json.dumps(self.meta, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row) -> Asset:
        import json

        return cls(
            id=row["id"],
            source=row["source"],
            uri=row["uri"],
            media_type=row["media_type"],
            title=row["title"],
            sha256=row["sha256"],
            size=row["size"],
            text=row["text"] or "",
            meta=json.loads(row["meta"] or "{}"),
        )


@dataclass
class Chunk:
    """编译层产物：可检索、可引用的最小证据单元。"""

    id: str
    asset_id: str
    ord: int
    text: str
    section: str = ""
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        import json

        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "ord": self.ord,
            "text": self.text,
            "section": self.section,
            "tokens": self.tokens,
            "meta": json.dumps(self.meta, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row) -> Chunk:
        import json

        return cls(
            id=row["id"],
            asset_id=row["asset_id"],
            ord=row["ord"],
            text=row["text"],
            section=row["section"] or "",
            tokens=row["tokens"] or 0,
            meta=json.loads(row["meta"] or "{}"),
        )


@dataclass
class Entity:
    """本体抽取结果，挂到 chunk 上，用于检索加权与可解释性。"""

    chunk_id: str
    type: str
    value: str
    normalized: str
    span_start: int = -1
    span_end: int = -1


@dataclass
class Hit:
    chunk: Chunk
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "asset_id": self.chunk.asset_id,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "entities": self.entities,
        }


@dataclass
class Citation:
    sid: str
    chunk_id: str
    asset_id: str
    title: str
    uri: str
    section: str
    quote: str
    score: float

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "chunk_id": self.chunk_id,
            "asset_id": self.asset_id,
            "title": self.title,
            "uri": self.uri,
            "section": self.section,
            "quote": self.quote,
            "score": round(self.score, 4),
        }


@dataclass
class ContextBundle:
    """上下文层产物：一次问答实际塞进模型的全部素材。"""

    query: str
    prompt_context: str
    citations: list[Citation] = field(default_factory=list)
    tokens_used: int = 0
    token_budget: int = 0
    notes: list[str] = field(default_factory=list)
    ontology_hits: list[str] = field(default_factory=list)
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "notes": self.notes,
            "ontology_hits": self.ontology_hits,
            "dropped": self.dropped,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False
    confidence: float = 0.0
    mode: str = "extractive"
    trace: list[dict] = field(default_factory=list)
    bundle: ContextBundle | None = None

    def to_dict(self, with_bundle: bool = True) -> dict:
        out = {
            "question": self.question,
            "text": self.text,
            "grounded": self.grounded,
            "confidence": round(self.confidence, 3),
            "mode": self.mode,
            "trace": self.trace,
            "citations": [c.to_dict() for c in self.citations],
        }
        if with_bundle and self.bundle is not None:
            out["bundle"] = self.bundle.to_dict()
        return out


CITATION_RE = re.compile(r"\[(S\d+)\]")
