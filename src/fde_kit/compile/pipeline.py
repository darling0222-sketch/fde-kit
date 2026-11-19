"""知识编译管线：RawDoc -> Asset -> Chunk -> Entity -> Vector。

一次编译 = 一份可交付的“知识库快照”。编译结果本身就是交付物：
进了多少资料、失败几份、切了多少块、本体哪几类没覆盖，全部要能写进报告。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from ..ingest.base import RawDoc
from ..models import Asset, Entity, stable_id
from ..retrieve.embed import Embedder
from ..store import Store
from ..text_utils import normalize_value
from .chunk import ChunkingConfig, chunks_from_text
from .ontology import Ontology
from .parse import parse_doc

Progress = Callable[[str, int, int, str], None]


@dataclass
class CompileResult:
    assets_total: int = 0
    assets_new: int = 0
    assets_skipped: int = 0
    assets_failed: int = 0
    chunks: int = 0
    entities: int = 0
    tokens: int = 0
    vectors: int = 0
    duration: float = 0.0
    warnings: list[dict] = field(default_factory=list)
    embedder: str = ""

    def to_dict(self) -> dict:
        return {
            "assets_total": self.assets_total,
            "assets_new": self.assets_new,
            "assets_skipped": self.assets_skipped,
            "assets_failed": self.assets_failed,
            "chunks": self.chunks,
            "entities": self.entities,
            "tokens": self.tokens,
            "vectors": self.vectors,
            "duration": round(self.duration, 2),
            "embedder": self.embedder,
            "warnings": self.warnings,
        }


class Compiler:
    def __init__(
        self,
        store: Store,
        ontology: Ontology,
        embedder: Embedder,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.store = store
        self.ontology = ontology
        self.embedder = embedder
        self.chunking = chunking or ChunkingConfig()

    # ------------------------------------------------------------------ 主流程
    def compile(
        self,
        docs: Iterable[RawDoc],
        reset: bool = False,
        progress: Progress | None = None,
    ) -> CompileResult:
        started = time.time()
        result = CompileResult(embedder=getattr(self.embedder, "name", "unknown"))
        docs = list(docs)
        result.assets_total = len(docs)

        if reset:
            self._reset()
            self._notify(progress, "reset", 0, len(docs), "已清空旧快照")

        for i, doc in enumerate(docs, start=1):
            self._compile_doc(doc, result)
            self._notify(progress, "parse", i, len(docs), f"{doc.media_type}:{doc.title}")

        self._notify(progress, "embed", 0, 1, "开始生成向量")
        result.vectors = self._embed_missing()
        result.tokens = sum(c.tokens for c in self.store.all_chunks())
        result.chunks = self.store.count_chunks()
        result.entities = sum(self.store.entity_stats().values())
        self.store.bump_revision()

        result.duration = time.time() - started
        self._notify(progress, "done", 1, 1, "编译完成")
        return result

    def _compile_doc(self, doc: RawDoc, result: CompileResult) -> None:
        sha = hashlib.sha256(doc.data).hexdigest()
        existing = self.store.find_asset_by_sha(sha)
        if existing is not None:
            if self.store.chunks_for_asset(existing.id):
                result.assets_skipped += 1
                return
            # 之前解析失败过，重新试一次
            self.store.delete_asset(existing.id)

        parsed = parse_doc(doc)
        if not parsed.text.strip():
            result.assets_failed += 1
            result.warnings.append(
                {"uri": doc.uri, "media_type": doc.media_type, "message": "；".join(parsed.warnings) or "无文本内容"}
            )
            return

        asset_id = stable_id("as", sha)
        meta = dict(doc.meta)
        meta.update(parsed.meta)
        meta.setdefault("kind", "table" if doc.media_type in {"csv", "jsonl", "json"} else "doc")
        asset = Asset(
            id=asset_id,
            source=doc.source,
            uri=doc.uri,
            media_type=doc.media_type,
            title=doc.title or doc.filename,
            sha256=sha,
            size=doc.size,
            text=parsed.text,
            meta=meta,
        )
        self.store.upsert_asset(asset)

        chunks = chunks_from_text(
            asset_id,
            parsed.text,
            title=asset.title,
            config=self.chunking,
            tabular=meta.get("kind") == "table",
        )
        if not chunks:
            result.assets_failed += 1
            result.warnings.append({"uri": doc.uri, "media_type": doc.media_type, "message": "切块为空"})
            return

        self.store.replace_asset_chunks(asset_id, chunks)

        entities: list[Entity] = []
        for chunk in chunks:
            for etype, value, start, end in self.ontology.extract(chunk.text):
                entities.append(
                    Entity(
                        chunk_id=chunk.id,
                        type=etype,
                        value=value,
                        normalized=normalize_value(value),
                        span_start=start,
                        span_end=end,
                    )
                )
        if entities:
            self.store.add_entities(entities)

        result.assets_new += 1
        for warning in parsed.warnings:
            result.warnings.append({"uri": doc.uri, "media_type": doc.media_type, "message": warning})

    # ------------------------------------------------------------------ 向量
    def _embed_missing(self, batch: int = 32) -> int:
        chunks = self.store.all_chunks()
        if not chunks:
            return 0
        have_ids, _ = self.store.load_vectors()
        have = set(have_ids)
        todo = [c for c in chunks if c.id not in have]
        if not todo:
            return 0
        written = 0
        for start in range(0, len(todo), batch):
            part = todo[start : start + batch]
            vectors = self.embedder.embed([c.text for c in part])
            if vectors.shape[0] != len(part):
                continue
            self.store.set_vectors({c.id: vectors[i] for i, c in enumerate(part)})
            written += len(part)
        return written

    def _reset(self) -> None:
        for asset in self.store.list_assets():
            self.store.delete_asset(asset.id)

    @staticmethod
    def _notify(progress: Progress | None, stage: str, done: int, total: int, message: str) -> None:
        if progress:
            try:
                progress(stage, done, total, message)
            except Exception:
                pass


def embedding_matrix(store: Store) -> tuple[list[str], np.ndarray]:
    return store.load_vectors()
