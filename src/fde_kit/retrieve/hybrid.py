"""混合检索器：BM25 + 向量 -> RRF -> 本体加权 -> MMR 去重。

四个动作各自解决一个现场问题：
- BM25：客户嘴里的型号、故障码必须字面命中（向量对稀有串不敏感）；
- 向量：客户换一种说法（“开不了机” vs “无法启动”）也要能召回；
- RRF：两路分数量纲不同，只融合名次；
- 本体加权 + MMR：命中本体的块优先，同时避免同一段话被切成 5 块占满名额。
"""

from __future__ import annotations

import numpy as np

from ..compile.ontology import Ontology
from ..models import Hit
from ..store import Store
from ..text_utils import content_chars, jaccard, normalize_value, tokenize
from .bm25 import BM25Index
from .embed import Embedder
from .fusion import rrf
from .vector import VectorIndex


class HybridRetriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        ontology: Ontology | None = None,
        candidate_k: int = 50,
        mmr_lambda: float = 0.72,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.ontology = ontology
        self.candidate_k = candidate_k
        self.mmr_lambda = mmr_lambda

        self._revision = -1
        self._bm25: BM25Index | None = None
        self._vector: VectorIndex | None = None
        self._chunks: dict[str, object] = {}
        self._char_sets: dict[str, set[str]] = {}
        self._entity_map: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ 索引
    def refresh(self, force: bool = False) -> None:
        revision = self.store.revision()
        if not force and revision == self._revision and self._bm25 is not None:
            return
        chunks = self.store.all_chunks()
        self._chunks = {c.id: c for c in chunks}
        self._char_sets = {c.id: set(content_chars(c.text)) for c in chunks}
        self._bm25 = BM25Index(chunks)
        ids, mat = self.store.load_vectors()
        if ids and mat.size:
            keep_idx = [i for i, cid in enumerate(ids) if cid in self._chunks]
            keep_ids = [ids[i] for i in keep_idx]
            keep_mat = mat[keep_idx] if keep_idx else np.zeros((0, 0), dtype=np.float32)
            self._vector = VectorIndex(keep_ids, keep_mat) if keep_ids else None
        else:
            self._vector = None
        self._entity_map = self.store.entity_index()
        self._revision = revision

    @property
    def index_info(self) -> dict:
        return {
            "chunks": len(self._chunks),
            "vectors": len(self._vector) if self._vector else 0,
            "revision": self._revision,
            "embedder": getattr(self.embedder, "name", "unknown"),
        }

    def chunk(self, chunk_id: str):
        self.refresh()
        return self._chunks.get(chunk_id)

    def query_coverage(self, query: str, hits: list[Hit], top: int = 5) -> float:
        """IDF 加权的查询覆盖率：只有“稀有词”被覆盖才算数。

        为什么不用原始 BM25 分数当闸门：BM25 无上界，长短块不可比，
        “设备 / 我们 / 是否”这类高频词也能把分数顶上去。
        把每个查询词的 idf 当权重，看最相关的那一块覆盖了多少权重，才可比、可调。
        """
        if not self._bm25 or not hits:
            return 0.0
        q_chars = content_chars(query)
        if not q_chars:
            return 0.0
        bm25 = self._bm25

        def idf(ch: str) -> float:
            # 语料里根本没出现过的字，按 df=0 计 idf——它恰恰最有信息量
            return bm25.char_idf.get(ch, bm25.max_char_idf)

        total = sum(idf(ch) for ch in q_chars) or 1e-9
        best = 0.0
        for hit in hits[:top]:
            chunk_chars = self._char_sets.get(hit.chunk.id, set())
            matched = sum(idf(ch) for ch in q_chars if ch in chunk_chars)
            best = max(best, matched / total)
        return best

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, k: int = 8, use_ontology: bool = True) -> list[Hit]:
        self.refresh()
        if not self._chunks or self._bm25 is None:
            return []

        lexical = self._bm25.search(query, self.candidate_k)
        dense: list[tuple[str, float]] = []
        if self._vector is not None:
            qvec = self.embedder.embed([query])[0]
            dense = self._vector.search(qvec, self.candidate_k)

        fused = rrf(
            [[doc_id for doc_id, _ in lexical], [doc_id for doc_id, _ in dense]],
            weights=[1.0, 1.0],
        )
        if not fused:
            return []

        bm25_scores = dict(lexical)
        dense_scores = dict(dense)
        boosts = self._query_boosts(query) if use_ontology else {}

        scored: list[tuple[float, str, dict[str, float]]] = []
        for doc_id, base in fused[: self.candidate_k * 2]:
            chunk = self._chunks.get(doc_id)
            if chunk is None:
                continue
            entity_boost = 0.0
            if boosts:
                chunk_entities = {
                    normal for normal in (e.split(":", 1)[-1] for e in self._entity_map.get(doc_id, []))
                }
                for norm, weight in boosts.items():
                    if norm in chunk_entities or norm in normalize_value(chunk.text):
                        entity_boost = max(entity_boost, weight)
            score = base * (1.0 + entity_boost)
            scored.append(
                (
                    score,
                    doc_id,
                    {
                        "bm25": bm25_scores.get(doc_id, 0.0),
                        "vector": dense_scores.get(doc_id, 0.0),
                        "rrf": base,
                        "entity": entity_boost,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        # 先取 k*3 候选做 MMR，避免多样性选择范围过窄
        pool = scored[: max(k * 3, k)]
        selected = self._mmr(pool, k)

        hits: list[Hit] = []
        for score, doc_id, parts in selected:
            chunk = self._chunks[doc_id]
            hits.append(
                Hit(
                    chunk=chunk,  # type: ignore[arg-type]
                    score=score,
                    scores=parts,
                    entities=self._entity_map.get(doc_id, []),
                )
            )
        return hits

    def _query_boosts(self, query: str) -> dict[str, float]:
        if self.ontology is None:
            return {}
        boosts: dict[str, float] = {}
        for _etype, value, weight in self.ontology.match_query(query):
            norm = normalize_value(value)
            if norm:
                boosts[norm] = max(boosts.get(norm, 0.0), weight)
        return boosts

    def _mmr(
        self, pool: list[tuple[float, str, dict[str, float]]], k: int
    ) -> list[tuple[float, str, dict[str, float]]]:
        if len(pool) <= k:
            return pool
        token_sets = {doc_id: set(tokenize(self._chunks[doc_id].text)) for _, doc_id, _ in pool}  # type: ignore[union-attr]
        max_score = max(score for score, _, _ in pool) or 1.0
        selected: list[tuple[float, str, dict[str, float]]] = []
        candidates = list(pool)
        while candidates and len(selected) < k:
            best = None
            best_value = -1e9
            for item in candidates:
                score, doc_id, _ = item
                rel = score / max_score
                sim = 0.0
                if selected:
                    sim = max(jaccard(token_sets[doc_id], token_sets[s[1]]) for s in selected)
                value = self.mmr_lambda * rel - (1 - self.mmr_lambda) * sim
                if value > best_value:
                    best_value = value
                    best = item
            if best is None:
                break
            selected.append(best)
            candidates.remove(best)
        return selected
