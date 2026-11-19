"""上下文层：把检索结果装配成“能直接喂给模型的证据包”。

这是 fde-kit 与普通 RAG 的第二个分水岭：
普通 RAG 把 top-k 原文拼起来就发；这里要做四件事——去重、压缩、编号、预算。
预算与编号是硬要求：客户经理要能在回答里指着 [S3] 找到出处，超预算要么截断要么降级。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Citation, ContextBundle, Hit
from ..retrieve.hybrid import HybridRetriever
from ..text_utils import clip, estimate_tokens, jaccard, split_sentences, tokenize
from .memory import MemoryStore


@dataclass
class AssemblerConfig:
    top_k: int = 8
    token_budget: int = 2400
    per_chunk_max_tokens: int = 300
    dedupe_threshold: float = 0.82
    min_coverage: float = 0.35
    min_vector_score: float = 0.32


class ContextAssembler:
    def __init__(
        self,
        retriever: HybridRetriever,
        memory: MemoryStore | None = None,
        config: AssemblerConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.memory = memory
        self.config = config or AssemblerConfig()

    def build(self, query: str, session_id: str | None = None) -> ContextBundle:
        cfg = self.config
        hits = self.retriever.search(query, k=cfg.top_k)
        notes: list[str] = []

        # 拒答闸门：查询里的“稀有字/专名”都没被覆盖，且向量相似度也低，
        # 说明库里确实没有这个信息。宁可直说“没有”，也不要拿弱相关段落拼一个像样的答案。
        coverage = self.retriever.query_coverage(query, hits)
        best_vec = max((h.scores.get("vector", 0.0) for h in hits), default=0.0)
        if hits and coverage < cfg.min_coverage and best_vec < cfg.min_vector_score:
            notes.append(
                f"证据不足：查询覆盖率 {coverage:.2f} < {cfg.min_coverage}，"
                f"向量相似度 {best_vec:.2f} < {cfg.min_vector_score}"
            )
            return ContextBundle(
                query=query,
                prompt_context="",
                tokens_used=0,
                token_budget=cfg.token_budget,
                notes=notes,
            )


        query_tokens = set(tokenize(query))
        kept: list[tuple[Hit, str]] = []
        seen: list[set[str]] = []
        deduped = 0

        for hit in hits:
            tokens = set(tokenize(hit.chunk.text))
            if any(jaccard(tokens, prev) >= cfg.dedupe_threshold for prev in seen):
                deduped += 1
                continue
            seen.append(tokens)
            kept.append((hit, self._compress(hit.chunk.text, query_tokens, cfg.per_chunk_max_tokens)))

        if deduped:
            notes.append(f"近重复块去重 {deduped} 段")

        citations: list[Citation] = []
        blocks: list[str] = []
        used = 0
        dropped = 0
        for hit, text in kept:
            cost = estimate_tokens(text) + 12  # 标题与编号的开销
            if used + cost > cfg.token_budget:
                dropped += 1
                continue
            sid = f"S{len(citations) + 1}"
            asset = self.retriever.store.get_asset(hit.chunk.asset_id)
            citation = Citation(
                sid=sid,
                chunk_id=hit.chunk.id,
                asset_id=hit.chunk.asset_id,
                title=asset.title if asset else hit.chunk.asset_id,
                uri=asset.uri if asset else "",
                section=hit.chunk.section,
                quote=clip(text, 320),
                score=hit.score,
            )
            citations.append(citation)
            blocks.append(
                f"[{sid}] 《{citation.title}》" + (f" · {citation.section}" if citation.section else "") + f"\n{text}"
            )
            used += cost

        if dropped:
            notes.append(f"预算 {cfg.token_budget} tokens 已满，丢弃 {dropped} 段候选")

        ontology_hits = self._ontology_hits(query)
        if ontology_hits:
            notes.append("本体命中：" + "、".join(ontology_hits))

        if self.memory and session_id:
            memory_text = self.memory.render(session_id)
            if memory_text:
                blocks.insert(0, f"[记忆] 本次会话已确认：\n{memory_text}")
                used += estimate_tokens(memory_text)
                notes.append("已注入会话记忆")

        if not citations:
            notes.append("检索为空：该问题在当前知识库里没有可用证据")

        return ContextBundle(
            query=query,
            prompt_context="\n\n".join(blocks),
            citations=citations,
            tokens_used=used,
            token_budget=cfg.token_budget,
            notes=notes,
            ontology_hits=ontology_hits,
            dropped=dropped,
        )

    # ------------------------------------------------------------------ 压缩
    def _compress(self, text: str, query_tokens: set[str], max_tokens: int) -> str:
        if estimate_tokens(text) <= max_tokens:
            return text
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return text[: max_tokens * 2]
        scored = []
        for idx, sent in enumerate(sentences):
            toks = set(tokenize(sent))
            overlap = len(toks & query_tokens) / (len(toks) ** 0.5 + 1e-6)
            scored.append((overlap, idx, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep: list[tuple[int, str]] = []
        used = 0
        for _score, idx, sent in scored:
            cost = estimate_tokens(sent)
            if used + cost > max_tokens and keep:
                continue
            keep.append((idx, sent))
            used += cost
            if used >= max_tokens:
                break
        keep.sort(key=lambda x: x[0])
        return "".join(sent for _, sent in keep)

    def _ontology_hits(self, query: str) -> list[str]:
        ontology = self.retriever.ontology
        if ontology is None:
            return []
        return [f"{etype}:{value}" for etype, value, _w in ontology.match_query(query)]
