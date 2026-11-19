"""评测层：先有验收标准，再谈效果。

所有指标都能离线算（不依赖 LLM 打分），这样 CI 里能跑、客户现场也能跑：
- hit@k / MRR / recall@k：检索层有没有把该找的找出来；
- keyword_coverage：答案有没有覆盖必须出现的口径（金额、时限、型号）；
- citation_validity：回答里的引用编号是否都真实存在；
- grounded：是否至少有一处有效引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Answer


@dataclass
class CaseSpec:
    id: str
    question: str
    must_cite: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    expect_no_answer: bool = False
    notes: str = ""


@dataclass
class CaseResult:
    id: str
    question: str
    hit: bool = False
    mrr: float = 0.0
    recall: float = 0.0
    keyword_coverage: float = 0.0
    citation_validity: float = 0.0
    grounded: bool = False
    confidence: float = 0.0
    answered: str = ""
    cited: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    latency_ms: int = 0
    lexical_hit: bool = False
    lexical_mrr: float = 0.0
    expect_no_answer: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "hit": self.hit,
            "mrr": round(self.mrr, 3),
            "recall": round(self.recall, 3),
            "keyword_coverage": round(self.keyword_coverage, 3),
            "citation_validity": round(self.citation_validity, 3),
            "grounded": self.grounded,
            "confidence": round(self.confidence, 3),
            "answered": self.answered,
            "cited": self.cited,
            "retrieved": self.retrieved,
            "latency_ms": self.latency_ms,
            "lexical_hit": self.lexical_hit,
            "lexical_mrr": round(self.lexical_mrr, 3),
            "expect_no_answer": self.expect_no_answer,
        }


def _matches(text: str, patterns: list[str]) -> bool:
    text = (text or "").lower()
    return any(p.lower() in text for p in patterns if p)


def _rank_of(titles: list[str], patterns: list[str]) -> int:
    for idx, title in enumerate(titles, start=1):
        if _matches(title, patterns):
            return idx
    return 0


def retrieval_metrics(titles: list[str], must_cite: list[str]) -> tuple[bool, float, float]:
    """titles 是按相关性排序的检索结果（标题/uri）。"""
    if not must_cite:
        return True, 1.0, 1.0
    found = [p for p in must_cite if _matches(" ".join(titles), p)]
    recall = len(found) / len(must_cite)
    rank = _rank_of(titles, must_cite)
    return rank > 0, (1.0 / rank if rank else 0.0), recall


def score_case(
    spec: CaseSpec,
    answer: Answer,
    retrieved_titles: list[str],
    lexical_titles: list[str],
    latency_ms: int = 0,
) -> CaseResult:
    valid_sids = {c.sid for c in answer.citations}
    cited = [c.sid for c in answer.citations]
    hit, mrr, recall = retrieval_metrics(retrieved_titles, spec.must_cite)
    lex_hit, lex_mrr, _ = retrieval_metrics(lexical_titles, spec.must_cite)

    keywords = spec.keywords
    coverage = (sum(1 for k in keywords if k in answer.text) / len(keywords)) if keywords else 1.0

    if spec.expect_no_answer:
        hit = not answer.citations
        mrr = 1.0 if hit else 0.0
        recall = 1.0 if hit else 0.0

    return CaseResult(
        id=spec.id,
        question=spec.question,
        hit=hit,
        mrr=mrr,
        recall=recall,
        keyword_coverage=coverage,
        citation_validity=1.0 if set(cited) <= valid_sids | set() else 0.0,
        grounded=answer.grounded,
        confidence=answer.confidence,
        answered=answer.text,
        cited=cited,
        retrieved=retrieved_titles,
        latency_ms=latency_ms,
        lexical_hit=lex_hit,
        lexical_mrr=lex_mrr,
        expect_no_answer=spec.expect_no_answer,
    )


def summarize(results: list[CaseResult]) -> dict:
    n = len(results) or 1
    return {
        "cases": len(results),
        "hit_rate": round(sum(r.hit for r in results) / n, 3),
        "mrr": round(sum(r.mrr for r in results) / n, 3),
        "recall": round(sum(r.recall for r in results) / n, 3),
        "keyword_coverage": round(sum(r.keyword_coverage for r in results) / n, 3),
        "citation_validity": round(sum(r.citation_validity for r in results) / n, 3),
        "grounded_rate": round(sum(r.grounded for r in results) / n, 3),
        "avg_confidence": round(sum(r.confidence for r in results) / n, 3),
        "avg_latency_ms": int(sum(r.latency_ms for r in results) / n),
        "lexical_hit_rate": round(sum(r.lexical_hit for r in results) / n, 3),
        "lexical_mrr": round(sum(r.lexical_mrr for r in results) / n, 3),
    }
