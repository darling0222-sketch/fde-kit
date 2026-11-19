"""评测执行器：加载 golden 用例 -> 逐条跑 -> 落库 -> 出汇总。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..agent.agent import DeliveryAgent
from ..retrieve.bm25 import BM25Index
from ..store import Store
from .metrics import CaseResult, CaseSpec, score_case, summarize


@dataclass
class EvalOutcome:
    run_id: str
    summary: dict
    results: list[CaseResult] = field(default_factory=list)
    mode: str = "offline"
    duration: float = 0.0
    k: int = 5

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "k": self.k,
            "mode": self.mode,
            "duration": round(self.duration, 2),
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }


def load_cases(path: str | Path) -> tuple[str, int, list[CaseSpec]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    name = str(raw.get("name", "eval"))
    k = int(raw.get("k", 5))
    cases = []
    for item in raw.get("cases") or []:
        cases.append(
            CaseSpec(
                id=str(item.get("id") or f"case-{len(cases) + 1}"),
                question=str(item["question"]),
                must_cite=list(item.get("must_cite") or []),
                keywords=list(item.get("keywords") or []),
                expect_no_answer=bool(item.get("expect_no_answer", False)),
                notes=str(item.get("notes") or ""),
            )
        )
    return name, k, cases


def run_eval(
    agent: DeliveryAgent,
    cases: list[CaseSpec],
    k: int = 5,
    store: Store | None = None,
    engine: str = "serial",
    progress=None,
) -> EvalOutcome:
    store = store or agent.store
    started = time.time()
    run_id = store.create_run("eval", {"cases": len(cases), "k": k, "engine": engine})

    # 纯词法基线：用来证明混合检索确实比单路强，而不是口头宣称
    lexical_index = BM25Index(store.all_chunks())
    results: list[CaseResult] = []

    for idx, spec in enumerate(cases, start=1):
        t0 = time.time()
        retriever_hits = agent.retriever.search(spec.question, k=k)
        titles = [_title_of(agent, hit.chunk.asset_id) for hit in retriever_hits]

        lexical = lexical_index.search(spec.question, k=k)
        lexical_titles = []
        for cid, _score in lexical:
            chunk = agent.retriever.chunk(cid)
            if chunk is not None:
                lexical_titles.append(_title_of(agent, chunk.asset_id))

        answer = agent.ask(spec.question, session_id=f"eval:{run_id}", engine=engine)
        latency = int((time.time() - t0) * 1000)
        result = score_case(spec, answer, titles, lexical_titles, latency)
        results.append(result)
        store.save_eval(run_id, spec.id, result.to_dict())
        if progress:
            progress(idx, len(cases), spec.id)

    summary = summarize(results)
    summary["mode"] = "online" if agent.llm.available else "offline"
    summary["engine"] = engine
    store.finish_run(run_id, "done", {"summary": summary, "k": k, "mode": summary["mode"]})
    return EvalOutcome(
        run_id=run_id,
        summary=summary,
        results=results,
        mode=summary["mode"],
        duration=time.time() - started,
        k=k,
    )


def _title_of(agent: DeliveryAgent, asset_id: str) -> str:
    asset = agent.store.get_asset(asset_id)
    return f"{asset.title} {asset.uri}" if asset else asset_id
