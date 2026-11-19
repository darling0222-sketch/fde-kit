"""交付 Agent：把四个节点串起来，并负责降级、记忆写入与流式输出。

两种执行引擎：
- serial（默认）：零依赖，逻辑与图完全一致，测试与离线演示都走这条；
- langgraph：装了 `fde-kit[langchain]` 后可切换，节点函数复用，不维护第二套逻辑。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..compile.ontology import Ontology
from ..config import Settings
from ..context.assembler import AssemblerConfig, ContextAssembler
from ..context.memory import MemoryStore
from ..llm.client import LLMClient, LLMUnavailable
from ..llm.prompts import build_answer_messages
from ..models import Answer, ContextBundle
from ..retrieve.embed import Embedder, build_embedder
from ..retrieve.hybrid import HybridRetriever
from ..store import Store
from ..text_utils import clip
from .nodes import draft, needs_retry, plan, retrieve, verify

MEMORY_TAG = "decision"


@dataclass
class AgentContext:
    store: Store
    settings: Settings
    ontology: Ontology
    retriever: HybridRetriever
    assembler: ContextAssembler
    llm: LLMClient
    memory: MemoryStore

    def memory_text(self, state: dict[str, Any]) -> str:
        session_id = state.get("session_id") or "default"
        return self.memory.render(session_id)


class DeliveryAgent:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        ontology: Ontology,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.ontology = ontology
        self.embedder = embedder or build_embedder(settings)
        self.llm = llm or LLMClient(settings)
        self.memory = MemoryStore(store)
        self.retriever = HybridRetriever(store, self.embedder, ontology)
        self.assembler = ContextAssembler(
            self.retriever,
            self.memory,
            AssemblerConfig(
                top_k=settings.top_k,
                token_budget=settings.token_budget,
                min_coverage=settings.min_coverage,
                min_vector_score=settings.min_vector_score,
            ),
        )
        self.ctx = AgentContext(
            store=store,
            settings=settings,
            ontology=ontology,
            retriever=self.retriever,
            assembler=self.assembler,
            llm=self.llm,
            memory=self.memory,
        )
        self._graph = None

    # ------------------------------------------------------------------ 查询
    def ask(self, question: str, session_id: str = "default", engine: str = "serial") -> Answer:
        if engine == "langgraph":
            return self._ask_graph(question, session_id)
        return self._ask_serial(question, session_id)

    def _ask_serial(self, question: str, session_id: str) -> Answer:
        state: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
            "trace": [],
            "attempts": 0,
        }
        state.update(plan(state, self.ctx))
        state.update(retrieve(state, self.ctx))
        state.update(draft(state, self.ctx))
        state.update(verify(state, self.ctx))

        if needs_retry(state):
            state["attempts"] = 1
            state.update(draft(state, self.ctx))
            state.update(verify(state, self.ctx))

        return self._finish(state)

    def _ask_graph(self, question: str, session_id: str) -> Answer:
        graph = self._compiled_graph()
        if graph is None:
            return self._ask_serial(question, session_id)
        state = graph.invoke(
            {"question": question, "session_id": session_id, "trace": [], "attempts": 0}
        )
        return self._finish(state)

    def _compiled_graph(self):
        if self._graph is not None:
            return self._graph
        try:
            from .graph_langgraph import build_graph
        except Exception:
            return None
        try:
            self._graph = build_graph(self.ctx)
        except Exception:
            self._graph = None
        return self._graph

    # ------------------------------------------------------------------ 流式
    def ask_stream(self, question: str, session_id: str = "default") -> Iterator[dict[str, Any]]:
        state: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
            "trace": [],
            "attempts": 0,
        }
        state.update(plan(state, self.ctx))
        yield {"type": "trace", "node": "plan", "detail": state["trace"][-1]["detail"]}

        state.update(retrieve(state, self.ctx))
        bundle: ContextBundle = state["bundle"]
        for step in state["trace"][1:]:
            yield {"type": "trace", "node": step["node"], "detail": step["detail"]}
        yield {"type": "citations", "citations": [c.to_dict() for c in bundle.citations]}

        if self.llm.available and bundle.citations:
            messages = build_answer_messages(question, bundle.prompt_context, self.ctx.memory_text(state))
            pieces: list[str] = []
            try:
                for piece in self.llm.stream(messages, temperature=0.2, max_tokens=1200):
                    pieces.append(piece)
                    yield {"type": "token", "text": piece}
                state["draft"] = "".join(pieces)
                state["mode"] = "llm"
            except LLMUnavailable as exc:
                yield {"type": "trace", "node": "draft", "detail": f"流式中断，降级抽取式：{exc}"}
                state.update(draft(state, self.ctx))
                for piece in _fake_stream(state["draft"]):
                    yield {"type": "token", "text": piece}
        else:
            state.update(draft(state, self.ctx))
            for piece in _fake_stream(state["draft"]):
                yield {"type": "token", "text": piece}

        state.update(verify(state, self.ctx))
        if needs_retry(state):
            state["attempts"] = 1
            state.update(draft(state, self.ctx))
            state.update(verify(state, self.ctx))
            yield {"type": "token", "text": "\n\n" + state["draft"]}

        answer = self._finish(state)
        yield {"type": "done", "answer": answer.to_dict()}

    # ------------------------------------------------------------------ 收尾
    def _finish(self, state: dict[str, Any]) -> Answer:
        bundle: ContextBundle = state["bundle"]
        text = state.get("draft", "")
        if state.get("unknown_citations"):
            # 把不存在的引用编号去掉，避免用户点开发现是假的
            for bad in state["unknown_citations"]:
                text = text.replace(f"[{bad}]", "")
        cited = set(state.get("cited", []))
        citations = [c for c in bundle.citations if c.sid in cited] or bundle.citations[:3]

        if state.get("mode") in {"llm", "extractive"} and cited:
            self.memory.remember(
                state.get("session_id") or "default",
                MEMORY_TAG,
                f"已答：{clip(state['question'], 60)} → {clip(text, 160)}",
                source=",".join(sorted(cited)),
            )

        return Answer(
            question=state["question"],
            text=text,
            citations=citations,
            grounded=bool(state.get("grounded")),
            confidence=float(state.get("confidence", 0.0)),
            mode=str(state.get("mode", "extractive")),
            trace=state.get("trace", []),
            bundle=bundle,
        )


def _fake_stream(text: str, size: int = 18) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]
