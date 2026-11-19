"""LangGraph 版本（可选依赖）。

`fde ask --graph` 或 `DeliveryAgent(..., engine="langgraph")` 时启用。
节点函数与串行版完全一致，这里只负责连边和重试条件。
"""

from __future__ import annotations

from typing import Any, TypedDict

from .nodes import draft, needs_retry, plan, retrieve, verify


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    query: str
    bundle: Any
    draft: str
    mode: str
    attempts: int
    cited: list[str]
    unknown_citations: list[str]
    grounded: bool
    confidence: float
    trace: list[dict]


def build_graph(ctx) -> Any:
    from langgraph.graph import END, StateGraph

    def _plan(state: AgentState) -> dict:
        return plan(state, ctx)

    def _retrieve(state: AgentState) -> dict:
        return retrieve(state, ctx)

    def _draft(state: AgentState) -> dict:
        return draft(state, ctx)

    def _verify(state: AgentState) -> dict:
        return verify(state, ctx)

    def _bump(state: AgentState) -> dict:
        return {"attempts": int(state.get("attempts", 0)) + 1}

    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("draft", _draft)
    graph.add_node("verify", _verify)
    graph.add_node("retry", _bump)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: "retry" if needs_retry(state) else END,
        {"retry": "retry", END: END},
    )
    graph.add_edge("retry", "draft")
    return graph.compile()
