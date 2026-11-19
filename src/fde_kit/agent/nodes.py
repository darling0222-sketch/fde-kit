"""Agent 的四个节点：规划 -> 检索 -> 起草 -> 校验。

刻意做成纯函数（state in / 状态增量 out），这样既能串行跑，
也能直接塞进 LangGraph 的 StateGraph，不需要维护两套逻辑。
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.client import LLMUnavailable, Message
from ..llm.prompts import EXTRACTIVE_HINT, STRICT_RETRY, build_answer_messages
from ..models import CITATION_RE
from ..text_utils import clip, tokenize

_LEAD_FILLER = re.compile(r"^(请问|那个|麻烦|帮我|帮忙|我想问一下|想问一下)[，,、\s]*")


def plan(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    question = state["question"].strip()
    query = _LEAD_FILLER.sub("", question).strip() or question
    ontology_terms = [f"{t}:{v}" for t, v, _w in ctx.ontology.match_query(query)] if ctx.ontology else []
    trace = state.setdefault("trace", [])
    trace.append(
        {
            "node": "plan",
            "detail": f"检索式：{clip(query, 80)}" + (f"；本体命中 {len(ontology_terms)} 项" if ontology_terms else ""),
        }
    )
    return {"query": query, "ontology_terms": ontology_terms}


def retrieve(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    bundle = ctx.assembler.build(state["query"], session_id=state.get("session_id"))
    trace = state.setdefault("trace", [])
    trace.append(
        {
            "node": "retrieve",
            "detail": f"召回 {len(bundle.citations)} 段证据 / 预算 {bundle.tokens_used}-{bundle.token_budget} tokens",
        }
    )
    for note in bundle.notes:
        trace.append({"node": "retrieve", "detail": note})
    return {"bundle": bundle}


def draft(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    bundle = state["bundle"]
    trace = state.setdefault("trace", [])
    if not bundle.citations:
        return {
            "draft": "现有资料中没有找到能回答这个问题的内容。建议补充相关文档后再问一次。",
            "mode": "no-evidence",
        }
    attempts = state.get("attempts", 0)
    if ctx.llm.available:
        messages = build_answer_messages(state["question"], bundle.prompt_context, ctx.memory_text(state))
        if attempts:
            messages.append(Message("user", STRICT_RETRY))
        try:
            text = ctx.llm.chat(messages, temperature=0.2, max_tokens=1200)
            trace.append({"node": "draft", "detail": f"模型作答：{ctx.llm.name}"})
            return {"draft": text, "mode": "llm"}
        except LLMUnavailable as exc:
            trace.append({"node": "draft", "detail": f"模型不可用，降级抽取式：{exc}"})
    text = _extractive_answer(bundle)
    trace.append({"node": "draft", "detail": "抽取式作答（离线）"})
    return {"draft": text, "mode": "extractive"}


def verify(state: dict[str, Any], ctx: Any) -> dict[str, Any]:
    bundle = state["bundle"]
    draft_text = state.get("draft", "")
    valid_ids = {c.sid for c in bundle.citations}
    cited = [sid for sid in CITATION_RE.findall(draft_text) if sid in valid_ids]
    unknown = sorted({sid for sid in CITATION_RE.findall(draft_text) if sid not in valid_ids})

    grounded = bool(cited) and (len(set(cited)) >= 1)
    top_score = max((c.score for c in bundle.citations), default=0.0)
    retrieval_conf = min(top_score / 0.045, 1.0)  # RRF 分数大约 0.01~0.045
    # 前三条证据来自几个不同资料：跨文档互相印证，比同一段话反复命中更可信
    agreement = min(len({c.asset_id for c in bundle.citations[:3]}) / 3.0, 1.0)
    confidence = (
        0.40 * retrieval_conf
        + 0.25 * agreement
        + 0.20 * float(grounded)
        + 0.15 * (1.0 if bundle.ontology_hits else 0.0)
    )
    if state.get("mode") == "extractive":
        confidence = min(confidence, 0.75)  # 抽取式没有归纳能力，不给高置信
    confidence = max(0.0, min(confidence, 0.95))

    trace = state.setdefault("trace", [])
    trace.append(
        {
            "node": "verify",
            "detail": f"引用 {len(set(cited))}/{len(valid_ids)} 有效"
            + (f"；非法引用 {','.join(unknown)}" if unknown else "")
            + f"；置信度 {confidence:.2f}",
        }
    )
    if unknown:
        trace.append({"node": "verify", "detail": "回答里出现了证据集之外的引用编号，已标记为不可信"})

    return {
        "cited": sorted(set(cited)),
        "unknown_citations": unknown,
        "grounded": grounded and not unknown,
        "confidence": confidence,
    }


def needs_retry(state: dict[str, Any]) -> bool:
    """模型作答但一个引用都没标 -> 用更严格的提示重写一次。"""
    return (
        state.get("mode") == "llm"
        and not state.get("cited")
        and state.get("attempts", 0) < 1
        and bool(state.get("bundle") and state["bundle"].citations)
    )


def _extractive_answer(bundle) -> str:
    lines = [EXTRACTIVE_HINT, ""]
    query_tokens = set(tokenize(bundle.query))
    picked = []
    for citation in bundle.citations:
        overlap = len(set(tokenize(citation.quote)) & query_tokens)
        if overlap >= 2:
            picked.append(citation)
        if len(picked) >= 3:
            break
    if not picked:
        picked = bundle.citations[:2]

    lines.append("**资料原文**")
    for citation in picked:
        body = clip(citation.quote.replace("\n", " "), 220)
        lines.append(f"- {body} [{citation.sid}]")

    rest = [c for c in bundle.citations if c not in picked][:3]
    if rest:
        lines.append("- 另有相关证据：" + "、".join(f"[{c.sid}]《{c.title}》" for c in rest))
    lines.append("")
    lines.append("**还需确认**：以上为原文摘录，未做归纳。配置模型后可自动生成结论式回答。")
    return "\n".join(lines)
