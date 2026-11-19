from __future__ import annotations

from fde_kit.eval.metrics import retrieval_metrics, summarize
from fde_kit.eval.runner import load_cases, run_eval

ROOT_CASES = "evals/golden.yaml"


def test_agent_answers_with_citations(agent):
    answer = agent.ask("A100 的整机保修期是多久？", session_id="t-agent")
    assert answer.citations
    assert answer.grounded
    assert answer.mode in {"extractive", "llm"}
    assert "24 个月" in answer.text
    assert answer.bundle is not None
    assert answer.trace and answer.trace[0]["node"] == "plan"


def test_agent_refuses_when_no_evidence(agent):
    answer = agent.ask("这批设备出口欧盟的 CE 认证编号是多少？", session_id="t-refuse")
    assert answer.citations == []
    assert answer.grounded is False
    assert "没有找到" in answer.text
    assert answer.mode == "no-evidence"


def test_agent_streaming_emits_tokens_and_answer(agent):
    events = list(agent.ask_stream("退货运费由谁承担？", session_id="t-stream"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "trace"
    assert "citations" in kinds
    assert "token" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["answer"]["citations"]


def test_streaming_matches_non_streaming_citation_set(agent):
    streamed = list(agent.ask_stream("退货运费由谁承担？", session_id="t-s2"))[-1]["answer"]
    direct = agent.ask("退货运费由谁承担？", session_id="t-s3").to_dict()
    assert {c["sid"] for c in streamed["citations"]} == {c["sid"] for c in direct["citations"]}


def test_retrieval_metrics():
    hit, mrr, recall = retrieval_metrics(["产品手册A100", "售后政策"], ["售后政策"])
    assert hit is True and mrr == 0.5 and recall == 1.0
    miss, mrr2, recall2 = retrieval_metrics(["无关文档"], ["售后政策"])
    assert miss is False and mrr2 == 0.0 and recall2 == 0.0


def test_summarize_averages():
    from fde_kit.eval.metrics import CaseResult

    rows = [
        CaseResult(id="1", question="q", hit=True, mrr=1.0, recall=1.0, keyword_coverage=1.0, grounded=True),
        CaseResult(id="2", question="q", hit=False, mrr=0.0, recall=0.0, keyword_coverage=0.0, grounded=False),
    ]
    summary = summarize(rows)
    assert summary["hit_rate"] == 0.5
    assert summary["grounded_rate"] == 0.5


def test_load_and_run_eval_on_subset(agent):
    name, k, cases = load_cases(ROOT_CASES)
    assert name and k > 0 and len(cases) >= 5
    outcome = run_eval(agent, cases[:2], k=3, store=agent.store)
    assert outcome.summary["cases"] == 2
    assert 0.0 <= outcome.summary["hit_rate"] <= 1.0
    assert len(agent.store.list_evals(outcome.run_id)) == 2
