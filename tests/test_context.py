from __future__ import annotations


def test_bundle_assigns_sequential_citation_ids(agent):
    bundle = agent.assembler.build("A100 的整机保修期是多久？")
    assert bundle.citations
    assert [c.sid for c in bundle.citations] == [f"S{i + 1}" for i in range(len(bundle.citations))]
    assert bundle.tokens_used <= bundle.token_budget
    assert "[S1]" in bundle.prompt_context


def test_bundle_dedupes_near_duplicate_chunks(agent):
    bundle = agent.assembler.build("A100 的整机保修期是多久？")
    texts = [c.quote[:60] for c in bundle.citations]
    assert len(texts) == len(set(texts))


def test_bundle_respects_tight_budget(agent):
    original = agent.assembler.config.token_budget
    agent.assembler.config.token_budget = 200
    try:
        bundle = agent.assembler.build("退货运费由谁承担？")
        assert bundle.tokens_used <= 200
        assert bundle.dropped > 0
        assert any("预算" in note for note in bundle.notes)
    finally:
        agent.assembler.config.token_budget = original


def test_bundle_refuses_out_of_domain_question(agent):
    bundle = agent.assembler.build("公司年会在哪里办？")
    assert bundle.citations == []
    assert any(("证据不足" in note) or ("检索为空" in note) for note in bundle.notes)


def test_memory_is_written_and_rendered(agent):
    agent.memory.remember("test-session", "fact", "这批设备是 2024 年出货，已过保")
    rendered = agent.memory.render("test-session")
    assert "已过保" in rendered
    agent.memory.clear("test-session")
    assert agent.memory.render("test-session") == ""
