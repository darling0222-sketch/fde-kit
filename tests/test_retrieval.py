from __future__ import annotations

from fde_kit.retrieve.embed import HashEmbedder
from fde_kit.retrieve.fusion import rrf


def test_hash_embedder_is_deterministic_and_normalized():
    emb = HashEmbedder(dim=256)
    a = emb.embed(["A100 整机保修 24 个月"])
    b = emb.embed(["A100 整机保修 24 个月"])
    assert a.shape == (1, 256)
    assert (a == b).all()
    assert abs(float((a**2).sum()) - 1.0) < 1e-5


def test_hash_embedder_similar_text_scores_higher():
    emb = HashEmbedder(dim=512)
    base, near, far = emb.embed(
        [
            "A100 整机保修 24 个月",
            "A100 整机保修期是 24 个月",
            "服务网点在苏州和东莞",
        ]
    )
    assert float(base @ near) > float(base @ far)


def test_rrf_prefers_documents_ranked_by_both_lists():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["d"]


def test_rrf_weights_change_order():
    base = dict(rrf([["a"], ["a", "b"], ], weights=[1.0, 1.0]))
    weighted = dict(rrf([["b"], ["a"]], weights=[3.0, 0.5]))
    assert base["a"] > base["b"]
    assert weighted["b"] > weighted["a"]


def test_hybrid_retrieval_finds_serial_number(agent):
    hits = agent.retriever.search("SN-2024-0311 是哪个客户的设备", k=5)
    assert hits
    top = hits[0]
    assert "SN-2024-0311" in top.chunk.text
    assert top.scores["bm25"] >= 0


def test_hybrid_retrieval_applies_ontology_boost(agent):
    hits = agent.retriever.search("E021 怎么处理", k=5)
    assert hits
    assert any(hit.scores.get("entity", 0) > 0 for hit in hits)


def test_query_coverage_separates_in_domain_from_out_of_domain(agent):
    good = agent.retriever.search("A100 的整机保修期是多久", k=5)
    bad = agent.retriever.search("公司年会在哪里办", k=5)
    assert agent.retriever.query_coverage("A100 的整机保修期是多久", good) > 0.5
    assert agent.retriever.query_coverage("公司年会在哪里办", bad) < 0.35
