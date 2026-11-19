"""Reciprocal Rank Fusion：不校准两路分数量纲，只融合名次。

BM25 的分数是“无上界的词频加权”，余弦是 [-1,1]，直接加权相加是错的；
RRF 用名次，天然免校准，是现场最稳的默认选择。
"""

from __future__ import annotations


def rrf(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 与 rankings 数量不一致")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
