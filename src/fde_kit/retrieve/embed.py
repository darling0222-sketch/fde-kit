"""Embedding：在线（OpenAI 兼容）优先，离线（Hash-ngram）兜底。

离线实现不是玩具：它用中文 bigram + 英文词的哈希投影，配合 BM25 做 RRF 融合，
在“术语密集、专有名词多”的现场资料上足够撑起一条可演示、可评测的检索链路。
没有它，这个仓库就变成“必须先申请 API key 才能跑”的样板房。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..text_utils import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - 协议
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashEmbedder:
    """确定性哈希投影：无网络、无模型、同文本同向量。"""

    name = "hash-ngram"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, idx] += sign * (1.0 + np.log(tf))
        return _l2_normalize(out)


class OpenAIEmbedder:
    """任何 OpenAI 兼容的 /v1/embeddings 都能用。"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024, batch: int = 32) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch = batch
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        rows: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), self.batch):
                chunk = list(texts[start : start + self.batch])
                if not chunk:
                    continue
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": chunk},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))
                rows.extend(item["embedding"] for item in data)
        mat = np.asarray(rows, dtype=np.float32)
        self.dim = int(mat.shape[1]) if mat.size else self.dim
        return _l2_normalize(mat)


def build_embedder(settings) -> Embedder:
    if settings.embed_ready:
        base = settings.embed_base_url or settings.llm_base_url
        key = settings.embed_api_key or settings.llm_api_key
        model = settings.embed_model or "text-embedding-3-small"
        try:
            probe = OpenAIEmbedder(base, key, model, settings.embed_dim, batch=1)
            probe.embed(["ping"])  # 一次探活，失败就退回离线
            return probe
        except Exception:
            return HashEmbedder(settings.embed_dim)
    return HashEmbedder(settings.embed_dim)
