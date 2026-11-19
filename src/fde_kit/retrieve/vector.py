"""向量索引：精确余弦（向量已在写入时 L2 归一化，内积即余弦）。"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, ids: list[str], matrix: np.ndarray) -> None:
        self.ids = ids
        self.matrix = matrix.astype(np.float32) if matrix.size else matrix
        if self.matrix.size and self.matrix.shape[0] != len(ids):
            raise ValueError("向量矩阵与 id 数量不一致")

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_vec: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        if not self.matrix.size or not len(self.ids):
            return []
        scores = self.matrix @ query_vec.astype(np.float32)
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
