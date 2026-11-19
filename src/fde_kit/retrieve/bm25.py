"""纯 Python BM25（Okapi）。中文按 bigram 建索引，零依赖、结果可复现。"""

from __future__ import annotations

import math
from collections import Counter

from ..models import Chunk
from ..text_utils import content_chars, tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.counters: list[Counter[str]] = []
        self.lens: list[int] = []
        df: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counter = Counter(tokens)
            self.ids.append(chunk.id)
            self.counters.append(counter)
            self.lens.append(len(tokens))
            for token in counter:
                df[token] += 1

        self.n = len(self.ids)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        self.idf = {
            token: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()
        }

        # 字级统计：只用于“查询覆盖率”判别，不参与打分
        char_df: Counter[str] = Counter()
        for chunk in chunks:
            for ch in set(content_chars(chunk.text)):
                char_df[ch] += 1
        self.char_idf = {
            ch: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for ch, freq in char_df.items()
        }
        self.max_char_idf = math.log(1 + (self.n + 0.5) / 0.5) if self.n else 1.0

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        tokens = [t for t in tokenize(query) if t in self.idf]
        if not tokens or not self.n:
            return []
        scores: list[tuple[str, float]] = []
        avgdl = self.avgdl or 1.0
        for i, counter in enumerate(self.counters):
            score = 0.0
            dl = self.lens[i]
            norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
            for token in tokens:
                tf = counter.get(token)
                if not tf:
                    continue
                score += self.idf[token] * (tf * (self.k1 + 1)) / (tf + norm)
            if score > 0:
                scores.append((self.ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
