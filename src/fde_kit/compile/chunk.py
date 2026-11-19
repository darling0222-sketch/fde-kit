"""切块：按标题分节 + 按段落装箱。

刻意不用通用 TextSplitter：现场资料里标题层级就是最强的语义边界，
按标题切再装箱，比按字符窗口滑切的可解释性高得多（引用能落到“哪一节”）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Chunk, stable_id
from ..text_utils import estimate_tokens, split_sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class ChunkingConfig:
    target_tokens: int = 360
    max_tokens: int = 560
    min_tokens: int = 40
    overlap_tokens: int = 60


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """返回 [(section_title, [paragraph, ...]), ...]。第一个 section 可能无标题。"""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush_para() -> None:
        if buf:
            para = "\n".join(buf).strip()
            if para:
                paragraphs.append(para)
            buf.clear()

    def flush_section() -> None:
        flush_para()
        sections.append((current_title, list(paragraphs)))
        paragraphs.clear()

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush_section()
            current_title = m.group(2).strip()
            continue
        if not line.strip():
            flush_para()
            continue
        buf.append(line.rstrip())
    flush_section()

    return [(title, paras) for title, paras in sections if paras]


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    if estimate_tokens(paragraph) <= max_tokens:
        return [paragraph]
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in split_sentences(paragraph):
        st = estimate_tokens(sent)
        if buf and size + st > max_tokens:
            pieces.append("".join(buf).strip())
            buf, size = [], 0
        if st > max_tokens:  # 单句就超限，硬切
            for i in range(0, len(sent), max_tokens):
                pieces.append(sent[i : i + max_tokens])
            continue
        buf.append(sent)
        size += st
    if buf:
        pieces.append("".join(buf).strip())
    return [p for p in pieces if p]


def _tail_overlap(paragraphs: list[str], budget: int) -> str:
    if budget <= 0 or not paragraphs:
        return ""
    tail: list[str] = []
    used = 0
    for para in reversed(paragraphs):
        for sent in reversed(split_sentences(para)):
            st = estimate_tokens(sent)
            if used + st > budget:
                break
            tail.append(sent)
            used += st
        if used >= budget:
            break
    return "".join(reversed(tail)).strip()


def chunks_from_text(
    asset_id: str,
    text: str,
    title: str = "",
    config: ChunkingConfig | None = None,
    tabular: bool = False,
) -> list[Chunk]:
    cfg = config or ChunkingConfig()
    # 表格/工单这类结构化文本不共享重叠窗口：按字段切开的重叠会把一行劈成两半，
    # 引用时就会出现“状态=已关闭处理=…”这种无法阅读的拼接。
    overlap = 0 if tabular else cfg.overlap_tokens
    chunks: list[Chunk] = []
    ordinal = 0

    for section_title, paragraphs in split_sections(text):
        expanded: list[str] = []
        for para in paragraphs:
            expanded.extend(_split_long_paragraph(para, cfg.max_tokens))

        buf: list[str] = []
        size = 0

        def flush(section_title: str = section_title) -> None:
            nonlocal buf, size, ordinal
            if not buf:
                return
            body = "\n\n".join(buf).strip()
            if estimate_tokens(body) < cfg.min_tokens and chunks:
                # 太短的尾巴并回上一块，避免产生无意义的碎片
                prev = chunks[-1]
                if estimate_tokens(prev.text) + estimate_tokens(body) <= cfg.max_tokens:
                    prev.text = prev.text + "\n\n" + body
                    prev.tokens = estimate_tokens(prev.text)
                    buf, size = [], 0
                    return
            section = section_title or title or "正文"
            chunks.append(
                Chunk(
                    id=stable_id("ck", asset_id, str(ordinal), body[:32]),
                    asset_id=asset_id,
                    ord=ordinal,
                    text=body,
                    section=section,
                    tokens=estimate_tokens(body),
                )
            )
            ordinal += 1
            buf, size = [], 0

        for para in expanded:
            pt = estimate_tokens(para)
            if buf and size + pt > cfg.target_tokens:
                tail = _tail_overlap(buf, overlap)
                flush()
                if tail:
                    buf.append(tail)
                    size = estimate_tokens(tail)
            buf.append(para)
            size += pt
        flush()

    return chunks
