"""文本工具：中英混排 tokenize、token 估算、句子切分。

中文用字 bigram 作为检索单元——比单字精确，比整词切分轻量，无需分词词典。
"""

from __future__ import annotations

import math
import re

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_LATIN_WORD = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-\.]*")
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|\n+")


def tokenize(text: str) -> list[str]:
    """中文 bigram + 英文小写词。单字中文串保留自身，避免长尾查询丢失。"""
    if not text:
        return []
    low = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(low):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    for word in _LATIN_WORD.findall(low):
        word = word.strip(".-")
        if len(word) > 1 or word.isdigit():
            tokens.append(word)
    return tokens


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中日韩 1 字≈1 token，拉丁字符 ≈4 字/token。"""
    if not text:
        return 0
    cjk = len(_CJK_RUN.findall(text))
    cjk_chars = sum(len(m) for m in _CJK_RUN.findall(text))
    latin_chars = max(len(text) - cjk_chars, 0)
    return int(cjk_chars + math.ceil(latin_chars / 4)) or max(cjk, 1)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def normalize_value(value: str) -> str:
    return re.sub(r"[\s\-_/]+", "", value).lower()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# 高频虚词/疑问词：判断“库里到底有没有这东西”时不该被它们带偏
STOP_CHARS = set(
    "的了是在和与及或有为对从到把被就都也还很更最会能可以要将并且而但如果所以因为"
    "吗呢吧啊么什怎多哪谁这那个我你他她它您请帮想要需是否上下中前后里外"
)


def content_chars(text: str) -> list[str]:
    """字级内容单元：中文去虚词后逐字，英文取词。

    用途是“查询覆盖率”这类判别性指标：bigram 会产生大量跨词边界的噪声
    （“批设”“备出”），抓不到真正的专有名词；字级统计更稳。
    """
    low = text.lower()
    out = [ch for ch in low if "\u4e00" <= ch <= "\u9fff" and ch not in STOP_CHARS]
    out.extend(w for w in _LATIN_WORD.findall(low) if len(w) > 1)
    return out


def clip(text: str, limit: int = 480) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.strip().lower()).strip("-")
    return slug[:limit] or "doc"
