from __future__ import annotations

from fde_kit.text_utils import (
    clip,
    content_chars,
    estimate_tokens,
    jaccard,
    split_sentences,
    tokenize,
)


def test_tokenize_mixes_cjk_bigrams_and_latin_words():
    tokens = tokenize("A100 整机保修 24 个月")
    assert "a100" in tokens
    assert "整机" in tokens
    assert "保修" in tokens
    assert "24" in tokens


def test_tokenize_single_cjk_char_kept():
    assert tokenize("好") == ["好"]


def test_estimate_tokens_counts_cjk_heavier_than_latin():
    assert estimate_tokens("中文五个字符") > estimate_tokens("abcde")
    assert estimate_tokens("") == 0


def test_split_sentences_keeps_delimiters():
    parts = split_sentences("第一句。第二句；第三句！")
    assert parts == ["第一句。", "第二句；", "第三句！"]


def test_content_chars_drops_stopwords_keeps_terms():
    chars = content_chars("这个设备的保修期是多久")
    assert "保" in chars and "修" in chars and "设" in chars
    assert "的" not in chars and "是" not in chars


def test_jaccard_and_clip():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard(set(), {"a"}) == 0.0
    assert clip("一二三四五", 3).endswith("…")
