from __future__ import annotations

from fde_kit.compile.chunk import ChunkingConfig, chunks_from_text, split_sections


def test_split_sections_tracks_headings():
    text = "# 标题\n\n正文一\n\n## 小节\n\n正文二"
    sections = split_sections(text)
    assert [s[0] for s in sections] == ["标题", "小节"]
    assert sections[0][1] == ["正文一"]


def test_chunk_ids_and_sections_are_stable():
    text = "# 政策\n\n" + "条款内容。" * 200
    first = chunks_from_text("as_test", text, title="政策", config=ChunkingConfig(target_tokens=120))
    second = chunks_from_text("as_test", text, title="政策", config=ChunkingConfig(target_tokens=120))
    assert [c.id for c in first] == [c.id for c in second]
    assert all(c.section == "政策" for c in first)


def test_tabular_chunks_do_not_overlap_records():
    rows = "\n\n".join(f"第{i}行：序列号=SN-000{i}；型号=A100；客户=客户{i}" for i in range(1, 30))
    chunks = chunks_from_text("as_table", rows, tabular=True, config=ChunkingConfig(target_tokens=80))
    assert len(chunks) > 1
    for chunk in chunks:
        # 每一块都必须从一整行开始，不能出现“客户3第4行”这类拼接
        assert chunk.text.startswith("第")
        assert "第1行：序列号=SN-0001" in chunk.text or "第1行" not in chunk.text


def test_long_paragraph_is_split_without_losing_text():
    text = "很长的句子。" * 400
    chunks = chunks_from_text("as_long", text, config=ChunkingConfig(target_tokens=100, max_tokens=140))
    joined = "".join(c.text for c in chunks)
    assert joined.count("很长的句子。") >= 380  # 允许重叠或合并，但不允许大面积丢失
