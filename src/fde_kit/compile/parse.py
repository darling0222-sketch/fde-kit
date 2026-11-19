"""多格式解析。

原则：能解析就解析，不能解析就明确记 warning（交付报告里要列出“哪些资料没进来”），
绝不用异常把整批资料打断——现场一次接 3000 份文件，坏一份不能全废。
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field

from ..ingest.base import RawDoc

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\u00a0]+")


@dataclass
class ParsedDoc:
    text: str
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def decode_bytes(data: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8/replace"


def parse_doc(doc: RawDoc) -> ParsedDoc:
    parser = {
        "md": _parse_markdown,
        "txt": _parse_plain,
        "csv": _parse_csv,
        "jsonl": _parse_jsonl,
        "json": _parse_json,
        "html": _parse_html,
        "pdf": _parse_pdf,
        "docx": _parse_docx,
    }.get(doc.media_type, _parse_plain)
    return parser(doc)


def _parse_plain(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    return ParsedDoc(text=_clean(text), meta={"encoding": enc})


def _parse_markdown(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    meta: dict = {"encoding": enc}
    m = _FRONTMATTER.match(text)
    if m:
        block = m.group(0)
        text = text[m.end():]
        meta["frontmatter"] = block.strip().strip("-").strip()
    headings = [h.strip("# ").strip() for h in re.findall(r"^#{1,6}\s+.+$", text, re.M)]
    meta["headings"] = headings[:20]
    return ParsedDoc(text=_clean(text), meta=meta)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_csv(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    delimiter = "\t" if doc.media_type == "csv" and doc.uri.lower().endswith(".tsv") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return ParsedDoc(text="", meta={"encoding": enc}, warnings=["CSV 为空"])
    header = [h.strip() or f"列{i + 1}" for i, h in enumerate(rows[0])]
    lines = ["表头：" + " | ".join(header)]
    for idx, row in enumerate(rows[1:], start=1):
        pairs = []
        for i, cell in enumerate(row):
            key = header[i] if i < len(header) else f"列{i + 1}"
            value = cell.strip()
            if value:
                pairs.append(f"{key}={value}")
        if pairs:
            lines.append(f"第{idx}行：" + "；".join(pairs))
    # 行间留空行：让每一行成为独立段落，切块时按行装箱，引用才能精确到行
    return ParsedDoc(
        text="\n\n".join(lines),
        meta={"encoding": enc, "rows": len(rows) - 1, "columns": header, "kind": "table"},
    )


def _render_record(record: dict, prefix: str = "") -> str:
    parts = []
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={value}")
    return f"{prefix}" + "；".join(parts)


def _parse_jsonl(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    lines: list[str] = []
    bad = 0
    for i, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            lines.append(_render_record(obj, prefix=f"第{i}条："))
        else:
            lines.append(f"第{i}条：{obj}")
    warnings = [f"{bad} 行 JSON 解析失败已跳过"] if bad else []
    return ParsedDoc(text="\n\n".join(lines), meta={"encoding": enc, "records": len(lines)}, warnings=warnings)


def _parse_json(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedDoc(text=_clean(text), meta={"encoding": enc}, warnings=[f"JSON 解析失败：{exc}"])

    def walk(node, path: str, out: list[str], depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key, out, depth + 1)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", out, depth + 1)
        else:
            out.append(f"{path}={node}")

    lines: list[str] = []
    walk(obj, "", lines)
    return ParsedDoc(text="\n".join(lines), meta={"encoding": enc, "fields": len(lines)})


def _parse_html(doc: RawDoc) -> ParsedDoc:
    text, enc = decode_bytes(doc.data)
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    if m:
        title = m.group(1).strip()
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        body = soup.get_text("\n")
    except Exception:
        body = _TAG.sub("\n", text)
    return ParsedDoc(
        text=_clean(body),
        meta={"encoding": enc, "html_title": title},
        warnings=[] if _has_bs4() else ["未安装 beautifulsoup4，已用正则降级提取正文"],
    )


def _has_bs4() -> bool:
    try:
        import bs4  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_pdf(doc: RawDoc) -> ParsedDoc:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return ParsedDoc(text="", meta={}, warnings=["未安装 pypdf，PDF 已跳过（pip install 'fde-kit[docs]'）"])
    try:
        reader = PdfReader(io.BytesIO(doc.data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(p for p in pages if p.strip())
        warn = [] if text.strip() else ["PDF 未提取到文本层，可能是扫描件（需 OCR/VLM，当前版本不支持）"]
        return ParsedDoc(text=_clean(text), meta={"pages": len(reader.pages)}, warnings=warn)
    except Exception as exc:  # 损坏的 PDF 不应该炸掉整批
        return ParsedDoc(text="", meta={}, warnings=[f"PDF 解析失败：{exc}"])


def _parse_docx(doc: RawDoc) -> ParsedDoc:
    try:
        import docx  # type: ignore
    except ImportError:
        return ParsedDoc(text="", meta={}, warnings=["未安装 python-docx，DOCX 已跳过（pip install 'fde-kit[docs]'）"])
    try:
        d = docx.Document(io.BytesIO(doc.data))
        blocks = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        return ParsedDoc(text=_clean("\n".join(blocks)), meta={"paragraphs": len(blocks)})
    except Exception as exc:
        return ParsedDoc(text="", meta={}, warnings=[f"DOCX 解析失败：{exc}"])
