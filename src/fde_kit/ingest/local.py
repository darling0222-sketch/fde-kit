"""本地目录 connector。客户现场最常见的就是“一个装满资料的共享盘目录”。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .base import RawDoc

MEDIA_TYPES = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".log": "txt",
    ".csv": "csv",
    ".tsv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "data", ".idea"}
MAX_BYTES = 32 * 1024 * 1024


class LocalConnector:
    name = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def iter_docs(self) -> Iterator[RawDoc]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*")):
            if path.is_dir():
                continue
            if any(part in SKIP_DIRS or part.startswith(".") for part in path.parts):
                continue
            media = MEDIA_TYPES.get(path.suffix.lower())
            if not media:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > MAX_BYTES:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            yield RawDoc(
                source=self.name,
                uri=str(path.as_posix()),
                media_type=media,
                title=path.stem,
                data=data,
                meta={"rel_path": path.relative_to(self.root).as_posix()},
            )
