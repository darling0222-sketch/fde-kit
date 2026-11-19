"""接入层契约。

一个 connector 只负责“把东西拿进来”，不做解析、不做切块——
这是 jonex 那类平台里很容易被糊在一起的地方，糊在一起后就没法单独换数据源。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RawDoc:
    source: str            # local / http / inline
    uri: str               # 稳定定位符，报告里要给人看
    media_type: str        # md / txt / csv / jsonl / html / pdf / docx
    title: str
    data: bytes
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def filename(self) -> str:
        return Path(self.uri).name


class Connector(Protocol):
    name: str

    def iter_docs(self) -> Iterator[RawDoc]:  # pragma: no cover - 协议
        ...
