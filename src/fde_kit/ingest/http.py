"""HTTP connector：拉一个 URL 当资料。用于客户的内网 wiki / 在线帮助中心。"""

from __future__ import annotations

from collections.abc import Iterator

import httpx

from .base import RawDoc

SUFFIX_MEDIA = {
    ".md": "md",
    ".txt": "txt",
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


class HttpConnector:
    name = "http"

    def __init__(self, urls: list[str], timeout: float = 20.0) -> None:
        self.urls = urls
        self.timeout = timeout

    def iter_docs(self) -> Iterator[RawDoc]:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for url in self.urls:
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue
                ctype = resp.headers.get("content-type", "")
                media = "html"
                for suffix, m in SUFFIX_MEDIA.items():
                    if url.lower().split("?")[0].endswith(suffix):
                        media = m
                        break
                else:
                    if "json" in ctype:
                        media = "json"
                    elif "markdown" in ctype:
                        media = "md"
                    elif "pdf" in ctype:
                        media = "pdf"
                title = url.rstrip("/").rsplit("/", 1)[-1] or url
                yield RawDoc(
                    source=self.name,
                    uri=url,
                    media_type=media,
                    title=title,
                    data=resp.content,
                    meta={"status": resp.status_code},
                )
