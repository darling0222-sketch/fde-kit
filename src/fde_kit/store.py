"""存储层：单文件 SQLite。资产、分块、本体实体、向量、运行记录、会话记忆。

设计约束：
- 不依赖外部数据库，clone 下来就能跑；
- 向量以 float32 BLOB 存库，检索时整体载入内存做精确余弦（万级 chunk 以内够用）；
- revision 单调递增，检索侧据此判断是否需要重建索引缓存。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .models import Asset, Chunk, Entity
from .text_utils import estimate_tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  uri TEXT NOT NULL,
  media_type TEXT NOT NULL,
  title TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  text TEXT,
  meta TEXT,
  created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_sha ON assets(sha256);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  asset_id TEXT NOT NULL,
  ord INTEGER NOT NULL,
  text TEXT NOT NULL,
  section TEXT,
  tokens INTEGER NOT NULL DEFAULT 0,
  meta TEXT,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_asset ON chunks(asset_id);

CREATE TABLE IF NOT EXISTS entities (
  chunk_id TEXT NOT NULL,
  type TEXT NOT NULL,
  value TEXT NOT NULL,
  normalized TEXT NOT NULL,
  span_start INTEGER DEFAULT -1,
  span_end INTEGER DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(normalized);

CREATE TABLE IF NOT EXISTS vectors (
  chunk_id TEXT PRIMARY KEY,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  finished_at REAL,
  meta TEXT
);

CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evals_run ON evals(run_id);

CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- 基础
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # ---------------------------------------------------------------- 资产
    def upsert_asset(self, asset: Asset) -> None:
        row = asset.to_row()
        row["created_at"] = time.time()
        self._exec(
            """INSERT INTO assets (id, source, uri, media_type, title, sha256, size, text, meta, created_at)
               VALUES (:id, :source, :uri, :media_type, :title, :sha256, :size, :text, :meta, :created_at)
               ON CONFLICT(id) DO UPDATE SET
                 source=excluded.source, uri=excluded.uri, media_type=excluded.media_type,
                 title=excluded.title, sha256=excluded.sha256, size=excluded.size,
                 text=excluded.text, meta=excluded.meta""",
            row,
        )

    def find_asset_by_sha(self, sha256: str) -> Asset | None:
        rows = self._query("SELECT * FROM assets WHERE sha256 = ?", (sha256,))
        return Asset.from_row(rows[0]) if rows else None

    def get_asset(self, asset_id: str) -> Asset | None:
        rows = self._query("SELECT * FROM assets WHERE id = ?", (asset_id,))
        return Asset.from_row(rows[0]) if rows else None

    def list_assets(self) -> list[Asset]:
        return [Asset.from_row(r) for r in self._query("SELECT * FROM assets ORDER BY created_at DESC")]

    def delete_asset(self, asset_id: str) -> None:
        self._exec("DELETE FROM entities WHERE chunk_id IN (SELECT id FROM chunks WHERE asset_id = ?)", (asset_id,))
        self._exec("DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE asset_id = ?)", (asset_id,))
        self._exec("DELETE FROM chunks WHERE asset_id = ?", (asset_id,))
        self._exec("DELETE FROM assets WHERE id = ?", (asset_id,))

    def count_assets(self) -> int:
        return int(self._query("SELECT COUNT(*) AS c FROM assets")[0]["c"])

    # ---------------------------------------------------------------- 分块
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks (id, asset_id, ord, text, section, tokens, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.id, c.asset_id, c.ord, c.text, c.section, c.tokens,
                     json.dumps(c.meta, ensure_ascii=False))
                    for c in chunks
                ],
            )
            self._conn.commit()

    def replace_asset_chunks(self, asset_id: str, chunks: list[Chunk]) -> None:
        self.delete_asset_chunks(asset_id)
        self.add_chunks(chunks)

    def delete_asset_chunks(self, asset_id: str) -> None:
        self._exec("DELETE FROM entities WHERE chunk_id IN (SELECT id FROM chunks WHERE asset_id = ?)", (asset_id,))
        self._exec("DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE asset_id = ?)", (asset_id,))
        self._exec("DELETE FROM chunks WHERE asset_id = ?", (asset_id,))

    def all_chunks(self) -> list[Chunk]:
        return [Chunk.from_row(r) for r in self._query("SELECT * FROM chunks ORDER BY asset_id, ord")]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        rows = self._query("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        return Chunk.from_row(rows[0]) if rows else None

    def chunks_for_asset(self, asset_id: str) -> list[Chunk]:
        rows = self._query("SELECT * FROM chunks WHERE asset_id = ? ORDER BY ord", (asset_id,))
        return [Chunk.from_row(r) for r in rows]

    def neighbour_chunks(self, chunk: Chunk, span: int = 1) -> list[Chunk]:
        rows = self._query(
            "SELECT * FROM chunks WHERE asset_id = ? AND ord BETWEEN ? AND ? ORDER BY ord",
            (chunk.asset_id, chunk.ord - span, chunk.ord + span),
        )
        return [Chunk.from_row(r) for r in rows]

    def count_chunks(self) -> int:
        return int(self._query("SELECT COUNT(*) AS c FROM chunks")[0]["c"])

    # ---------------------------------------------------------------- 实体
    def add_entities(self, entities: Iterable[Entity]) -> None:
        with self._lock:
            self._conn.executemany(
                """INSERT INTO entities (chunk_id, type, value, normalized, span_start, span_end)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(e.chunk_id, e.type, e.value, e.normalized, e.span_start, e.span_end) for e in entities],
            )
            self._conn.commit()

    def entities_for_chunk(self, chunk_id: str) -> list[Entity]:
        rows = self._query("SELECT * FROM entities WHERE chunk_id = ?", (chunk_id,))
        return [
            Entity(
                chunk_id=r["chunk_id"],
                type=r["type"],
                value=r["value"],
                normalized=r["normalized"],
                span_start=r["span_start"],
                span_end=r["span_end"],
            )
            for r in rows
        ]

    def entity_index(self) -> dict[str, list[str]]:
        """chunk_id -> ["Type:value", ...]，检索与上下文面板共用。"""
        out: dict[str, list[str]] = {}
        for r in self._query("SELECT chunk_id, type, value FROM entities"):
            out.setdefault(r["chunk_id"], []).append(f"{r['type']}:{r['value']}")
        return out

    def chunks_by_entity(self, normalized: str, limit: int = 20) -> list[str]:
        rows = self._query(
            "SELECT DISTINCT chunk_id FROM entities WHERE normalized = ? LIMIT ?", (normalized, limit)
        )
        return [r["chunk_id"] for r in rows]

    def entity_stats(self) -> dict[str, int]:
        rows = self._query("SELECT type, COUNT(*) AS c FROM entities GROUP BY type ORDER BY c DESC")
        return {r["type"]: int(r["c"]) for r in rows}

    # ---------------------------------------------------------------- 向量
    def set_vectors(self, vectors: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, dim, vec) VALUES (?, ?, ?)",
                [(cid, int(v.shape[0]), v.astype(np.float32).tobytes()) for cid, v in vectors.items()],
            )
            self._conn.commit()

    def load_vectors(self) -> tuple[list[str], np.ndarray]:
        rows = self._query("SELECT chunk_id, dim, vec FROM vectors")
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["chunk_id"] for r in rows]
        dim = int(rows[0]["dim"])
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows]).astype(np.float32)
        if mat.shape[0] != len(ids) or mat.shape[1] != dim:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, mat

    def count_vectors(self) -> int:
        return int(self._query("SELECT COUNT(*) AS c FROM vectors")[0]["c"])

    # ---------------------------------------------------------------- 版本
    def bump_revision(self) -> int:
        rev = self.revision() + 1
        self.kv_set("revision", str(rev))
        return rev

    def revision(self) -> int:
        return int(self.kv_get("revision", "0") or 0)

    # ---------------------------------------------------------------- KV
    def kv_get(self, key: str, default: Any = None) -> Any:
        rows = self._query("SELECT value FROM kv WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def kv_set(self, key: str, value: str) -> None:
        self._exec("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))

    # ---------------------------------------------------------------- 运行
    def create_run(self, kind: str, meta: dict | None = None) -> str:
        from .models import stable_id

        run_id = stable_id("run", kind, str(time.time()))
        self._exec(
            "INSERT INTO runs (id, kind, status, created_at, meta) VALUES (?, ?, ?, ?, ?)",
            (run_id, kind, "running", time.time(), json.dumps(meta or {}, ensure_ascii=False)),
        )
        return run_id

    def finish_run(self, run_id: str, status: str, meta: dict | None = None) -> None:
        current = self.get_run(run_id)
        merged = dict(current.get("meta", {})) if current else {}
        merged.update(meta or {})
        self._exec(
            "UPDATE runs SET status = ?, finished_at = ?, meta = ? WHERE id = ?",
            (status, time.time(), json.dumps(merged, ensure_ascii=False), run_id),
        )

    def get_run(self, run_id: str) -> dict | None:
        rows = self._query("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r["id"],
            "kind": r["kind"],
            "status": r["status"],
            "created_at": r["created_at"],
            "finished_at": r["finished_at"],
            "meta": json.loads(r["meta"] or "{}"),
        }

    def list_runs(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        if kind:
            rows = self._query(
                "SELECT * FROM runs WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self._query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "status": r["status"],
                "created_at": r["created_at"],
                "finished_at": r["finished_at"],
                "meta": json.loads(r["meta"] or "{}"),
            }
            for r in rows
        ]

    # ---------------------------------------------------------------- 评测
    def save_eval(self, run_id: str, case_id: str, payload: dict) -> None:
        self._exec(
            "INSERT INTO evals (run_id, case_id, payload) VALUES (?, ?, ?)",
            (run_id, case_id, json.dumps(payload, ensure_ascii=False)),
        )

    def list_evals(self, run_id: str) -> list[dict]:
        rows = self._query("SELECT case_id, payload FROM evals WHERE run_id = ? ORDER BY id", (run_id,))
        return [{"case_id": r["case_id"], **json.loads(r["payload"])} for r in rows]

    # ---------------------------------------------------------------- 记忆
    def add_memory(self, session_id: str, kind: str, text: str, source: str = "") -> None:
        self._exec(
            "INSERT INTO memory (session_id, kind, text, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, kind, text, source, time.time()),
        )

    def list_memory(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self._query(
            "SELECT kind, text, source, created_at FROM memory WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in rows][::-1]

    # ---------------------------------------------------------------- 概览
    def stats(self) -> dict:
        assets = self._query("SELECT media_type, COUNT(*) AS c, SUM(size) AS s FROM assets GROUP BY media_type")
        total_tokens = int(self._query("SELECT COALESCE(SUM(tokens),0) AS t FROM chunks")[0]["t"])
        return {
            "assets": self.count_assets(),
            "chunks": self.count_chunks(),
            "tokens": total_tokens,
            "vectors": self.count_vectors(),
            "entities": sum(self.entity_stats().values()),
            "entity_types": self.entity_stats(),
            "by_media": {
                r["media_type"]: {"count": int(r["c"]), "bytes": int(r["s"] or 0)} for r in assets
            },
            "revision": self.revision(),
        }


def chunk_tokens(text: str) -> int:
    return estimate_tokens(text)
