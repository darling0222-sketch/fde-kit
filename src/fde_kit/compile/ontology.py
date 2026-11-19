"""本体（TBox）：先定义“这个现场有什么东西”，再谈检索。

这是 fde-kit 与普通 RAG demo 的第一个分水岭：
普通 RAG 上来就 embedding，fde-kit 要求先用一份 YAML 把领域里的实体类型写清楚——
它是编译期的约束（抽取器）、检索期的加权（命中型号/故障码就加大权重）、
也是交付期的验收口径（“这 6 类实体是否都覆盖到了”）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..text_utils import normalize_value


@dataclass
class EntityType:
    name: str
    patterns: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    weight: float = 0.12
    description: str = ""
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def compile(self) -> None:
        self._compiled = []
        for pattern in self.patterns:
            try:
                self._compiled.append(re.compile(pattern))
            except re.error:
                continue

    def find(self, text: str) -> list[tuple[str, int, int]]:
        hits: list[tuple[str, int, int]] = []
        for pattern in self._compiled:
            for m in pattern.finditer(text):
                value = (m.group(1) if m.groups() else m.group(0)).strip()
                if value:
                    hits.append((value, m.start(), m.end()))
        return hits


@dataclass
class Ontology:
    name: str = "default"
    version: str = "0"
    domain: str = ""
    description: str = ""
    entity_types: dict[str, EntityType] = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    source_path: str = ""

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, path: str | Path) -> Ontology:
        p = Path(path)
        if not p.is_file():
            return cls.empty()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, source_path=str(p))

    @classmethod
    def from_dict(cls, raw: dict, source_path: str = "") -> Ontology:
        types: dict[str, EntityType] = {}
        for name, spec in (raw.get("entity_types") or {}).items():
            spec = spec or {}
            et = EntityType(
                name=name,
                patterns=list(spec.get("patterns") or []),
                aliases=list(spec.get("aliases") or []),
                weight=float(spec.get("weight", 0.12)),
                description=str(spec.get("description", "")),
            )
            et.compile()
            types[name] = et
        return cls(
            name=str(raw.get("name", "default")),
            version=str(raw.get("version", "0")),
            domain=str(raw.get("domain", "")),
            description=str(raw.get("description", "")),
            entity_types=types,
            relations=list(raw.get("relations") or []),
            retrieval=dict(raw.get("retrieval") or {}),
            source_path=source_path,
        )

    @classmethod
    def empty(cls) -> Ontology:
        return cls(name="empty", entity_types={})

    # ------------------------------------------------------------------ 抽取
    def extract(self, text: str, per_type: int = 8) -> list[tuple[str, str, int, int]]:
        """返回 [(type, value, start, end)]，同类上限 per_type 个，按位置排序。"""
        out: list[tuple[str, str, int, int]] = []
        seen: set[tuple[str, str]] = set()
        for et in self.entity_types.values():
            count = 0
            for value, start, end in sorted(et.find(text), key=lambda x: x[1]):
                key = (et.name, normalize_value(value))
                if key in seen:
                    continue
                seen.add(key)
                out.append((et.name, value, start, end))
                count += 1
                if count >= per_type:
                    break
        return sorted(out, key=lambda x: x[2])

    def match_query(self, query: str) -> list[tuple[str, str, float]]:
        """查询侧命中：返回 [(type, value, weight)]，用于给候选块加权。"""
        hits: list[tuple[str, str, float]] = []
        norm_q = normalize_value(query)
        for et in self.entity_types.values():
            for value, _, _ in et.find(query):
                hits.append((et.name, value, et.weight))
            for alias in et.aliases:
                if alias and normalize_value(alias) and normalize_value(alias) in norm_q:
                    hits.append((et.name, alias, et.weight * 0.6))
        # 去重保序
        seen: set[tuple[str, str]] = set()
        uniq: list[tuple[str, str, float]] = []
        for item in hits:
            key = (item[0], normalize_value(item[1]))
            if key not in seen:
                seen.add(key)
                uniq.append(item)
        return uniq

    @property
    def boost(self) -> float:
        return float(self.retrieval.get("boost", 0.15))

    # ------------------------------------------------------------------ 导出
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "description": self.description,
            "source_path": self.source_path,
            "boost": self.boost,
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "weight": et.weight,
                    "patterns": et.patterns,
                    "aliases": et.aliases,
                }
                for et in self.entity_types.values()
            ],
            "relations": self.relations,
        }

    def coverage(self, entity_stats: dict[str, int]) -> list[dict]:
        """交付报告用：本体声明了哪些实体，实际抽到了多少。"""
        rows = []
        for et in self.entity_types.values():
            rows.append(
                {
                    "type": et.name,
                    "description": et.description,
                    "hits": int(entity_stats.get(et.name, 0)),
                    "covered": int(entity_stats.get(et.name, 0)) > 0,
                }
            )
        return rows
