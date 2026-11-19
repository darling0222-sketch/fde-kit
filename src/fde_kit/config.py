"""配置：只用标准库解析 .env 与环境变量，避免引入额外依赖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载：KEY=VALUE，# 开头为注释，已存在的环境变量优先。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path("./data"))
    corpus: Path = field(default_factory=lambda: Path("./examples/corpus"))
    ontology: Path = field(default_factory=lambda: Path("./ontology/after_sales.yaml"))

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    embed_dim: int = 1024

    top_k: int = 8
    token_budget: int = 2400
    tenant: str = "demo"
    min_coverage: float = 0.35
    min_vector_score: float = 0.32

    # 静态服务目录，供 API 使用
    project_root: Path = field(default_factory=lambda: Path("."))

    @property
    def db_path(self) -> Path:
        return self.home / "fde.db"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def embed_ready(self) -> bool:
        base = self.embed_base_url or self.llm_base_url
        key = self.embed_api_key or self.llm_api_key
        model = self.embed_model or "text-embedding-3-small"
        return bool(base and key and model)

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def as_public_dict(self) -> dict:
        """给前端用的脱敏配置。"""
        return {
            "tenant": self.tenant,
            "mode": "online" if self.has_llm else "offline",
            "llm_model": self.llm_model or "",
            "embed_model": (self.embed_model or "hash-ngram") if self.embed_ready else "hash-ngram",
            "top_k": self.top_k,
            "token_budget": self.token_budget,
            "min_coverage": self.min_coverage,
            "min_vector_score": self.min_vector_score,
            "corpus": str(self.corpus),
            "ontology": str(self.ontology),
        }


def load_settings(**overrides) -> Settings:
    _load_dotenv(Path(".env"))
    s = Settings(
        home=Path(_env("FDE_HOME", "./data")),
        corpus=Path(_env("FDE_CORPUS", "./examples/corpus")),
        ontology=Path(_env("FDE_ONTOLOGY", "./ontology/after_sales.yaml")),
        llm_base_url=_env("FDE_LLM_BASE_URL"),
        llm_api_key=_env("FDE_LLM_API_KEY"),
        llm_model=_env("FDE_LLM_MODEL"),
        embed_base_url=_env("FDE_EMBED_BASE_URL"),
        embed_api_key=_env("FDE_EMBED_API_KEY"),
        embed_model=_env("FDE_EMBED_MODEL"),
        embed_dim=_env_int("FDE_EMBED_DIM", 1024),
        top_k=_env_int("FDE_TOP_K", 8),
        token_budget=_env_int("FDE_TOKEN_BUDGET", 2400),
        tenant=_env("FDE_TENANT", "demo"),
        min_coverage=float(_env("FDE_MIN_COVERAGE") or 0.35),
        min_vector_score=float(_env("FDE_MIN_VECTOR") or 0.32),
    )
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
