"""运行时装配：CLI 与 API 共用同一个装配入口，避免两套初始化逻辑漂移。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent.agent import DeliveryAgent
from .compile.ontology import Ontology
from .compile.pipeline import Compiler, CompileResult
from .config import Settings, load_settings
from .ingest.local import LocalConnector
from .retrieve.embed import Embedder, build_embedder
from .store import Store


@dataclass
class Runtime:
    settings: Settings
    store: Store
    ontology: Ontology
    embedder: Embedder
    agent: DeliveryAgent

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        ontology_path: str | Path | None = None,
    ) -> Runtime:
        settings = settings or load_settings()
        settings.ensure_dirs()
        store = Store(settings.db_path)
        ontology = Ontology.load(ontology_path or settings.ontology)
        embedder = build_embedder(settings)
        agent = DeliveryAgent(store, settings, ontology, embedder=embedder)
        return cls(settings=settings, store=store, ontology=ontology, embedder=embedder, agent=agent)

    # ------------------------------------------------------------------ 编译
    def compile(self, corpus: str | Path | None = None, reset: bool = False, progress=None) -> CompileResult:
        corpus_path = corpus or self.settings.corpus
        docs = list(LocalConnector(corpus_path).iter_docs())
        run_id = self.store.create_run("compile", {"corpus": str(corpus_path), "docs": len(docs)})
        compiler = Compiler(self.store, self.ontology, self.embedder)
        try:
            result = compiler.compile(docs, reset=reset, progress=progress)
        except Exception as exc:
            self.store.finish_run(run_id, "error", {"error": str(exc)})
            raise
        self.store.finish_run(run_id, "done", {"result": result.to_dict()})
        self.agent.retriever.refresh(force=True)
        return result

    def close(self) -> None:
        self.store.close()
