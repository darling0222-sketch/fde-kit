"""FastAPI 应用：工作台后端。

接口刻意保持“少而稳”：编译、检索、问答、评测、报告。
前端只有一页，不做多页路由——现场演示时最怕点三层才看到答案。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .. import __version__
from ..config import Settings, get_settings
from ..eval.runner import load_cases, run_eval
from ..report import write_report
from ..runtime import Runtime

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    engine: str = "serial"


class CompileRequest(BaseModel):
    reset: bool = False
    corpus: str | None = None


class EvalRequest(BaseModel):
    golden: str | None = None
    k: int = 5


class Job:
    def __init__(self, kind: str) -> None:
        self.id = f"job_{uuid.uuid4().hex[:12]}"
        self.kind = kind
        self.status = "running"
        self.stage = "queued"
        self.message = ""
        self.done = 0
        self.total = 0
        self.started = time.time()
        self.finished: float | None = None
        self.result: dict | None = None
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "done": self.done,
            "total": self.total,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "result": self.result,
            "error": self.error,
        }


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runtime: Runtime | None = None
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def get_runtime(self) -> Runtime:
        with self.lock:
            if self.runtime is None:
                self.runtime = Runtime.create(self.settings)
        return self.runtime

    def add_job(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 40:
                oldest = sorted(self.jobs.values(), key=lambda j: j.started)[:10]
                for j in oldest:
                    self.jobs.pop(j.id, None)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = AppState(settings)

    app = FastAPI(title="fde-kit", version=__version__, docs_url="/api/docs")
    app.state.app_state = state

    # ------------------------------------------------------------------ 页面
    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>fde-kit</h1><p>前端资源缺失</p>", status_code=500)
        return FileResponse(page)

    @app.get("/api/config")
    def config() -> dict:
        rt = state.get_runtime()
        rt.agent.retriever.refresh()
        return {
            "version": __version__,
            "settings": settings.as_public_dict(),
            "index": rt.agent.retriever.index_info,
        }

    @app.get("/api/stats")
    def stats() -> dict:
        rt = state.get_runtime()
        return rt.store.stats()

    @app.get("/api/ontology")
    def ontology() -> dict:
        rt = state.get_runtime()
        data = rt.ontology.to_dict()
        data["coverage"] = rt.ontology.coverage(rt.store.entity_stats())
        return data

    @app.get("/api/assets")
    def assets() -> list[dict]:
        rt = state.get_runtime()
        out = []
        for asset in rt.store.list_assets():
            chunks = rt.store.chunks_for_asset(asset.id)
            out.append(
                {
                    "id": asset.id,
                    "title": asset.title,
                    "uri": asset.uri,
                    "media_type": asset.media_type,
                    "size": asset.size,
                    "chunks": len(chunks),
                    "tokens": sum(c.tokens for c in chunks),
                }
            )
        return out

    # ------------------------------------------------------------------ 编译
    @app.post("/api/compile")
    def compile_corpus(req: CompileRequest) -> dict:
        rt = state.get_runtime()
        job = Job("compile")
        state.add_job(job)

        def run() -> None:
            def progress(stage: str, done: int, total: int, message: str) -> None:
                job.stage = stage
                job.done = done
                job.total = total
                job.message = message

            try:
                result = rt.compile(corpus=req.corpus, reset=req.reset, progress=progress)
                job.result = result.to_dict()
                job.status = "done"
                job.stage = "done"
                job.message = "编译完成"
            except Exception as exc:  # 让前端看到原因，而不是转圈
                job.status = "error"
                job.error = str(exc)
                job.message = "编译失败"
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return job.to_dict()

    # ------------------------------------------------------------------ 问答
    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        rt = state.get_runtime()
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        answer = rt.agent.ask(req.question, session_id=req.session_id, engine=req.engine)
        return answer.to_dict()

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), k: int = 6) -> dict:
        rt = state.get_runtime()
        hits = rt.agent.retriever.search(q, k=k)
        out = []
        for hit in hits:
            asset = rt.store.get_asset(hit.chunk.asset_id)
            out.append({**hit.to_dict(), "title": asset.title if asset else "", "uri": asset.uri if asset else ""})
        return {"query": q, "hits": out, "index": rt.agent.retriever.index_info}

    @app.get("/api/ask/stream")
    def ask_stream(q: str, session_id: str = "default") -> StreamingResponse:
        rt = state.get_runtime()

        def event_gen():
            try:
                for event in rt.agent.ask_stream(q, session_id=session_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                payload = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/memory")
    def memory(session_id: str = "default") -> dict:
        rt = state.get_runtime()
        return {"session_id": session_id, "items": rt.agent.memory.history(session_id)}

    @app.delete("/api/memory")
    def clear_memory(session_id: str = "default") -> dict:
        rt = state.get_runtime()
        rt.agent.memory.clear(session_id)
        return {"ok": True}

    # ------------------------------------------------------------------ 评测
    @app.post("/api/eval")
    def start_eval(req: EvalRequest) -> dict:
        rt = state.get_runtime()
        golden = req.golden or "evals/golden.yaml"
        if not Path(golden).is_file():
            raise HTTPException(status_code=400, detail=f"评测集不存在：{golden}")
        name, k, cases = load_cases(golden)
        job = Job("eval")
        state.add_job(job)
        job.total = len(cases)
        job.message = f"评测集《{name}》"

        def run() -> None:
            def progress(done: int, total: int, case_id: str) -> None:
                job.done = done
                job.total = total
                job.message = f"跑用例 {case_id}"

            try:
                outcome = run_eval(rt.agent, cases, k=k, store=rt.store, progress=progress)
                job.result = outcome.to_dict()
                job.status = "done"
                job.stage = "done"
                job.message = "评测完成"
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job.id, "cases": len(cases), "k": k}

    # ------------------------------------------------------------------ 报告
    @app.post("/api/report")
    def make_report() -> dict:
        rt = state.get_runtime()
        out = write_report(rt.store, rt.ontology, settings)
        return {"markdown": out["content"], "path": out["markdown"], "html": out["html"]}

    @app.get("/api/report/view", response_class=HTMLResponse)
    def view_report() -> Any:
        path = settings.reports_dir / "latest.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="尚未生成报告，先在左侧点击“生成交付报告”")
        return FileResponse(path)

    @app.get("/api/report/latest")
    def latest_report() -> JSONResponse:
        path = settings.reports_dir / "latest.md"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="尚未生成报告")
        return JSONResponse({"markdown": path.read_text(encoding="utf-8")})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    return app


app = create_app()
