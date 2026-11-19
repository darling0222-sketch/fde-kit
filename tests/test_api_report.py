from __future__ import annotations

from fastapi.testclient import TestClient

from fde_kit.api.app import create_app
from fde_kit.report import render_html, render_markdown, write_report


def _client(runtime) -> TestClient:
    app = create_app(runtime.settings)
    return TestClient(app)


def test_health_and_config(runtime):
    with _client(runtime) as client:
        assert client.get("/api/health").json()["ok"] is True
        cfg = client.get("/api/config").json()
        assert cfg["settings"]["mode"] in {"offline", "online"}
        assert cfg["index"]["chunks"] > 0


def test_index_page_served(runtime):
    with _client(runtime) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "fde-kit" in resp.text


def test_stats_and_assets(runtime):
    with _client(runtime) as client:
        stats = client.get("/api/stats").json()
        assert stats["assets"] >= 5
        assert stats["chunks"] == stats["vectors"]
        assets = client.get("/api/assets").json()
        assert any("政策" in a["title"] for a in assets)


def test_ask_endpoint_returns_citations(runtime):
    with _client(runtime) as client:
        resp = client.post("/api/ask", json={"question": "A100 的整机保修期是多久？", "session_id": "api-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["citations"]
        assert data["bundle"]["token_budget"] > 0


def test_search_endpoint(runtime):
    with _client(runtime) as client:
        data = client.get("/api/search", params={"q": "E021", "k": 3}).json()
        assert data["hits"]
        assert data["hits"][0]["title"]


def test_stream_endpoint_emits_sse(runtime):
    with _client(runtime) as client:
        with client.stream("GET", "/api/ask/stream", params={"q": "退货运费由谁承担？"}) as resp:
            body = "".join(chunk for chunk in resp.iter_text())
        assert "data:" in body
        assert '"type": "done"' in body


def test_compile_job_flow(runtime):
    with _client(runtime) as client:
        started = client.post("/api/compile", json={"reset": False}).json()
        job_id = started["job_id"]
        for _ in range(60):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"done", "error"}:
                break
            import time

            time.sleep(0.1)
        assert job["status"] == "done"
        assert job["result"]["assets_skipped"] >= 5


def test_report_generation(runtime):
    out = write_report(runtime.store, runtime.ontology, runtime.settings)
    assert out["markdown"].endswith(".md")
    assert "交付验收报告" in out["content"]
    assert "本体覆盖" in out["content"]
    html = render_html([("h1", "标题"), ("table", (["a"], [["1"]]))], "t")
    assert "<table>" in html and "<h1>" in html
    assert render_markdown([("h2", "小节")]).strip() == "## 小节"
