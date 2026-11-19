"""MCP 服务（stdio）：把知识库暴露给任意支持 MCP 的 agent。

这是"交付一次、复用到下一个客户"的关键接口：
Cursor / Claude Code / 客户自己的 agent 接上之后，不需要改它们的代码，
就能用同一个知识库检索与问答。
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .runtime import Runtime

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "kb_search",
        "description": "在客户知识库里做混合检索（BM25 + 向量 + 本体加权），返回可引用的证据块。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索式，尽量包含型号、故障码等专有名词"},
                "k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kb_ask",
        "description": "基于知识库回答问题，回答中的 [S1] 形式编号对应返回的 citations。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "session_id": {"type": "string", "default": "mcp"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "kb_stats",
        "description": "查看知识库规模、本体覆盖情况与运行模式。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpServer:
    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime

    # ------------------------------------------------------------------ 协议
    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fde-kit", "version": __version__},
                },
            )
        if method in {"notifications/initialized", "initialized"}:
            return None
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            return self._ok(req_id, {"tools": TOOLS})
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                text = self._call_tool(name, args)
            except Exception as exc:
                return self._err(req_id, -32000, f"工具执行失败：{exc}")
            return self._ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        if method == "resources/list":
            return self._ok(req_id, {"resources": []})
        if method == "prompts/list":
            return self._ok(req_id, {"prompts": []})
        return self._err(req_id, -32601, f"未实现的方法：{method}")

    # ------------------------------------------------------------------ 工具
    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "kb_search":
            query = str(args.get("query", "")).strip()
            k = int(args.get("k", 5))
            hits = self.rt.agent.retriever.search(query, k=k)
            if not hits:
                return json.dumps({"query": query, "hits": []}, ensure_ascii=False)
            payload = []
            for i, hit in enumerate(hits, start=1):
                asset = self.rt.store.get_asset(hit.chunk.asset_id)
                payload.append(
                    {
                        "rank": i,
                        "score": round(hit.score, 4),
                        "title": asset.title if asset else "",
                        "uri": asset.uri if asset else "",
                        "section": hit.chunk.section,
                        "text": hit.chunk.text[:800],
                        "entities": hit.entities,
                    }
                )
            return json.dumps({"query": query, "hits": payload}, ensure_ascii=False, indent=2)

        if name == "kb_ask":
            answer = self.rt.agent.ask(str(args.get("question", "")), session_id=str(args.get("session_id", "mcp")))
            return json.dumps(answer.to_dict(with_bundle=False), ensure_ascii=False, indent=2)

        if name == "kb_stats":
            stats = self.rt.store.stats()
            stats["mode"] = "online" if self.rt.settings.has_llm else "offline"
            stats["ontology"] = self.rt.ontology.name
            stats["entity_coverage"] = self.rt.ontology.coverage(stats["entity_types"])
            return json.dumps(stats, ensure_ascii=False, indent=2)

        raise ValueError(f"未知工具：{name}")

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _ok(req_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def serve_stdio() -> None:
    runtime = Runtime.create()
    server = McpServer(runtime)
    print(f"fde-kit MCP server 已启动（{len(runtime.store.all_chunks())} 块证据）", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "解析失败"}}))
            sys.stdout.flush()
            continue
        response = server.handle(request)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
