"""命令行入口：fde <command>。

设计原则：没有子命令时打印“下一步该做什么”，而不是一屏 help。
现场工程师第一次拿到仓库，应该 3 行命令看到答案。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_settings
from .runtime import Runtime


def _print_compile(result) -> None:
    data = result.to_dict()
    print("\n编译完成")
    print(f"  资料  {data['assets_new']} 新增 / {data['assets_skipped']} 跳过 / {data['assets_failed']} 失败")
    print(f"  分块  {data['chunks']} 块 · {data['tokens']:,} tokens · 向量 {data['vectors']}")
    print(f"  实体  {data['entities']} 个 · 编译器 {data['embedder']} · 耗时 {data['duration']}s")
    if data["warnings"]:
        print("\n  未纳入的资料：")
        for w in data["warnings"][:10]:
            print(f"    - {w['uri']} ({w['media_type']}): {w['message']}")
        if len(data["warnings"]) > 10:
            print(f"    ... 另有 {len(data['warnings']) - 10} 条，见交付报告")


def cmd_init(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.ensure_dirs()
    ontology = Path(settings.ontology)
    if not ontology.exists():
        ontology.parent.mkdir(parents=True, exist_ok=True)
        ontology.write_text(
            "name: 我的领域本体\nversion: \"0.1.0\"\ndomain: 待填写\n"
            "description: 用 5 分钟把领域里的实体类型列出来，比调 5 天检索参数有用\n"
            "retrieval:\n  boost: 0.15\nentity_types:\n  Product:\n    description: 产品/设备型号\n"
            "    weight: 0.2\n    patterns:\n      - '[A-Z]\\d{3,4}'\n",
            encoding="utf-8",
        )
        print(f"已生成本体骨架：{ontology}")
    print(f"数据目录：{settings.home}")
    print("\n下一步：")
    print("  1. fde compile --reset        # 编译资料")
    print("  2. fde ask \"你们的退货政策是什么？\"")
    print("  3. fde eval                   # 跑评测集")
    print("  4. fde report                 # 出交付报告")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    rt = Runtime.create()
    docs_count = len(list(Path(args.corpus or rt.settings.corpus).rglob("*"))) if args.verbose else 0
    if args.verbose:
        print(f"扫描目录：{args.corpus or rt.settings.corpus}（{docs_count} 个条目）")
    result = rt.compile(corpus=args.corpus, reset=args.reset)
    _print_compile(result)
    rt.close()
    return 0 if result.assets_new or result.assets_skipped else 1


def cmd_search(args: argparse.Namespace) -> int:
    rt = Runtime.create()
    hits = rt.agent.retriever.search(args.query, k=args.k)
    if not hits:
        print("没有命中。先确认已执行 fde compile。")
        return 1
    for i, hit in enumerate(hits, start=1):
        asset = rt.store.get_asset(hit.chunk.asset_id)
        print(f"\n[{i}] {hit.score:.4f}  {asset.title if asset else ''} · {hit.chunk.section}")
        print(f"    bm25={hit.scores.get('bm25', 0):.2f} vec={hit.scores.get('vector', 0):.3f} "
              f"rrf={hit.scores.get('rrf', 0):.4f} 本体={hit.scores.get('entity', 0):.2f}")
        print("    " + hit.chunk.text.replace("\n", " ")[:180])
    rt.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    rt = Runtime.create()
    engine = "langgraph" if args.graph else "serial"
    answer = rt.agent.ask(args.question, session_id=args.session, engine=engine)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(answer.text)
        if answer.citations:
            print("\n引用：")
            for c in answer.citations:
                print(f"  [{c.sid}] {c.title}" + (f" · {c.section}" if c.section else ""))
                print(f"       {c.uri}")
        print(f"\n模式={answer.mode} 有据={answer.grounded} 置信度={answer.confidence:.2f}")
        if args.trace:
            print("\n执行轨迹：")
            for step in answer.trace:
                print(f"  {step['node']:<9} {step['detail']}")
    rt.close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import load_cases, run_eval

    rt = Runtime.create()
    name, k, cases = load_cases(args.golden)
    if not cases:
        print(f"评测集为空：{args.golden}")
        return 1
    print(f"评测集《{name}》 {len(cases)} 条 · k={k} · 引擎 {args.engine}")
    outcome = run_eval(rt.agent, cases, k=k, store=rt.store, engine=args.engine)
    s = outcome.summary
    print("\n结果")
    print(f"  hit@{k}        {s['hit_rate']:.1%}   （纯词法基线 {s['lexical_hit_rate']:.1%}）")
    print(f"  MRR           {s['mrr']:.3f}    （纯词法基线 {s['lexical_mrr']:.3f}）")
    print(f"  关键词覆盖     {s['keyword_coverage']:.1%}")
    print(f"  引用有效率     {s['citation_validity']:.1%}")
    print(f"  有据可依率     {s['grounded_rate']:.1%}")
    print(f"  平均延迟       {s['avg_latency_ms']} ms")
    fails = [
        r
        for r in outcome.results
        if not r.hit or (not r.grounded and not r.expect_no_answer)
    ]
    if fails:
        print(f"\n未通过 {len(fails)} 条：")
        for r in fails:
            print(f"  - {r.id} {r.question[:40]} (hit={r.hit} grounded={r.grounded})")
    refused = [r for r in outcome.results if r.expect_no_answer and r.hit]
    if refused:
        print(f"\n反例通过 {len(refused)} 条：无证据时正确拒答。")
    rt.close()
    return 0 if s["hit_rate"] >= 0.5 else 2


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    rt = Runtime.create()
    out = write_report(rt.store, rt.ontology, rt.settings, args.out)
    print(f"已生成交付报告：\n  {out['markdown']}\n  {out['html']}")
    rt.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    rt = Runtime.create()
    stats = rt.store.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n运行模式：{'online' if rt.settings.has_llm else 'offline'} · 编译器 {rt.embedder.name}")
    rt.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve_stdio

    serve_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fde", description="FDE 现场交付底座")
    parser.add_argument("--version", action="version", version=f"fde-kit {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="初始化数据目录与本体骨架")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("compile", help="编译资料到知识库")
    p.add_argument("--corpus", help="资料目录（默认读 FDE_CORPUS）")
    p.add_argument("--reset", action="store_true", help="清空旧快照后重建")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("search", help="只做检索，不看模型")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="提问（带引用）")
    p.add_argument("question")
    p.add_argument("--session", default="default")
    p.add_argument("--graph", action="store_true", help="用 LangGraph 引擎（需安装 fde-kit[langchain]）")
    p.add_argument("--json", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="跑评测集")
    p.add_argument("--golden", default="evals/golden.yaml")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--engine", default="serial", choices=["serial", "langgraph"])
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="生成交付验收报告")
    p.add_argument("--out", help="输出目录")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("stats", help="知识库概览")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="启动工作台")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="以 stdio 方式启动 MCP 服务")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        print("\n首次使用：fde init && fde compile && fde ask \"...\"")
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
