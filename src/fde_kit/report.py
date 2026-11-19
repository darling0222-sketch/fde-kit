"""交付报告：把一次编译 + 一次评测，变成一份客户能签字的东西。

FDE 的交付物不是"一个能聊天的页面"，而是：
进了什么资料、抽到了什么、测得怎么样、哪里不行、下一个现场怎么复用。
这份报告就是 fde-kit 的最终产物，markdown 给人看，html 给客户看。
"""

from __future__ import annotations

import time
from pathlib import Path

from .compile.ontology import Ontology
from .config import Settings
from .retrieve.embed import build_embedder
from .store import Store

Block = tuple[str, object]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_blocks(
    store: Store,
    ontology: Ontology,
    settings: Settings,
    eval_run: dict | None = None,
    compile_run: dict | None = None,
) -> list[Block]:
    stats = store.stats()
    blocks: list[Block] = []

    compile_runs = [r for r in store.list_runs("compile", limit=5)]
    compile_run = compile_run or compile_runs[0] if compile_runs else compile_run
    eval_runs = [r for r in store.list_runs("eval", limit=5)]
    eval_run = eval_run or eval_runs[0] if eval_runs else eval_run

    mode = "online（模型）" if settings.has_llm else "offline（抽取式）"
    blocks.append(("h1", f"交付验收报告 · {settings.tenant}"))
    blocks.append(
        (
            "p",
            f"生成时间 {_now()} ｜ 知识库版本 r{stats['revision']} ｜ 运行模式 {mode} ｜ "
            f"本体 {ontology.name} v{ontology.version}",
        )
    )

    # ---------------------------------------------------------------- 概览
    blocks.append(("h2", "一、交付概览"))
    compile_meta = (compile_run or {}).get("meta", {}) if compile_run else {}
    rows = [
        ["资料数量", str(stats["assets"]), "进入知识库的原始文件数"],
        ["证据分块", str(stats["chunks"]), "可检索、可引用的最小单元"],
        ["上下文体量", f"{stats['tokens']:,} tokens", "全库可检索文本量"],
        ["向量", str(stats["vectors"]), "覆盖的分块数（应等于分块数）"],
        ["本体实体", str(stats["entities"]), f"覆盖 {len(stats['entity_types'])} 类实体"],
        ["最近编译耗时", f"{compile_meta.get('result', {}).get('duration', '-')} s", "一次全量编译"],
    ]
    blocks.append(("table", (["项目", "数值", "说明"], rows)))

    # ---------------------------------------------------------------- 语料
    blocks.append(("h2", "二、语料覆盖"))
    media_rows = [
        [media, str(info["count"]), f"{info['bytes'] / 1024:.0f} KB"]
        for media, info in sorted(stats["by_media"].items(), key=lambda kv: -kv[1]["count"])
    ]
    blocks.append(("table", (["类型", "数量", "体积"], media_rows or [["-", "0", "0 KB"]])))

    warnings = compile_meta.get("result", {}).get("warnings", []) if compile_meta else []
    if warnings:
        blocks.append(("p", f"以下 {len(warnings)} 份资料未纳入知识库（需客户补充或人工处理）："))
        blocks.append(("ul", [f"{w.get('uri', '?')} — {w.get('message', '')}" for w in warnings[:12]]))
    else:
        blocks.append(("p", "全部资料均已纳入知识库，无解析失败项。"))

    # ---------------------------------------------------------------- 本体
    blocks.append(("h2", "三、本体覆盖"))
    coverage = ontology.coverage(stats["entity_types"])
    if coverage:
        rows = [
            [c["type"], c["description"] or "-", str(c["hits"]), "已覆盖" if c["covered"] else "**未覆盖**"]
            for c in coverage
        ]
        blocks.append(("table", (["实体类型", "说明", "命中数", "状态"], rows)))
        missing = [c["type"] for c in coverage if not c["covered"]]
        if missing:
            blocks.append(("p", f"未覆盖实体类型：{'、'.join(missing)}。建议补充对应资料，或复核抽取正则。"))
    else:
        blocks.append(("p", "未加载本体文件，仅做通用文本切块。"))

    # ---------------------------------------------------------------- 评测
    blocks.append(("h2", "四、检索与问答评测"))
    if eval_run:
        summary = eval_run.get("meta", {}).get("summary", {})
        blocks.append(
            (
                "p",
                f"用例数 {summary.get('cases', 0)} ｜ hit@{eval_run.get('meta', {}).get('k', 5)} "
                f"{summary.get('hit_rate', 0):.0%} ｜ MRR {summary.get('mrr', 0):.3f} ｜ "
                f"关键词覆盖 {summary.get('keyword_coverage', 0):.0%} ｜ "
                f"引用有效率 {summary.get('citation_validity', 0):.0%} ｜ "
                f"平均延迟 {summary.get('avg_latency_ms', 0)} ms",
            )
        )
        compare_rows = [
            [
                "hit@k",
                f"{summary.get('hit_rate', 0):.0%}",
                f"{summary.get('lexical_hit_rate', 0):.0%}",
                "混合检索 vs 纯 BM25",
            ],
            ["MRR", f"{summary.get('mrr', 0):.3f}", f"{summary.get('lexical_mrr', 0):.3f}", "首个正确结果的名次"],
            ["关键词覆盖", f"{summary.get('keyword_coverage', 0):.0%}", "-", "答案是否覆盖必含口径"],
            ["引用有效率", f"{summary.get('citation_validity', 0):.0%}", "-", "回答里的 [S#] 是否真实存在"],
            ["有据可依率", f"{summary.get('grounded_rate', 0):.0%}", "-", "至少一处有效引用"],
        ]
        blocks.append(("table", (["指标", "混合检索", "纯词法基线", "口径"], compare_rows)))

        details = store.list_evals(eval_run["id"])
        if details:
            rows = []
            for d in details:
                no_answer = d.get("expect_no_answer")
                rows.append(
                    [
                        str(d["id"]),
                        d["question"][:34] + ("…" if len(d["question"]) > 34 else ""),
                        "—" if no_answer else ("✅" if d["hit"] else "❌"),
                        "—" if no_answer else f"{d['mrr']:.2f}",
                        f"{d['keyword_coverage']:.0%}",
                        "✅" if d["grounded"] else ("拒答✓" if no_answer else "❌"),
                    ]
                )
            blocks.append(("table", (["用例", "问题", "召回", "MRR", "覆盖", "有据"], rows)))
    else:
        blocks.append(("p", "尚未执行评测。建议先跑 `fde eval`，再出交付报告——没有评测数字的报告不算交付。"))

    # ---------------------------------------------------------------- 风险
    blocks.append(("h2", "五、失败与风险"))
    risks: list[str] = []
    if warnings:
        risks.append(f"{len(warnings)} 份资料未纳入，可能是扫描件 PDF 或加密文档，检索时会漏答。")
    if coverage and any(not c["covered"] for c in coverage):
        risks.append("本体存在未覆盖实体类型，说明这类资料还没进库。")
    if eval_run:
        summary = eval_run.get("meta", {}).get("summary", {})
        if summary.get("hit_rate", 1) < 0.8:
            risks.append("hit@k 低于 80%：先查语料缺口，再调切块与检索参数，别急着换模型。")
        if summary.get("citation_validity", 1) < 1:
            risks.append("存在无效引用编号，需在提示词或后处理里收紧引用约束。")
        if summary.get("mode") == "offline":
            risks.append("当前为离线抽取式回答，结论性表述能力有限，接入模型后应重跑本报告。")
    if not risks:
        risks.append("本轮未发现阻断性风险。")
    blocks.append(("ul", risks))

    # ---------------------------------------------------------------- 复用
    blocks.append(("h2", "六、下一个现场的复用清单"))
    blocks.append(
        (
            "ul",
            [
                "替换 ontology/*.yaml 中的实体类型与正则 —— 这是最值钱的一步，别跳过。",
                "把 FDE_CORPUS 指向新客户的资料目录，`fde compile --reset` 重建快照。",
                "重写 evals/golden.yaml：每个必答口径配一个用例，含一条“应该答不出”的反例。",
                "复用同一套 API / MCP，前端不用改：知识库换了，交付界面不变。",
                "把本次的 warnings 与失败用例整理成《客户资料补充清单》，这是二次进场的抓手。",
            ],
        )
    )
    blocks.append(
        (
            "p",
            f"—— 由 fde-kit 生成 ｜ 编译器 {build_embedder(settings).name} ｜ 报告口径见 docs/fde-playbook.md",
        )
    )
    return blocks


# ---------------------------------------------------------------------- 渲染
def render_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    for kind, payload in blocks:
        if kind == "h1":
            out.append(f"# {payload}\n")
        elif kind == "h2":
            out.append(f"\n## {payload}\n")
        elif kind == "p":
            out.append(f"{payload}\n")
        elif kind == "ul":
            out.append("\n".join(f"- {item}" for item in payload) + "\n")  # type: ignore[union-attr]
        elif kind == "table":
            headers, rows = payload  # type: ignore[misc]
            out.append("| " + " | ".join(headers) + " |")
            out.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                out.append("| " + " | ".join(str(c) for c in row) + " |")
            out.append("")
    return "\n".join(out)


def render_html(blocks: list[Block], title: str) -> str:
    body: list[str] = []
    for kind, payload in blocks:
        if kind == "h1":
            body.append(f"<h1>{payload}</h1>")
        elif kind == "h2":
            body.append(f"<h2>{payload}</h2>")
        elif kind == "p":
            body.append(f"<p>{_inline(payload)}</p>")
        elif kind == "ul":
            items = "".join(f"<li>{_inline(item)}</li>" for item in payload)  # type: ignore[union-attr]
            body.append(f"<ul>{items}</ul>")
        elif kind == "table":
            headers, rows = payload  # type: ignore[misc]
            head = "".join(f"<th>{h}</th>" for h in headers)
            body_rows = "".join(
                "<tr>" + "".join(f"<td>{_inline(str(c))}</td>" for c in row) + "</tr>" for row in rows
            )
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>")

    css = """
    body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;max-width:880px;
    margin:48px auto;padding:0 24px;color:#1f2328;line-height:1.7;background:#fff}
    h1{font-size:26px;font-weight:600;margin-bottom:8px}
    h2{font-size:17px;font-weight:600;margin-top:36px;padding-bottom:6px;border-bottom:1px solid #e6e8eb}
    p{font-size:14px;color:#3d444d}
    ul{font-size:14px;color:#3d444d}li{margin:4px 0}
    table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
    th{text-align:left;background:#f6f8fa;font-weight:600;color:#1f2328}
    th,td{border:1px solid #e6e8eb;padding:8px 10px}
    tr:nth-child(even) td{background:#fcfcfd}
    """
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>{''.join(body)}</body></html>"""


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    while "**" in text:
        text = text.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return text


# ---------------------------------------------------------------------- 落盘
def write_report(
    store: Store,
    ontology: Ontology,
    settings: Settings,
    out_dir: str | Path | None = None,
) -> dict:
    blocks = build_blocks(store, ontology, settings)
    out = Path(out_dir or settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = out / f"delivery-report-{stamp}.md"
    html_path = out / f"delivery-report-{stamp}.html"
    md = render_markdown(blocks)
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(render_html(blocks, f"交付验收报告 · {settings.tenant}"), encoding="utf-8")
    (out / "latest.md").write_text(md, encoding="utf-8")
    (out / "latest.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path), "content": md}
