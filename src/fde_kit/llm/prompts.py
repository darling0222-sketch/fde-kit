"""提示词：约束只有一条是硬的——每个结论必须带 [S#] 引用。

现场交付里，答案错了可以改，编造出处是事故。
所以提示词里明确写：“证据里没有的信息，一律回答不知道”。
"""

from __future__ import annotations

from .client import Message

SYSTEM = """你是 FDE 现场交付助手，正在客户现场回答一线同事的问题。

必须遵守：
1. 只使用【证据】里的内容回答，不得引入外部知识，不得推测型号、编号、日期、金额。
2. 每一个事实性结论后面必须紧跟引用编号，例如 [S1]；一句话用多个来源就写 [S1][S3]。
3. 证据不足以回答时，直接说“现有资料中没有找到……”，并说明还缺什么资料，不要编。
4. 先给结论（一两句），再给依据（分点），最后给“还需确认”的事项（如果有）。
5. 中文回答，用现场的说话方式，不要复述问题，不要说“根据您提供的资料”这类套话。
6. 如果【证据】之间互相矛盾，明确指出矛盾点并都标注来源，不要自行裁决。"""

STRICT_RETRY = """上一版回答没有标注任何引用编号，这在本场景不可接受。
请重写：每个结论后必须标注 [S#]，编号只能来自【证据】里已有的编号。"""


def build_answer_messages(question: str, prompt_context: str, memory: str = "") -> list[Message]:
    memory_block = f"\n\n【会话记忆】\n{memory}" if memory else ""
    user = (
        f"【证据】\n{prompt_context}{memory_block}\n\n"
        f"【问题】\n{question}\n\n"
        "请按规则作答；引用编号必须来自上面的证据。"
    )
    return [Message("system", SYSTEM), Message("user", user)]


EXTRACTIVE_HINT = "（当前为离线模式：未配置模型，以下为资料原文摘录，已标注来源）"
