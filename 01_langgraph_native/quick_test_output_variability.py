"""
Prompt 优化对比实验 —— 让 LLM 稳定调用 save_memory

3 种 prompt × 2 种输入 = 6 组对比

变体:
  V0 baseline   — 原 prompt (温和引导)
  V1 strict     — 硬性规则 + 触发条件枚举 + 失败模式警告
  V2 few-shot   — V1 + 3 个 input/正确流程 示例
  V3 + 自检     — V2 + 承诺词反向校验 ("如果想说'记住了'但没调 tool,回退")

输入:
  I1 simple   — "我叫 Alice,在量化交易做研究员"
  I2 complex  — "我叫 Alice,我喜欢用 Python。顺便帮我写一个计算列表平均值的函数。"

观察:
  每组 tool 调用次数 + bot 回复 + 最终汇总表
"""

from __future__ import annotations
import sys
sys.path.insert(0, r"D:\work\Memory\01_langgraph_native")

from langchain_core.messages import HumanMessage
import agent as _agent


# ==================== Prompt 变体 ====================

V0 = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 当用户透露偏好/事实/约定时,主动调用 save_memory 工具记下,并简短确认"已记住"。
2. 回答前如果不确定用户背景,先调用 search_memory 检索一次。
3. 每轮我会把 top-3 相关记忆自动拼在下方;能直接用就别再多调一次 search_memory。
"""


V1 = """你是一个有长期记忆的对话助手。

【硬性记忆规则 - 必须遵守】

第一步:检查用户消息中是否包含以下任何一类内容
  (a) 自我介绍 - 姓名 / 身份 / 职业 / 所在公司 / 所属团队
  (b) 偏好声明 - "我喜欢 X" / "我希望 X" / "以后都用 X"
  (c) 事实陈述 - "我在 X 工作" / "我们用 Y 工具" / "我的项目是 Z"
  (d) 明确请求 - "请记住 X" / "记一下 Y"

第二步:只要命中其中任何一条,在生成自然语言回复之前,先调用 save_memory 工具
  - content 字段:用第三人称转述这条信息(例:"用户的名字是 Alice")
  - category 字段:preference / fact / rule / decision 之一
  - 如果一条消息里有多条信息(姓名+偏好),要分别调用 save_memory 多次

第三步:tool 返回后,再正常回复用户的其他诉求(打招呼/回答问题/写代码等)

【失败模式 - 必须避免】
× 嘴上说"已记住"但没调 save_memory tool
× 因为用户的请求里有"帮我写代码"就跳过 memory 记录
× 在没有命中条件时也调 save_memory(浪费)
"""


V2 = V1 + """

【示例】

输入: "我叫 Alice"
正确流程:
  1. tool_call: save_memory(content="用户的名字是 Alice", category="fact")
  2. tool_result: [memory saved] ...
  3. 回复: "你好,Alice!"

输入: "我喜欢简洁回答,顺便告诉我今天天气"
正确流程:
  1. tool_call: save_memory(content="用户喜欢简洁回答", category="preference")
  2. tool_result: [memory saved] ...
  3. 回复: "好的,会简洁回答。关于天气,你能告诉我所在城市吗?"

输入: "帮我写个函数"
正确流程:
  (无需调用 memory tool,直接回复代码)

输入: "我叫 Bob,我用 Rust 写代码,顺便帮我看下这段代码"
正确流程:
  1. tool_call: save_memory(content="用户的名字是 Bob", category="fact")
  2. tool_call: save_memory(content="用户用 Rust 写代码", category="preference")
  3. 回复: "你好 Bob!请把代码贴出来,我帮你看。"
"""


V3 = V2 + """

【最后自检 - 关键!!】

在生成最终回复前,自检一次:
  问: 我的回复里有没有"记住了" / "已记住" / "记下了" / "OK 没问题" 这种承诺词?
  问: 如果有,我之前调用过 save_memory tool 吗?

如果答案是"有承诺词但没调过 tool" —— 这是严重的"假装记忆"错误。
请回退,先调用 save_memory 让事实真正落到 store,再生成回复。

记住:用户看到"已记住"会预期下次对话能召回这条信息。
       如果你没真调 tool,下次它就召回不到,会被用户发现你在敷衍。
"""


# ==================== 输入 ====================

I1 = "我叫 Alice,在量化交易做研究员"
I2 = "我叫 Alice,我喜欢用 Python。顺便帮我写一个计算列表平均值的函数。"


# ==================== 实验 ====================

results = []  # 收集 (variant, input_label, tool_count, save_count, reply_preview)


def run(v_label: str, v_prompt: str, i_label: str, i_text: str, tid: str, uid: str):
    _agent.SYSTEM_PROMPT = v_prompt
    a, _ = _agent.build_agent()
    res = a.invoke(
        {"messages": [HumanMessage(content=i_text)]},
        config={"configurable": {"thread_id": tid, "user_id": uid}},
    )
    reply = res["messages"][-1].content
    tcs = [m for m in res["messages"] if getattr(m, "type", None) == "tool"]
    save_count = sum(1 for tc in tcs if tc.name == "save_memory")

    print(f"\n{'─' * 70}")
    print(f"[{v_label}] [{i_label}]")
    print(f"Input : {i_text}")
    print(f"Tool calls ({len(tcs)} total, {save_count} save_memory):")
    for tc in tcs:
        print(f"  · {tc.name}: {tc.content[:90]}")
    if not tcs:
        print("  (无)")
    preview = reply.replace("\n", " ")[:120]
    print(f"Reply : {preview}{'...' if len(reply) > 120 else ''}")

    results.append((v_label, i_label, len(tcs), save_count, preview))


variants = [("V0 baseline", V0), ("V1 strict", V1), ("V2 few-shot", V2), ("V3 + 自检", V3)]
inputs = [("I1 simple", I1), ("I2 complex", I2)]

for v_label, v_prompt in variants:
    for i_label, i_text in inputs:
        # 每组用独立 thread/user,避免互相污染
        tag = f"{v_label.split()[0]}_{i_label.split()[0]}"
        run(v_label, v_prompt, i_label, i_text, tid=tag, uid=f"u_{tag}")


# ==================== 汇总 ====================

print("\n" + "=" * 70)
print("Summary — save_memory 调用次数")
print("=" * 70)
print(f"{'Variant':<14} {'Input':<13} {'#save':<7} Reply preview")
print("-" * 70)
for v, i, _, sc, r in results:
    flag = "✓" if sc > 0 else "✗"
    print(f"{v:<14} {i:<13} {sc:<3}{flag:<4} {r[:50]}")
