"""
Prompt 优化对比实验 —— 让 LLM 稳定调用 save_memory

实验设计:
  4 种 prompt 变体(V0/V1/V2/V3) × 2 种输入(I1/I2) × N 次重复
  = 8N 次 invoke

每组重复 N 次以降低单次抽样方差。报告 mean ± std + 两种命中率:
  hit_rate     = N 次中 save_memory 至少调用 1 次的比率(衡量"漏调")
  perfect_rate = N 次中 save_memory 调用次数 = 期望次数的比率(衡量"全调对")

跑法:
    $env:OPENAI_API_KEY = "<key>"     # 智谱默认,或 OPENAI_BASE_URL=https://api.openai.com/v1 等
    python experiments/prompt_variants.py            # N=5 (默认)
    python experiments/prompt_variants.py --n 10     # 提高样本量,但 API 调用翻倍
    python experiments/prompt_variants.py --n 3 --only V0,V2   # 只跑指定变体

API 成本估算(glm-4-flash): N=5 全跑 ≈ 40 次 invoke ≈ 0.04 元
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_langgraph_native"))

from langchain_core.messages import HumanMessage

import agent as _agent
from llm_factory import get_api_key, get_model


# ============================================================
# Prompt 变体 —— 每个变体在上一个基础上累加,看哪一步带来最大提升
# ============================================================

V0_DESIGN = "温和引导(baseline)。陈述记忆使用规则,不强制流程。"
V0 = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 当用户透露偏好/事实/约定时,主动调用 save_memory 工具记下,并简短确认"已记住"。
2. 回答前如果不确定用户背景,先调用 search_memory 检索一次。
3. 每轮我会把 top-3 相关记忆自动拼在下方;能直接用就别再多调一次 search_memory。
"""


V1_DESIGN = (
    "硬性规则。把'何时调 save_memory'展开成'触发条件清单 + 失败模式警告',"
    "用第一步/第二步/第三步明确执行顺序,目标:让 LLM 不再'嘴上说已记住但漏调 tool'。"
)
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


V2_DESIGN = (
    "V1 + 4 个 few-shot 示例(简单声明 / 复合声明 + 任务 / 纯任务无需记 / "
    "多事实复合)。让模型按示例'临摹'调用形式,目标:降低复杂输入下的漏调。"
)
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


V3_DESIGN = (
    "V2 + 反向自检。生成回复前用'承诺词反向校验'让模型自查:"
    "'我有没有写已记住但其实没调 tool?'目标:堵住'假装记忆'这个最常见的失败模式。"
)
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


VARIANTS = [
    ("V0_baseline", V0, V0_DESIGN),
    ("V1_strict",   V1, V1_DESIGN),
    ("V2_few_shot", V2, V2_DESIGN),
    ("V3_selfcheck", V3, V3_DESIGN),
]


# ============================================================
# 输入 —— 设计目的:看 LLM 在简单 vs 含干扰任务的复杂输入下表现差距
# ============================================================

I1_TEXT = "我叫 Alice,在量化交易做研究员"
I1_DESIGN = (
    "**简单输入** — 纯自我介绍,没有任何其他诉求。"
    "理论上 LLM 应该 100% 命中 save_memory,因为没有任何干扰任务争夺注意力。"
)
I1_EXPECTED_SAVES = 1  # 期望调用次数:姓名+职业作为一条事实存

I2_TEXT = "我叫 Alice,我喜欢用 Python。顺便帮我写一个计算列表平均值的函数。"
I2_DESIGN = (
    "**复杂输入** — 同时包含 2 条待存事实(姓名 + 编程偏好)和 1 个**无关编程任务**。"
    "故意混入编程请求,看 LLM 会不会'被任务带跑'而漏调 save_memory。"
    "这是生产场景常见模式:用户的诉求里事实和任务混杂。"
)
I2_EXPECTED_SAVES = 2  # 期望调用次数:姓名 + 编程偏好

INPUTS = [
    ("I1_simple",  I1_TEXT, I1_DESIGN, I1_EXPECTED_SAVES),
    ("I2_complex", I2_TEXT, I2_DESIGN, I2_EXPECTED_SAVES),
]


# ============================================================
# 单次跑 + 统计
# ============================================================

def run_once(prompt: str, text: str, tid: str, uid: str) -> int:
    """单次 invoke,返回该轮 save_memory 实际被调用的次数。"""
    _agent.SYSTEM_PROMPT = prompt
    a, _ = _agent.build_agent()
    res = a.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": tid, "user_id": uid}},
    )
    tool_msgs = [m for m in res["messages"] if getattr(m, "type", None) == "tool"]
    return sum(1 for tc in tool_msgs if tc.name == "save_memory")


def run_group(
    v_label: str,
    v_prompt: str,
    i_label: str,
    i_text: str,
    expected: int,
    n: int,
) -> dict:
    """同一 prompt × input 跑 N 次,返回汇总。"""
    counts = []
    for k in range(n):
        # 每次跑独立 user_id 防 dynamic_prompt 召回干扰
        tid = f"{v_label}_{i_label}_{k}"
        uid = f"u_{tid}"
        try:
            c = run_once(v_prompt, i_text, tid, uid)
        except Exception as e:
            print(f"  ! run {k+1}/{n} failed: {e}")
            c = 0
        counts.append(c)
        print(f"  [{v_label} × {i_label}] run {k+1}/{n}: save_memory called {c} 次")

    mean = statistics.mean(counts)
    std = statistics.pstdev(counts) if n > 1 else 0.0
    hit_rate = sum(1 for c in counts if c >= 1) / n            # 至少调一次
    perfect_rate = sum(1 for c in counts if c == expected) / n  # 调对次数
    return {
        "variant": v_label,
        "input": i_label,
        "n": n,
        "expected": expected,
        "counts": counts,
        "mean": mean,
        "std": std,
        "hit_rate": hit_rate,
        "perfect_rate": perfect_rate,
    }


# ============================================================
# 报告输出
# ============================================================

def print_design_recap() -> None:
    print("=" * 78)
    print("Prompt 变体设计")
    print("=" * 78)
    for label, _, design in VARIANTS:
        print(f"\n  {label}")
        print(f"    {design}")

    print()
    print("=" * 78)
    print("输入设计")
    print("=" * 78)
    for label, text, design, expected in INPUTS:
        print(f"\n  {label}  (期望 save_memory 调用 {expected} 次)")
        print(f"    输入文本: {text}")
        print(f"    {design}")


def print_results(results: list[dict]) -> None:
    print()
    print("=" * 78)
    print(f"汇总  ({results[0]['n']} 次重复)")
    print("=" * 78)
    header = f"{'Variant':<14} {'Input':<12} {'mean ± std':<14} {'hit_rate':<10} {'perfect_rate':<13} expected"
    print(header)
    print("-" * len(header))
    for r in results:
        ms = f"{r['mean']:.2f} ± {r['std']:.2f}"
        hr = f"{r['hit_rate']*100:.0f}%"
        pr = f"{r['perfect_rate']*100:.0f}%"
        print(f"{r['variant']:<14} {r['input']:<12} {ms:<14} {hr:<10} {pr:<13} {r['expected']}")

    print()
    print("术语:")
    print("  mean ± std    — N 次跑中 save_memory 被调用次数的平均值和标准差")
    print("  hit_rate      — N 次中至少调用 1 次的比率('没漏'的概率)")
    print("  perfect_rate  — N 次中调用次数 = expected 的比率('全调对'的概率)")


# ============================================================
# 入口
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt 变体命中率实验")
    p.add_argument(
        "--n", type=int, default=5,
        help="每组 (prompt × input) 跑多少次。默认 5,提高方差稳定性但成本翻倍。",
    )
    p.add_argument(
        "--only",
        help="只跑指定变体,逗号分隔。例如 --only V0_baseline,V2_few_shot",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    get_api_key()  # 没配 key 在此抛错

    selected = set(args.only.split(",")) if args.only else None
    variants = [v for v in VARIANTS if selected is None or v[0] in selected]
    total_runs = len(variants) * len(INPUTS) * args.n

    print(f"模型: {get_model()}    样本量 N={args.n}    总 invoke 数: {total_runs}")
    print_design_recap()

    print()
    print("=" * 78)
    print("逐组采样")
    print("=" * 78)
    results: list[dict] = []
    for v_label, v_prompt, _ in variants:
        for i_label, i_text, _, expected in INPUTS:
            print(f"\n→ {v_label} × {i_label}  (期望 {expected})")
            r = run_group(v_label, v_prompt, i_label, i_text, expected, args.n)
            results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
