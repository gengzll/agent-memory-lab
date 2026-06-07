"""
两个 session 演示 (mem0 版)

与 01/02 的关键差异:
  chat() 在 agent.invoke 后调一次 ingest_turn(),让 mem0 自动从对话里
  抽取事实存进 memory —— 不依赖 LLM 主动调 save_memory tool。

CLI:
  python run_demo.py                  # 增量模式,沿用上次的持久化记忆
  python run_demo.py --reset          # 清空整个 ./mem0_chroma 后重跑
  python run_demo.py --reset-user alice  # 只清 alice 的记忆,保留 bob
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from agent import build_agent
from llm_factory import get_api_key
from memory_module import ingest_turn, reset_user_memory


def chat(agent, mem, text: str, thread_id: str, user_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content
    print(f"\n[user={user_id} thread={thread_id}]")
    print(f"  User: {text}")
    print(f"  Bot : {reply}")
    tool_calls = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    for tc in tool_calls:
        print(f"  · tool[{tc.name}] -> {str(tc.content)[:120]}")

    # mem0 核心用法:对话结束后自动抽取事实
    ingest_turn(mem, user_id, text, reply)
    return reply


def dump_memory(mem, user_id: str) -> None:
    print(f"\n--- mem0 当前 user_id={user_id} 的全部 memory ---")
    try:
        # mem0 2.0: get_all 也改成 filters
        items = mem.get_all(filters={"user_id": user_id})
        items = items.get("results", []) if isinstance(items, dict) else items
    except Exception as e:
        print(f"  (读取失败: {e})")
        return
    if not items:
        print("  (空)")
        return
    for it in items:
        print(f"  - {it.get('memory', it)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mem0 demo")
    p.add_argument(
        "--reset",
        action="store_true",
        help="清空整个 ./mem0_chroma 持久化目录后从零开始(所有用户的记忆都会丢)",
    )
    p.add_argument(
        "--reset-user",
        metavar="USER_ID",
        help="只清空指定 user_id 的记忆(保留其他用户)。与 --reset 互斥",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    get_api_key()
    agent, mem = build_agent(reset_memory=args.reset)

    if args.reset_user and not args.reset:
        n = reset_user_memory(mem, args.reset_user)
        print(f"[mem0] 已清空 user_id={args.reset_user} 的 {n} 条记忆")

    print("=" * 70)
    print("Session 1 — alice 告诉 agent 一些事实和偏好")
    print("=" * 70)
    chat(agent, mem, "我叫 Alice,在量化交易做研究员", "t1", "alice")
    chat(agent, mem, "我喜欢回答尽量简短,代码示例只用 Python", "t1", "alice")
    dump_memory(mem, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新 thread (t2),短期记忆已清空")
    print("=" * 70)
    chat(agent, mem, "帮我写一段计算夏普比率的代码", "t2", "alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换用户 bob,验证 user_id 隔离")
    print("=" * 70)
    chat(agent, mem, "你知道我叫什么吗?", "t3", "bob")
    dump_memory(mem, "bob")


if __name__ == "__main__":
    main()
