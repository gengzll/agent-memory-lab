"""
两个 session 演示 (mem0 版)

与 01/02 的关键差异:
  chat() 在 agent.invoke 后调一次 ingest_turn(),让 mem0 自动从对话里
  抽取事实存进 memory —— 不依赖 LLM 主动调 save_memory tool。
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from agent import build_agent
from memory_module import ingest_turn


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


def main() -> None:
    if not os.getenv("ZHIPUAI_API_KEY"):
        raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量")

    agent, mem = build_agent(model_name="glm-4-flash")

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
