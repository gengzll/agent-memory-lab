"""
两个 session 演示 —— 与 01 完全相同的脚本,验证 langmem 等价行为。
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from agent import build_agent


def chat(agent, text: str, thread_id: str, user_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content
    print(f"\n[user={user_id} thread={thread_id}]")
    print(f"  User: {text}")
    print(f"  Bot : {reply}")
    tool_calls = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    for tc in tool_calls:
        print(f"  · tool[{tc.name}] -> {str(tc.content)[:120]}")
    return reply


def dump_store(store, user_id: str) -> None:
    print(f"\n--- store 当前内容 (namespace=('memories','{user_id}')) ---")
    try:
        items = store.search(("memories", user_id), query="", limit=20)
    except Exception:
        items = store.search(("memories", user_id), limit=20)
    if not items:
        print("  (空)")
        return
    for it in items:
        content = it.value.get("content") if isinstance(it.value, dict) else it.value
        print(f"  - {content}")


def main() -> None:
    if not os.getenv("ZHIPUAI_API_KEY"):
        raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量")

    agent, store = build_agent(model_name="glm-4-flash")

    print("=" * 70)
    print("Session 1 — alice 告诉 agent 一些事实和偏好")
    print("=" * 70)
    chat(agent, "我叫 Alice,在量化交易做研究员", thread_id="t1", user_id="alice")
    chat(agent, "我喜欢回答尽量简短,代码示例只用 Python", thread_id="t1", user_id="alice")
    dump_store(store, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新 thread (t2),短期记忆已清空")
    print("=" * 70)
    chat(agent, "帮我写一段计算夏普比率的代码", thread_id="t2", user_id="alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换用户 bob,验证 namespace 隔离")
    print("=" * 70)
    chat(agent, "你知道我叫什么吗?", thread_id="t3", user_id="bob")
    dump_store(store, "bob")


if __name__ == "__main__":
    main()
