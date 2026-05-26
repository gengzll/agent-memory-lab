"""
两个 session 的演示 —— 证明长期记忆能跨 thread 生效。

Session 1 (thread_id=t1, user_id=alice)
  用户透露身份 + 偏好,期望 agent 调用 save_memory 写入 store。

Session 2 (thread_id=t2, user_id=alice)   ← 新 thread,短期记忆清零!
  用户提个会用到偏好的问题,期望:
    - dynamic_prompt 自动把 t1 写入的 memory 召回拼到 system
    - 或 agent 主动调用 search_memory
  → 回答应当遵循 alice 在 t1 透露的偏好。

跑法:
    set OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."
    pip install -r requirements.txt
    python run_demo.py
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
    # 打印 tool 调用轨迹,看 agent 有没有真的写/查 memory
    tool_calls = [
        m for m in result["messages"]
        if getattr(m, "type", None) == "tool"
    ]
    if tool_calls:
        for tc in tool_calls:
            print(f"  · tool[{tc.name}] -> {tc.content[:120]}")
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
        print(f"  - [{it.value.get('category', '?')}] {it.value['content']}")


def main() -> None:
    if not os.getenv("ZHIPUAI_API_KEY"):
        raise RuntimeError(
            "请先设置 ZHIPUAI_API_KEY。\n"
            "  PowerShell: $env:ZHIPUAI_API_KEY=\"<your-key>\""
        )

    agent, store = build_agent(model_name="glm-4-flash")

    print("=" * 70)
    print("Session 1 — alice 告诉 agent 一些事实和偏好")
    print("=" * 70)
    chat(agent, "我叫 Alice,在量化交易做研究员", thread_id="t1", user_id="alice")
    chat(agent, "我喜欢回答尽量简短,代码示例只用 Python", thread_id="t1", user_id="alice")

    dump_store(store, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新 thread (t2),短期记忆已清空")
    print("        看长期记忆是否能跨 thread 生效:回答应当用 Python + 简短风格")
    print("=" * 70)
    chat(agent, "帮我写一段计算夏普比率的代码", thread_id="t2", user_id="alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换个用户 bob,同一个 agent 实例")
    print("        看 namespace 隔离:bob 应当看不到 alice 的 memory")
    print("=" * 70)
    chat(agent, "你知道我叫什么吗?", thread_id="t3", user_id="bob")

    dump_store(store, "bob")


if __name__ == "__main__":
    main()
