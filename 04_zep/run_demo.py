"""
两个 session 演示 (Zep 版)

与 03_mem0 类似:chat() 在 agent.invoke 后写入 zep,服务端自动抽取事实。
额外:每个 session 启动前要 ensure_user / ensure_session。

前置:
    # 1. Zep 服务 —— ⚠️ Vendor status (as of 2026-05):
    #    Zep CE (self-host docker) 已被官方废弃。推荐用 Zep Cloud:
    export ZEP_API_KEY=<your-zep-cloud-key>          # https://app.getzep.com/api-keys
    #
    #    如必须 self-host 图谱能力,改用 Graphiti(API 与 Zep client 不同,
    #    本 demo 代码需重写):https://github.com/getzep/graphiti

    # 2. 本 demo client 端 LLM(跟 01/02/03 一致,走智谱)
    export ZHIPUAI_API_KEY=...

    ⚠️ 注意:Zep server **自己也要 LLM** 做事实抽取/图谱构建,
    Zep Cloud 已自带 LLM(无需额外配),self-host(如 Graphiti)需自己配 LLM key。
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from agent import build_agent
from memory_module import ensure_session, ensure_user, ingest_turn


def chat(agent, client, text: str, thread_id: str, user_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content
    print(f"\n[user={user_id} thread={thread_id}]")
    print(f"  User: {text}")
    print(f"  Bot : {reply}")
    tool_calls = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    for tc in tool_calls:
        print(f"  · tool[{tc.name}] -> {str(tc.content)[:120]}")

    # zep 核心用法:对话写入 → 服务端自动抽取 facts
    ingest_turn(client, thread_id, text, reply)
    return reply


def dump_facts(client, user_id: str) -> None:
    print(f"\n--- Zep 抽取出的 facts (user_id={user_id}) ---")
    try:
        res = client.memory.search_sessions(
            user_id=user_id,
            text="user profile and preferences",
            search_scope="facts",
            limit=20,
        )
    except Exception as e:
        print(f"  (读取失败: {e})")
        return
    results = getattr(res, "results", None) or []
    if not results:
        print("  (空 —— 注意:Zep 抽取是异步的,刚写入的事实可能要几秒后才检索得到)")
        return
    for r in results:
        fact = getattr(r, "fact", None) or getattr(r, "content", None) or str(r)
        print(f"  - {fact}")


def main() -> None:
    if not os.getenv("ZHIPUAI_API_KEY"):
        raise RuntimeError("请先设置 ZHIPUAI_API_KEY 环境变量(LLM 用智谱 glm-4-flash)")

    agent, client = build_agent(model_name="glm-4-flash")

    # Zep 需要先创建 user 和 session
    ensure_user(client, "alice")
    ensure_user(client, "bob")
    ensure_session(client, "t1", "alice")
    ensure_session(client, "t2", "alice")
    ensure_session(client, "t3", "bob")

    print("=" * 70)
    print("Session 1 — alice 告诉 agent 一些事实和偏好")
    print("=" * 70)
    chat(agent, client, "我叫 Alice,在量化交易做研究员", "t1", "alice")
    chat(agent, client, "我喜欢回答尽量简短,代码示例只用 Python", "t1", "alice")

    # 给 zep 服务端一点时间抽取 facts(异步处理)
    import time; time.sleep(3)
    dump_facts(client, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新 thread (t2),短期记忆已清空")
    print("=" * 70)
    chat(agent, client, "帮我写一段计算夏普比率的代码", "t2", "alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换用户 bob,验证 user_id 隔离")
    print("=" * 70)
    chat(agent, client, "你知道我叫什么吗?", "t3", "bob")
    dump_facts(client, "bob")


if __name__ == "__main__":
    main()
