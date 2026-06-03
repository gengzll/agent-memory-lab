"""
Letta demo —— 完整 agent runtime,不是 LangGraph plug-in

测试场景(类比 demo 01-04,但 Letta 是不同范式):
  Session 1: alice 透露身份+偏好
             → 期望 Letta 内部调 core_memory_append 更新 human block
             → 同时可能调 archival_memory_insert 存到长期档案
  Session 2: 新消息触发,Letta 自动从 memory 召回回答
             (Letta 没有"跨 thread"概念 —— agent_id 持续,自动续聊)
  Session 3: 不同 user_name → 新 agent_id,完全隔离

前置 —— ⚠️ Vendor status (as of 2026-05):
  Letta 官方主文档以 Letta Cloud 为主推荐(本地 server 仍可用但不强调)。

  ✅ 推荐:Letta Cloud
    export LETTA_API_TOKEN=<your-token>
    export LETTA_BASE_URL=https://api.letta.com
    python run_demo.py

  ⚠ 可选:本地 server(以 docs.letta.com 为准)
    # 终端 1
    letta server
    # 终端 2
    export OPENAI_API_KEY=sk-...
    # 或 ZhipuAI(走 OpenAI 兼容):
    export OPENAI_API_KEY=<zhipu-key>
    export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
    python run_demo.py
"""

from __future__ import annotations

from letta_client.errors import ApiError

from agent import ensure_agent
from memory_module import build_client


def chat(client, agent, text: str) -> None:
    """发一句话,打印 Bot 回复 + 显示 Letta 内部 tool 调用轨迹。"""
    print(f"\n[{agent.name}]")
    print(f"  User: {text}")

    response = client.agents.messages.create(
        agent_id=agent.id,
        messages=[{"role": "user", "content": text}],
    )

    # Letta 返回的 messages 是异构的:reasoning / tool_call / tool_response / assistant
    for msg in response.messages:
        mtype = getattr(msg, "message_type", None)
        if mtype == "assistant_message":
            print(f"  Bot : {msg.content}")
        elif mtype == "reasoning_message":
            # 内部思考,可选打印
            pass
        elif mtype == "tool_call_message":
            tc = msg.tool_call
            args_preview = (tc.arguments or "")[:120].replace("\n", " ")
            print(f"  · tool_call[{tc.name}] -> {args_preview}")
        elif mtype == "tool_return_message":
            ret_preview = (msg.tool_return or "")[:120].replace("\n", " ")
            print(f"  · tool_return -> {ret_preview}")


def dump_core_memory(client, agent) -> None:
    """打印 agent 的 core memory blocks(persona + human + ...)。"""
    print(f"\n--- {agent.name} 的 core memory blocks ---")
    blocks = client.agents.blocks.list(agent_id=agent.id)
    for b in blocks:
        print(f"  [{b.label}] ({len(b.value)} chars):")
        for line in b.value.splitlines()[:6]:
            print(f"    {line}")


def main() -> None:
    try:
        client = build_client()
        # 探活:列一下 agents 看 server 是否在
        client.agents.list(limit=1)
    except (ApiError, ConnectionError) as e:
        raise RuntimeError(
            "无法连接到 Letta server。\n"
            "  ✅ 推荐:用 Letta Cloud → export LETTA_API_TOKEN=... + "
            "LETTA_BASE_URL=https://api.letta.com\n"
            "  ⚠ 或本地:在另一个终端跑 `letta server`(以 docs.letta.com 为准)\n"
            f"详情: {e}"
        ) from e

    # Session 1: alice 透露信息
    print("=" * 70)
    print("Session 1 — alice 透露身份+偏好")
    print("=" * 70)
    alice = ensure_agent(client, user_name="alice")
    chat(client, alice, "我叫 Alice,在量化交易做研究员")
    chat(client, alice, "我喜欢回答尽量简短,代码示例只用 Python")
    dump_core_memory(client, alice)

    # Session 2: 同一 agent,新话题。Letta 自动从 memory 召回
    print("\n" + "=" * 70)
    print("Session 2 — 同一 agent 继续(没有'跨 thread'概念)")
    print("=" * 70)
    chat(client, alice, "帮我写一段计算夏普比率的代码")

    # Session 3: 不同 user_name → 新 agent,验证隔离
    print("\n" + "=" * 70)
    print("Session 3 — bob 是新 agent,完全隔离")
    print("=" * 70)
    bob = ensure_agent(client, user_name="bob")
    chat(client, bob, "你知道我叫什么吗?")
    dump_core_memory(client, bob)


if __name__ == "__main__":
    main()
