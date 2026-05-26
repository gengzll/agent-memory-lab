"""
对话 Agent (Letta 版)

与 LangGraph create_react_agent 的对比:
  LangGraph                    Letta
  ─────────                    ─────
  agent.invoke(messages, ...)  client.agents.messages.create(agent_id, messages)
  thread_id                    agent_id (per-user agent,持久化)
  store / checkpointer         Letta server 自带 (PostgreSQL 后端)
  prompt 字符串                memory_blocks (persona + human + ...)
  自己写 tool                  内置 core_memory_* / archival_memory_* 等

注意:Letta 的 LLM 配置在 server 端(server 启动时读环境变量),
client 只指定 model handle (例如 "openai/gpt-4o-mini") + embedding handle。
"""

from __future__ import annotations

from letta_client import Letta


# 默认 LLM / Embedding handle —— 走 OpenAI 兼容,可由 server 端 env 切换到 ZhipuAI 等
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING = "openai/text-embedding-3-small"

DEFAULT_PERSONA = """You are a helpful assistant with long-term memory.

When the user shares personal info (name / job / preferences), you proactively
update the `human` memory block by calling `core_memory_append` or
`core_memory_replace` so it stays accurate across sessions.

For broader factual knowledge they share that doesn't fit in core memory,
you call `archival_memory_insert`.

When you need to recall something from earlier conversations, call
`conversation_search`. To find facts in your archive, call
`archival_memory_search`.
"""


def ensure_agent(
    client: Letta,
    user_name: str = "unknown_user",
    model: str = DEFAULT_MODEL,
    embedding: str = DEFAULT_EMBEDDING,
):
    """获取或创建 agent。

    Letta agent 是**持久化的有 ID 的实体**,跟 LangGraph 的 ephemeral graph 不同。
    多 session / 多设备访问同一个 agent_id,memory 自动连续。
    """
    # 先查是否已有同名 agent
    existing = client.agents.list(name=f"agent_for_{user_name}", limit=1)
    if existing:
        return existing[0]

    return client.agents.create(
        name=f"agent_for_{user_name}",
        memory_blocks=[
            {"label": "persona", "value": DEFAULT_PERSONA},
            {
                "label": "human",
                "value": (
                    f"This is the user. Their identifier is '{user_name}'. "
                    "Update this block as you learn their actual name, "
                    "role, preferences, and ongoing context."
                ),
            },
        ],
        model=model,
        embedding=embedding,
    )
