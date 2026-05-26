"""
对话 Agent (Zep 版)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from memory_module import (
    build_checkpointer,
    build_zep_client,
    load_relevant_memories,
    make_memory_tools,
)


SYSTEM_PROMPT = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 系统会在每轮对话后把对话写入 Zep,Zep 服务端会自动抽取事实/更新知识图谱。
2. 仅当用户**明确要求"请记住X"**时,才调用 save_memory 工具立即写入。
3. 回答前如果不确定用户背景,可调用 search_memory 检索。
4. 每轮我会把 top-3 相关事实自动拼在下方;能直接用就别再多调一次 search_memory。
"""


def build_agent(model_name: str = "gpt-4o-mini") -> tuple[Any, Any]:
    llm = ChatOpenAI(model=model_name, temperature=0)
    checkpointer = build_checkpointer()
    client = build_zep_client()
    tools = make_memory_tools(client)

    def dynamic_prompt(state, config):
        user_id = config["configurable"].get("user_id", "anonymous")
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m.type == "human":
                last_user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break
        memory_block = load_relevant_memories(client, user_id, last_user_msg, k=3)
        return [SystemMessage(content=SYSTEM_PROMPT + memory_block)] + state["messages"]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=dynamic_prompt,
    )
    return agent, client
