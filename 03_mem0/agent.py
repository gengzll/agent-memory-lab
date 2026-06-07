"""
对话 Agent (mem0 版)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from llm_factory import build_llm
from memory_module import (
    build_checkpointer,
    build_memory,
    load_relevant_memories,
    make_memory_tools,
)


SYSTEM_PROMPT = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 系统会在每轮对话后自动从对话里抽取事实存入长期记忆,你不必为常规事实主动调用 save_memory。
2. 仅当用户**明确要求"请记住X"**时,才调用 save_memory 工具立即写入。
3. 回答前如果不确定用户背景,可调用 search_memory 检索。
4. 每轮我会把 top-3 相关记忆自动拼在下方;能直接用就别再多调一次 search_memory。
"""


def build_agent(
    model_name: str | None = None,
    reset_memory: bool = False,
) -> tuple[Any, Any]:
    """装配 mem0 版 agent。

    Args:
        reset_memory: True 时清空持久化目录后重建 mem0 实例 —— 跨进程持久化下
            想要每次跑 demo 都从干净状态开始时打开。
    """
    llm = build_llm(model=model_name)
    checkpointer = build_checkpointer()
    mem = build_memory(reset=reset_memory)
    tools = make_memory_tools(mem)

    def dynamic_prompt(state, config):
        user_id = config["configurable"].get("user_id", "anonymous")
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m.type == "human":
                last_user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break
        memory_block = load_relevant_memories(mem, user_id, last_user_msg, k=3)
        return [SystemMessage(content=SYSTEM_PROMPT + memory_block)] + state["messages"]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=dynamic_prompt,
    )
    return agent, mem
