"""
对话 Agent (langmem 版) —— 与 01 几乎完全相同,只是 tools 由 langmem 工厂生成。
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
    MEMORY_TOOLS,
    build_checkpointer,
    build_store,
    load_relevant_memories,
)


SYSTEM_PROMPT = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 处理新的偏好/事实/约定时,根据情况调用 manage_memory:
   - 全新事实           → action="create", content=<新内容>(不要带 id)
   - 修正/补充下方某条已有 memory → action="update", id=<那条的 id>, content=<新内容>
   - 用户明确撤销/否定下方某条     → action="delete", id=<那条的 id>
   操作后简短确认"已记住"/"已更新"/"已删除"。
2. 回答前如果不确定用户背景,先调用 search_memory 检索。
3. 每轮我会把 top-3 相关记忆自动拼在下方(带 id);能直接用就别再多调一次 search_memory。
"""


def build_agent(
    model_name: str | None = None,
    sqlite_path: str | None = None,
    use_embeddings: bool = True,
    persist_path: str | None = None,
) -> tuple[Any, Any]:
    """persist_path: 长期记忆 store 的 JSON 持久化文件;None 时纯内存。"""
    llm = build_llm(model=model_name)
    checkpointer = build_checkpointer(sqlite_path=sqlite_path)
    store = build_store(use_embeddings=use_embeddings, persist_path=persist_path)

    def dynamic_prompt(state, config):
        user_id = config["configurable"].get("user_id", "anonymous")
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m.type == "human":
                last_user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break
        memory_block = load_relevant_memories(store, user_id, last_user_msg, k=3)
        return [SystemMessage(content=SYSTEM_PROMPT + memory_block)] + state["messages"]

    agent = create_react_agent(
        model=llm,
        tools=MEMORY_TOOLS,
        checkpointer=checkpointer,
        store=store,
        prompt=dynamic_prompt,
    )
    return agent, store
