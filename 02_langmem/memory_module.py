"""
Memory 模块 (langmem 版)

与 01_langgraph_native 的差异:
  - 短期记忆 checkpointer:  相同 (MemorySaver / SqliteSaver)
  - 长期记忆 store:          相同 (InMemoryStore + embeddings) ← langmem 直接基于 LangGraph store
  - save / search tool:      不再手写,改用 langmem 工厂函数
      * create_manage_memory_tool   → 支持 create / update / delete,signature 更完整
      * create_search_memory_tool   → 内置 query / filter 支持
  - namespace 模板:          langmem 支持 "{user_id}" 占位,自动从 RunnableConfig 注入
  - prompt 自动注入:         保留 (langmem 本身不强制做这事,与 tool 调用互补)

可选高级用法(本 demo 未启用,顶部注释提示):
  - create_memory_store_manager:  让 LLM 在后台批量整理 memory(去重/归并)
  - ReflectionExecutor:           异步反思,不阻塞主对话
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from llm_factory import build_embeddings
from persistent_store import PersistentInMemoryStore, reset_user


# ============================================================
# 1. 短期记忆 — 与 01 完全相同
# ============================================================
def build_checkpointer(sqlite_path: Optional[str] = None):
    if sqlite_path is None:
        return MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(sqlite_path)


# ============================================================
# 2. 长期记忆 — 与 01 相同 (langmem 复用 LangGraph store)
# ============================================================
def build_store(
    use_embeddings: bool = True,
    persist_path: Optional[str] = None,
) -> BaseStore:
    """长期记忆 store。persist_path 非 None 时跨进程持久化到 JSON 文件。"""
    index = None
    if use_embeddings:
        try:
            embeddings, dims = build_embeddings()
            index = {"embed": embeddings, "dims": dims, "fields": ["content"]}
        except Exception:
            index = None

    if persist_path:
        return PersistentInMemoryStore(persist_path=persist_path, index=index)
    if index is not None:
        return InMemoryStore(index=index)
    return InMemoryStore()


# ============================================================
# 3. Memory Tools — 由 langmem 工厂生成,无需手写函数体
# ------------------------------------------------------------
# namespace 里的 "{user_id}" 是 langmem 模板语法,
# invoke 时自动从 config["configurable"]["user_id"] 取值。
#
# 对比 01 的手写 tool:
#   - 01 手写:     @tool save_memory(content, category) -> store.put
#   - langmem:    工厂直接给 create / update / delete 统一接口
# ============================================================
MEMORY_TOOLS = [
    # 开放 create/update/delete 三种 action。
    # update/delete 需要 id —— 通过 load_relevant_memories() 在 system prompt
    # 里把已有 memory 的 id 一起注入,LLM 就不必再走 "search 拿 id → update"
    # 两步规划,直接引用即可。这把多步推理卸到代码侧,对弱模型更鲁棒。
    create_manage_memory_tool(namespace=("memories", "{user_id}")),
    create_search_memory_tool(namespace=("memories", "{user_id}")),
]


# ============================================================
# 4. prompt 自动召回 — 与 01 相同
# ============================================================
def load_relevant_memories(
    store: BaseStore, user_id: str, query: str, k: int = 3
) -> str:
    if not query:
        return ""
    namespace = ("memories", user_id)
    try:
        results = store.search(namespace, query=query, limit=k)
    except Exception:
        results = store.search(namespace, limit=k)
    if not results:
        return ""
    # langmem 写入的 value 字段是 {"content": "...", ...}
    bullets = []
    for r in results:
        content = r.value.get("content") if isinstance(r.value, dict) else str(r.value)
        # id 一起拼出来,LLM 想 update/delete 时可直接引用,无需先 search
        bullets.append(f"- [id={r.key}] {content}")
    return (
        "\n已知用户信息(从长期记忆召回,要修正/删除某条时引用对应 id):\n"
        + "\n".join(bullets) + "\n"
    )
