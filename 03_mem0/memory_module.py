"""
Memory 模块 (mem0 版)

与 01_langgraph_native 的核心差异:
  - 长期记忆后端:           不再是 LangGraph store,改用 mem0.Memory
                            (mem0 自带向量库 + KV,这里配 chroma 本地落盘)
  - InjectedStore:          不再使用 —— mem0 不是 LangGraph 原生组件
                            tools 改用闭包绑定 mem0 实例
  - "事实抽取":             mem0 的卖点:m.add(messages) 会用 LLM 自动从对话提取事实
                            → demo 在每轮对话后自动 m.add(),不依赖 LLM 主动调 save_memory
  - search_memory tool:    仍然暴露,让 agent 也能主动检索
  - 短期记忆 checkpointer:  仍用 LangGraph MemorySaver(管对话状态,不抢 mem0 的活)

数据隔离: mem0 用 user_id / agent_id / run_id 三维隔离,demo 只用 user_id。
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from mem0 import Memory


# ============================================================
# 1. 短期记忆 — 沿用 LangGraph
# ============================================================
def build_checkpointer(sqlite_path: Optional[str] = None):
    if sqlite_path is None:
        return MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(sqlite_path)


# ============================================================
# 2. 长期记忆 — mem0
# ------------------------------------------------------------
# mem0 的 LLM/embedder/向量库都在 config 里声明:
#   - LLM:        用于自动提取事实(m.add 时调用)
#   - embedder:   用于 m.search 的语义检索
#   - vector_store: 这里用 chroma 本地落盘,生产可换 qdrant/pgvector/weaviate
# ============================================================
def build_memory(persist_dir: str = "./mem0_chroma") -> Memory:
    # 智谱走 OpenAI 兼容协议:llm + embedder 都设 openai_base_url 指向智谱网关
    zhipu_key = os.environ["ZHIPUAI_API_KEY"]
    zhipu_base = "https://open.bigmodel.cn/api/paas/v4/"
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "glm-4-flash",
                "temperature": 0,
                "api_key": zhipu_key,
                "openai_base_url": zhipu_base,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "embedding-2",  # 智谱 embedding,1024 维
                "api_key": zhipu_key,
                "openai_base_url": zhipu_base,
                "embedding_dims": 1024,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_demo",
                "path": persist_dir,
            },
        },
    }
    return Memory.from_config(config)


# ============================================================
# 3. Memory Tools (闭包绑定 mem0 实例)
# ------------------------------------------------------------
# 对比 01 的 InjectedStore: mem0 不是 LangGraph 一等公民,
# 没有 inject 机制,改用闭包工厂返回绑定好的 tool 列表。
# 这意味着 build_agent 需为每个 mem0 实例构造一组 tools(本 demo 单实例够用)。
# ============================================================
def make_memory_tools(mem: Memory):

    @tool
    def save_memory(content: str, *, config: RunnableConfig) -> str:
        """主动把一条事实写入长期记忆。

        注意:mem0 默认在每轮对话后会自动提取事实,这个 tool 只在 LLM
        判断"必须立即记住"时用(例如用户明确说"请记住X")。
        """
        user_id = config["configurable"].get("user_id", "anonymous")
        mem.add(messages=content, user_id=user_id)
        return f"[memory saved] {content}"

    @tool
    def search_memory(query: str, *, config: RunnableConfig, limit: int = 5) -> str:
        """从长期记忆按语义相似度检索事实。"""
        user_id = config["configurable"].get("user_id", "anonymous")
        # mem0 2.0: 用 filters 传 user_id,limit → top_k
        results = mem.search(query=query, filters={"user_id": user_id}, top_k=limit)
        items = results.get("results", []) if isinstance(results, dict) else results
        if not items:
            return "(无相关记忆)"
        return "\n".join(f"- {it.get('memory', it)}" for it in items)

    return [save_memory, search_memory]


# ============================================================
# 4. prompt 自动召回
# ============================================================
def load_relevant_memories(mem: Memory, user_id: str, query: str, k: int = 3) -> str:
    if not query:
        return ""
    try:
        # mem0 2.0 API
        results = mem.search(query=query, filters={"user_id": user_id}, top_k=k)
    except Exception:
        return ""
    items = results.get("results", []) if isinstance(results, dict) else results
    if not items:
        return ""
    bullets = "\n".join(f"- {it.get('memory', it)}" for it in items)
    return f"\n已知用户信息(从长期记忆召回):\n{bullets}\n"


# ============================================================
# 5. 对话后自动入库 (mem0 的核心用法)
# ------------------------------------------------------------
# 每轮 agent 回复后调一次,让 mem0 用 LLM 自动从对话里提取事实。
# 这是与 01 / 02 最大的行为差异:不靠 LLM 自觉调 save_memory。
# ============================================================
def ingest_turn(mem: Memory, user_id: str, user_text: str, bot_text: str) -> None:
    mem.add(
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ],
        user_id=user_id,
    )
