"""
Memory 模块 demo —— LangGraph 三层架构

┌──────────────────────────────┬──────────────────────────────────┐
│  短期记忆 Short-term         │  长期记忆 Long-term              │
│  ────────────────────────    │  ──────────────────────────────  │
│  Checkpointer                │  Store (with embeddings)         │
│  - 按 thread_id 隔离         │  - 按 namespace 隔离 (如 user_id)│
│  - 存 graph state (messages) │  - 存事实 (content + metadata)   │
│  - MemorySaver / SqliteSaver │  - InMemoryStore / PostgresStore │
│  - 跨 turn,不跨 thread      │  - 跨 thread,跨 session         │
└──────────────────────────────┴──────────────────────────────────┘

Agent 自主读写:
  - save_memory(content, category)  → store.put
  - search_memory(query, limit)     → store.search (语义召回)

prompt 自动召回:
  - load_relevant_memories() → 每轮取 top-k 相关 memory 拼到 system message
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from llm_factory import build_embeddings, get_api_key
from persistent_store import PersistentInMemoryStore, reset_user


# ============================================================
# 1. 短期记忆: Checkpointer
# ============================================================
def build_checkpointer(sqlite_path: Optional[str] = None):
    """同一 thread_id 内跨 turn 持久化 graph state(messages、tool 调用历史等)。

    sqlite_path=None: MemorySaver,进程内,重启即丢(适合 demo / 单元测试)
    sqlite_path=str : SqliteSaver,落盘,重启可恢复(生产基础款)

    ⚠️ SqliteSaver 不在 langgraph 主包里 —— 需要单独安装:
        pip install langgraph-checkpoint-sqlite

    生产环境分布式部署换 PostgresSaver(同样需要额外安装):
        pip install langgraph-checkpoint-postgres
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string(os.environ["PG_DSN"])
    """
    if sqlite_path is None:
        return MemorySaver()
    # 这一行只有 sqlite_path 非 None 时才执行,所以默认 demo 不需要装该包
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(sqlite_path)


# ============================================================
# 2 + 3. 长期记忆 + 语义召回: Store
# ============================================================
def build_store(
    use_embeddings: bool = True,
    persist_path: Optional[str] = None,
) -> BaseStore:
    """长期记忆,跨 thread / 跨 session 共享。

    use_embeddings=True : 启用向量索引,store.search() 走语义相似度
    use_embeddings=False: 纯 KV,store.search() 只按 namespace 列举(无 OPENAI_API_KEY 时降级)

    persist_path=None : 进程内 InMemoryStore,重启即丢(适合单元测试)
    persist_path=str  : PersistentInMemoryStore,put/delete 后写盘,启动时从 JSON 恢复 ——
                        每个 user_id 的 history 跨进程持久化共享

    生产换 PostgresStore(同样支持 embeddings):

    ⚠️ PostgresStore 不在 langgraph 主包里 —— 需要单独安装:
        pip install langgraph-store-postgres

    然后:
        from langgraph.store.postgres import PostgresStore
        return PostgresStore.from_conn_string(os.environ["PG_DSN"])
    """
    index = None
    if use_embeddings:
        try:
            embeddings, dims = build_embeddings()
            index = {
                "embed": embeddings,
                "dims": dims,
                "fields": ["content"],  # 对 value["content"] 做向量化
            }
        except Exception:
            # 没配 key 时降级到纯 KV
            index = None

    if persist_path:
        return PersistentInMemoryStore(persist_path=persist_path, index=index)
    if index is not None:
        return InMemoryStore(index=index)
    return InMemoryStore()


# ============================================================
# 4. Agent 自主读写的 Tools
# ------------------------------------------------------------
# 关键: 用 InjectedStore + RunnableConfig 解耦
#   - store 由 graph 注入,tool 不需要闭包绑定
#   - user_id 由 config["configurable"]["user_id"] 注入,支持多用户
# ============================================================

@tool
def save_memory(
    content: str,
    category: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """把一条事实写入长期记忆。

    何时调用:
      - 用户透露偏好  → category='preference'  (例: "我喜欢简洁回答")
      - 用户给出事实  → category='fact'        (例: "我在北京工作")
      - 用户立下约定  → category='rule'        (例: "以后都用 Python 示例")
      - 重要历史决定  → category='decision'    (例: "这个项目用 PostgreSQL")

    Args:
        content: 自然语言描述的事实,会被向量化用于后续语义检索
        category: 分类标签
    """
    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("memories", user_id)
    store.put(
        namespace,
        key=str(uuid.uuid4()),
        value={
            "content": content,
            "category": category,
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    return f"[memory saved] {content}"


@tool
def search_memory(
    query: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
    limit: int = 5,
) -> str:
    """从长期记忆按语义相似度检索事实。

    何时调用:
      - 用户问题暗示需要个性化回答
      - 用户引用历史 ("还记得我说过的...")
      - 回答前不确定用户背景
    """
    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("memories", user_id)
    results = store.search(namespace, query=query, limit=limit)
    if not results:
        return "(无相关记忆)"
    return "\n".join(
        f"- [{r.value.get('category', '?')}] {r.value['content']}"
        for r in results
    )


MEMORY_TOOLS = [save_memory, search_memory]


# ============================================================
# 5. prompt 自动召回(与 tool 调用互补)
# ------------------------------------------------------------
# 区别:
#   search_memory tool: LLM 显式调用,可控,但要多一轮 tool round-trip
#   load_relevant_memories: 每轮自动注入 top-k,省一轮但可能注入无关项
# 实践: 两者并用 —— 自动注入兜底,工具调用按需深挖
# ============================================================
def load_relevant_memories(
    store: BaseStore, user_id: str, query: str, k: int = 3
) -> str:
    """返回拼好的 markdown 片段,供 system prompt 拼接。"""
    if not query:
        return ""
    namespace = ("memories", user_id)
    try:
        results = store.search(namespace, query=query, limit=k)
    except Exception:
        # 无 embedding 时降级:列举 namespace 下最近 k 条
        results = store.search(namespace, limit=k)
    if not results:
        return ""
    bullets = "\n".join(f"- {r.value['content']}" for r in results)
    return f"\n已知用户信息(从长期记忆召回):\n{bullets}\n"
