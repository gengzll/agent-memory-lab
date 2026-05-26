"""
Memory 模块 (Zep 版)

与 01/02/03 的核心差异:
  - Zep 是独立服务,demo 跑之前必须先起后端:
    * Zep Cloud:    export ZEP_API_KEY=...
    * Self-hosted:  docker run -p 8000:8000 ghcr.io/getzep/zep:latest
                    export ZEP_BASE_URL=http://localhost:8000
  - 数据模型:  User → Session → Message,zep 强制三层结构
  - 自动能力: 服务端自动抽取事实(facts)+ 实体关系图(knowledge graph)+ 会话摘要(summary)
              比 mem0 更"重",但召回质量(尤其是跨 session 事实)更稳
  - 检索 API: client.memory.search_sessions(text=..., search_scope="facts" | "messages")
  - 短期记忆: zep 自带 session-level 短期上下文,本 demo 仍用 LangGraph checkpointer
              管 graph state,两者互不干扰

数据隔离: zep 用 user_id + session_id 两层,本 demo 用 user_id 做长期隔离。
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 1. 短期记忆 — 仍用 LangGraph
# ============================================================
def build_checkpointer(sqlite_path: Optional[str] = None):
    if sqlite_path is None:
        return MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(sqlite_path)


# ============================================================
# 2. Zep client
# ============================================================
def build_zep_client():
    """两种部署:
       - Zep Cloud:    ZEP_API_KEY
       - Self-hosted:  ZEP_BASE_URL (例如 http://localhost:8000)
    """
    from zep_cloud.client import Zep

    api_key = os.getenv("ZEP_API_KEY")
    base_url = os.getenv("ZEP_BASE_URL")
    if not api_key and not base_url:
        raise RuntimeError(
            "请设置 ZEP_API_KEY (Zep Cloud) 或 ZEP_BASE_URL (self-hosted)。\n"
            "  本地起服务: docker run -p 8000:8000 ghcr.io/getzep/zep:latest"
        )
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return Zep(**kwargs)


# ============================================================
# 3. Zep 数据模型操作 (User / Session / Message)
# ============================================================
def ensure_user(client, user_id: str) -> None:
    try:
        client.user.get(user_id)
    except Exception:
        try:
            client.user.add(user_id=user_id)
        except Exception:
            pass  # 已存在


def ensure_session(client, session_id: str, user_id: str) -> None:
    try:
        client.memory.get_session(session_id)
    except Exception:
        try:
            client.memory.add_session(session_id=session_id, user_id=user_id)
        except Exception:
            pass


def ingest_turn(client, session_id: str, user_text: str, bot_text: str) -> None:
    """对话结束后写入 zep —— 服务端会自动抽取 facts / 更新 graph / 生成 summary。"""
    from zep_cloud.types import Message

    client.memory.add(
        session_id=session_id,
        messages=[
            Message(role="user", role_type="user", content=user_text),
            Message(role="assistant", role_type="assistant", content=bot_text),
        ],
    )


# ============================================================
# 4. Memory Tools (闭包绑定 zep client)
# ============================================================
def make_memory_tools(client):

    @tool
    def save_memory(content: str, *, config: RunnableConfig) -> str:
        """主动写入一条事实。

        注:zep 通常依赖服务端自动抽取事实,这个 tool 用于 LLM 判定"必须立即记住"
        的场景。实现上写为一条特殊 message 让 zep 抽取。
        """
        session_id = config["configurable"]["thread_id"]
        ingest_turn(
            client,
            session_id,
            user_text=f"[memory note] {content}",
            bot_text="(noted)",
        )
        return f"[memory saved] {content}"

    @tool
    def search_memory(query: str, *, config: RunnableConfig, limit: int = 5) -> str:
        """从 zep 按语义相似度检索事实(scope=facts)。"""
        user_id = config["configurable"].get("user_id", "anonymous")
        try:
            res = client.memory.search_sessions(
                user_id=user_id,
                text=query,
                search_scope="facts",
                limit=limit,
            )
        except Exception as e:
            return f"(检索失败: {e})"
        results = getattr(res, "results", None) or []
        if not results:
            return "(无相关记忆)"
        lines = []
        for r in results:
            fact = getattr(r, "fact", None) or getattr(r, "content", None) or str(r)
            lines.append(f"- {fact}")
        return "\n".join(lines)

    return [save_memory, search_memory]


# ============================================================
# 5. prompt 自动召回
# ============================================================
def load_relevant_memories(client, user_id: str, query: str, k: int = 3) -> str:
    if not query:
        return ""
    try:
        res = client.memory.search_sessions(
            user_id=user_id,
            text=query,
            search_scope="facts",
            limit=k,
        )
    except Exception:
        return ""
    results = getattr(res, "results", None) or []
    if not results:
        return ""
    bullets = []
    for r in results:
        fact = getattr(r, "fact", None) or getattr(r, "content", None) or str(r)
        bullets.append(f"- {fact}")
    return f"\n已知用户信息(从长期记忆召回):\n" + "\n".join(bullets) + "\n"
