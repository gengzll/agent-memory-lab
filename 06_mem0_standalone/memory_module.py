"""
Memory 模块 (mem0 standalone 版) —— 完全不依赖 LangGraph

这个 demo 的核心论点:**mem0 是个独立库,不是 LangGraph 组件**。
对比 03_mem0:
  - 03 把 mem0 硬塞进 LangGraph:make_memory_tools() 把 mem0 包成 @tool,
    再交给 create_react_agent —— 绕了一圈。
  - 06(本 demo)直接用 mem0 原生 API:mem.add() / mem.search(),
    对话由你自己调 LLM endpoint(见 chat.py)。

下面这 4 个函数和 03_mem0 里的**一模一样** —— 因为它们本来就跟框架无关。
这正好证明:你从 03 迁到"纯手撸",这部分代码一行都不用改,
只是把 make_memory_tools() 那段 LangGraph 胶水删掉而已。

数据隔离:mem0 用 user_id 维度隔离,跨进程持久化由 chroma 后端负责。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mem0 import Memory

from llm_factory import get_mem0_embedder_config, get_mem0_llm_config


# 默认落盘目录:本 demo 目录下的 mem0_chroma_standalone(已被 .gitignore 覆盖)
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent / "mem0_chroma_standalone")


def build_memory(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    reset: bool = False,
) -> Memory:
    """mem0 实例 —— LLM/embedder 走 llm_factory,vector store 用本地 chroma。

    注意:这里的 config 跟 03_mem0 的 build_memory() 完全一致 ——
    LLM 和 embedder 都通过 OpenAI 兼容协议接入,换 provider 只改 env var。

    Args:
        persist_dir: chroma 本地落盘目录,跨进程持久化共享。
        reset: True 时先删掉 persist_dir 再重建(**所有用户**的记忆都清空)。
    """
    if reset:
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"[mem0] reset=True,已清空 {persist_dir}")

    config = {
        "llm": get_mem0_llm_config(),
        "embedder": get_mem0_embedder_config(),
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_standalone",
                "path": persist_dir,
            },
        },
    }
    return Memory.from_config(config)


def load_relevant_memories(mem: Memory, user_id: str, query: str, k: int = 3) -> str:
    """语义召回该用户的 top-k 相关记忆,拼成 markdown 片段供 system prompt 使用。

    与 03_mem0 的同名函数完全一致 —— 纯 mem.search,无框架耦合。
    """
    if not query:
        return ""
    try:
        results = mem.search(query=query, filters={"user_id": user_id}, top_k=k)
    except Exception:
        return ""
    items = results.get("results", []) if isinstance(results, dict) else results
    if not items:
        return ""
    bullets = "\n".join(f"- {it.get('memory', it)}" for it in items)
    return f"\n已知用户信息(从长期记忆召回):\n{bullets}\n"


def ingest_turn(mem: Memory, user_id: str, user_text: str, bot_text: str) -> None:
    """对话结束后调一次,让 mem0 用 LLM 自动从这轮对话里抽取事实。

    与 03_mem0 的同名函数完全一致 —— 纯 mem.add,无框架耦合。
    这是 mem0 的核心用法:不靠 LLM 自觉调 save_memory tool,而是程序化每轮入库。
    """
    mem.add(
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ],
        user_id=user_id,
    )


def reset_user_memory(mem: Memory, user_id: str) -> int:
    """清空单个用户的全部 memory,返回删除条数(其他用户不受影响)。"""
    try:
        items = mem.get_all(filters={"user_id": user_id})
        items = items.get("results", []) if isinstance(items, dict) else items
    except Exception:
        return 0
    count = 0
    for it in items:
        mid = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
        if mid:
            try:
                mem.delete(memory_id=mid)
                count += 1
            except Exception:
                pass
    return count
