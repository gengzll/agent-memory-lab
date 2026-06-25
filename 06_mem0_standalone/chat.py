"""
对话引擎 (mem0 standalone 版) —— 直接调 LLM endpoint,无 LangGraph

这是本 demo 的关键文件。看清楚:这里**没有** create_react_agent、没有 graph、
没有 checkpointer、没有 InjectedStore —— 只有一个普通函数 chat(),里面:

    1. mem.search()   读:召回这个用户的相关长期记忆
    2. 拼 system prompt
    3. client.chat.completions.create()  调你自己的 LLM endpoint
    4. mem.add()      写:让 mem0 自动抽取这轮的事实

LLM 客户端用官方 openai SDK。endpoint / model / key 全走 llm_factory 的
环境变量(默认智谱 glm-4-flash),所以 OpenAI / DeepSeek / 任何兼容服务都能用。

⚠️ 关于"短期记忆":
   LangGraph 的 checkpointer 会按 thread_id 自动保存对话历史(短期记忆)。
   脱离框架后,标准库里**没有这个东西** —— 短期记忆要你自己用一个 message list
   管理。chat() 提供可选的 history 参数演示这一点:
     - 传 history → 多轮对话(模型看得到上下文)
     - 不传 history → 单轮无状态(只靠 mem0 的长期记忆)
   本 demo 的跨 session 测试故意不传 history,以证明"召回完全来自 mem0 长期记忆"。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from llm_factory import get_api_key, get_base_url, get_model
from memory_module import ingest_turn, load_relevant_memories


SYSTEM_PROMPT_TEMPLATE = """你是一个有长期记忆的对话助手。
回答时如果下方有相关的用户信息,自然地利用它,不要重复确认。
{memory_block}"""


def build_client() -> OpenAI:
    """官方 openai SDK 客户端,指向 llm_factory 配置的 endpoint。"""
    return OpenAI(api_key=get_api_key(), base_url=get_base_url())


def chat(
    client: OpenAI,
    mem: Any,
    text: str,
    user_id: str,
    *,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """一轮对话:读长期记忆 → 调 endpoint → 写长期记忆。

    Args:
        client: openai SDK 客户端(build_client() 得到)
        mem: mem0 Memory 实例
        text: 用户这轮的输入
        user_id: 长期记忆隔离维度
        history: 可选的短期对话历史(list of {"role","content"});
                 不传则本轮无上下文,纯靠 mem0 长期记忆。
        model: 覆盖 LLM 模型;None 时走 llm_factory 默认。
        verbose: True 时打印召回的记忆 + tool 轨迹,便于教学观察。

    Returns:
        assistant 的回复文本
    """
    # ---------- 1. 读:召回该用户的相关长期记忆 ----------
    memory_block = load_relevant_memories(mem, user_id, text, k=3)
    if verbose:
        print(f"  [召回的长期记忆]{memory_block or ' (无)'}")

    # ---------- 2. 拼 system prompt + 组装 messages ----------
    system = SYSTEM_PROMPT_TEMPLATE.format(
        memory_block=memory_block or "(暂无该用户的已知信息)"
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)  # 短期记忆:你自己管理的对话历史
    messages.append({"role": "user", "content": text})

    # ---------- 3. 直接调 LLM endpoint(这里没有任何框架)----------
    resp = client.chat.completions.create(
        model=model or get_model(),
        temperature=0,
        messages=messages,
    )
    reply = resp.choices[0].message.content or ""

    # ---------- 4. 写:让 mem0 自动从这轮对话抽取事实 ----------
    ingest_turn(mem, user_id, text, reply)

    return reply
