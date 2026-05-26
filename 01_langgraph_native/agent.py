"""
对话 Agent —— 装配 LangGraph + Memory

形态: create_react_agent(LangGraph 预制 ReAct loop)
  - state: messages
  - 短期记忆: checkpointer 按 thread_id 自动持久化 messages
  - 长期记忆: store 按 user_id namespace 跨 thread 共享
  - 自主读写: tools 由 graph 注入 store,从 config 拿 user_id

调用方式:
    agent.invoke(
        {"messages": [HumanMessage(content="...")]},
        config={"configurable": {
            "thread_id": "...",   # 短期记忆隔离 (一个对话窗口)
            "user_id":   "...",   # 长期记忆隔离 (一个用户)
        }},
    )

换模型:
  - 国内厂商: 把 ChatOpenAI 换成 langchain_community.chat_models.QianfanChatEndpoint
    / ChatTongyi / ChatZhipuAI 等,或者用 OpenAI 兼容协议指向 DeepSeek 等:
        ChatOpenAI(model="deepseek-chat",
                   base_url="https://api.deepseek.com",
                   api_key=os.environ["DEEPSEEK_API_KEY"])
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from memory_module import (
    MEMORY_TOOLS,
    build_checkpointer,
    build_store,
    load_relevant_memories,
)


SYSTEM_PROMPT = """你是一个有长期记忆的对话助手。

记忆使用规则:
1. 当用户透露偏好/事实/约定时,主动调用 save_memory 工具记下,并简短确认"已记住"。
2. 回答前如果不确定用户背景,先调用 search_memory 检索一次。
3. 每轮我会把 top-3 相关记忆自动拼在下方;能直接用就别再多调一次 search_memory。
"""


def build_agent(
    model_name: str = "glm-4-flash",
    sqlite_path: str | None = None,
    use_embeddings: bool = True,
) -> tuple[Any, Any]:
    """装配一个可服务多 user/多 thread 的 agent 实例。

    Returns:
        (agent, store) —— store 暴露出来便于 demo 时直接检视 memory
    """
    import os
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.environ["ZHIPUAI_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    checkpointer = build_checkpointer(sqlite_path=sqlite_path)
    store = build_store(use_embeddings=use_embeddings)

    def dynamic_prompt(state, config):
        """每次 invoke 时动态构造 system message:基础 prompt + top-k 相关 memory。"""
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
