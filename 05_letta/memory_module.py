"""
Memory 模块 (Letta 版)

⚠️ 与 demo 01-04 的本质区别 —— Letta 不是 LangGraph 的 plug-in,
   而是一个**完整的 agent runtime**,跟 LangGraph 是平行替代关系。

   - "短期记忆" 由 Letta 自己管(messages in agent state,跨 session 自动续聊)
   - "长期记忆" 由 Letta 三层架构提供:
       * core memory       — 始终在 context 里(persona block + human block)
       * recall memory     — 历史消息库,LLM 可用 conversation_search 查
       * archival memory   — 无限大事实库,LLM 可用 archival_memory_insert/search 调
   - "自主读写" 是 Letta 的核心 — LLM 直接调内置的 core_memory_* / archival_memory_* 工具

前置 —— ⚠️ Vendor status (as of 2026-05):
  Letta 官方主文档以 Letta Cloud(https://app.letta.com)为主推荐。
  本地 server 命令 (`letta server`) 在 pip 包里仍提供,但官方主文档已不强调。

  ✅ 推荐路径 A:Letta Cloud(零本地资源,本 demo 改 env 即可用)
       export LETTA_API_TOKEN=<your-token>     # https://app.letta.com/api-keys
       export LETTA_BASE_URL=https://api.letta.com

  ⚠ 可选路径 B:本地 server(以 docs.letta.com 为准,这里仅供参考)
       1. pip install letta letta-client
       2. letta server                          # 默认 http://localhost:8283
       3. server 配置 LLM provider:
            export OPENAI_API_KEY=sk-...
            # 或 ZhipuAI 兼容:
            export OPENAI_API_KEY=<zhipu-key>
            export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
"""

from __future__ import annotations

import os

from letta_client import Letta


def build_client(base_url: str | None = None) -> Letta:
    """连接到 Letta server。

    base_url=None → 默认本地 http://localhost:8283
    Letta Cloud   → base_url="https://api.letta.com" + token=...
    """
    base_url = base_url or os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    token = os.getenv("LETTA_API_TOKEN")  # 本地 server 不需要 token
    if token:
        return Letta(base_url=base_url, token=token)
    return Letta(base_url=base_url)
