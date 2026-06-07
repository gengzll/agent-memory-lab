"""
统一 LLM / Embedding 工厂 —— 所有 demo 共用,一处配置处处生效。

环境变量(全部可选,有合理默认值):
    OPENAI_API_KEY          主 API key。未设时 fallback 到 ZHIPUAI_API_KEY(保持历史兼容)
    OPENAI_BASE_URL         OpenAI 兼容 endpoint。默认智谱 paas v4
    OPENAI_MODEL            LLM 模型名。默认 glm-4-flash
    OPENAI_EMBEDDING_MODEL  embedding 模型名。默认 embedding-2(智谱 1024 维)

常用 provider 切换 cheatsheet(PowerShell):

    # 智谱 (默认,不需要配 BASE_URL/MODEL)
    $env:OPENAI_API_KEY = "<zhipu-key>"

    # 官方 OpenAI
    $env:OPENAI_API_KEY = "sk-..."
    $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
    $env:OPENAI_MODEL = "gpt-4o-mini"
    $env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    # DeepSeek (LLM 走 DeepSeek,embedding 仍走智谱或 OpenAI —— DeepSeek 暂无 embedding)
    $env:OPENAI_API_KEY = "<deepseek-key>"
    $env:OPENAI_BASE_URL = "https://api.deepseek.com"
    $env:OPENAI_MODEL = "deepseek-chat"
    # embedding 单独配:
    $env:ZHIPUAI_API_KEY = "<zhipu-key>"   # 让 embedding 走智谱

embedding 实现选择规则:
    - OPENAI_BASE_URL 含 "bigmodel.cn" → 用 langchain_community.ZhipuAIEmbeddings
    - 否则                              → 用 langchain_openai.OpenAIEmbeddings
"""

from __future__ import annotations

import os
from typing import Tuple

from langchain_openai import ChatOpenAI


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4-flash"
DEFAULT_EMBEDDING_MODEL = "embedding-2"

EMBEDDING_DIMS = {
    "embedding-2": 1024,
    "embedding-3": 2048,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not key:
        raise RuntimeError(
            "请设置 OPENAI_API_KEY(推荐)或 ZHIPUAI_API_KEY 环境变量。\n"
            "  PowerShell: $env:OPENAI_API_KEY=\"<your-key>\""
        )
    return key


def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def is_zhipu_endpoint(base_url: str | None = None) -> bool:
    return "bigmodel.cn" in (base_url or get_base_url())


def build_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """LangChain ChatOpenAI,所有 demo 走这里。"""
    return ChatOpenAI(
        model=model or get_model(),
        temperature=temperature,
        api_key=get_api_key(),
        base_url=get_base_url(),
    )


def build_embeddings() -> Tuple[object, int]:
    """返回 (embeddings 实例, 维度)。

    维度从已知 model 查表;未知 model 默认 1024,可以通过设 OPENAI_EMBEDDING_DIMS
    显式覆盖。
    """
    model = get_embedding_model()
    dims = int(os.getenv("OPENAI_EMBEDDING_DIMS") or EMBEDDING_DIMS.get(model, 1024))

    if is_zhipu_endpoint():
        # 智谱 embedding 必须走 ZhipuAIEmbeddings,因为它内部要 zhipuai SDK 鉴权
        from langchain_community.embeddings import ZhipuAIEmbeddings

        zhipu_key = os.getenv("ZHIPUAI_API_KEY") or get_api_key()
        return ZhipuAIEmbeddings(model=model, api_key=zhipu_key), dims

    from langchain_openai import OpenAIEmbeddings

    return (
        OpenAIEmbeddings(
            model=model,
            api_key=get_api_key(),
            base_url=get_base_url(),
        ),
        dims,
    )


def get_mem0_llm_config() -> dict:
    """给 mem0 用的 LLM provider 配置块。"""
    return {
        "provider": "openai",
        "config": {
            "model": get_model(),
            "temperature": 0,
            "api_key": get_api_key(),
            "openai_base_url": get_base_url(),
        },
    }


def get_mem0_embedder_config() -> dict:
    """给 mem0 用的 embedder provider 配置块。"""
    model = get_embedding_model()
    dims = int(os.getenv("OPENAI_EMBEDDING_DIMS") or EMBEDDING_DIMS.get(model, 1024))
    # mem0 走 openai 通道,智谱 endpoint 也兼容
    return {
        "provider": "openai",
        "config": {
            "model": model,
            "api_key": get_api_key(),
            "openai_base_url": get_base_url(),
            "embedding_dims": dims,
        },
    }
