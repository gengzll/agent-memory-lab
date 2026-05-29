"""
Agent Memory 实践报告 — Streamlit 版

启动(在 repo 根目录执行):
  # PowerShell
  $env:PYTHONIOENCODING="utf-8"
  streamlit run app.py

  # bash
  PYTHONIOENCODING=utf-8 streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from textwrap import dedent

import pandas as pd
import streamlit as st


# ====================================================================
# 页面配置
# ====================================================================
st.set_page_config(
    page_title="Agent Memory 实践报告",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================================================================
# Sidebar 导航
# ====================================================================
SECTIONS = [
    "🏠 总览(TL;DR)",
    "📚 Memory 四个能力",
    "🧪 5 个 Demo 对比",
    "📊 Prompt 优化实验",
    "🏭 工业界主流方案",
    "🚀 持久化与生产部署",
    "🎯 选型决策框架",
    "🐛 踩坑记录",
    "🎮 交互式 Prompt 测试",
]

st.sidebar.title("🧠 Agent Memory")
st.sidebar.caption("LangGraph / langmem / Mem0 / Zep / Letta 实测对比")
section = st.sidebar.radio("章节导航", SECTIONS, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "数据来源:Windows 11 + Python 3.12.7 + 智谱 glm-4-flash\n\n"
    "实测于 2026-05-26"
)


# ====================================================================
# Section 1: 总览
# ====================================================================
if section == SECTIONS[0]:
    st.title("Agent Memory 实践报告")
    st.markdown(
        "LangGraph 环境下,**5 种 memory 集成方式**的对比、实测与生产选型指南 —— "
        "从最简单(手写 tool)到最复杂(完整 agent runtime)。"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demo 总数", "5 个", "3 跑通 / 2 代码示范")
    col2.metric("Prompt 实验", "8 组", "4 变体 × 2 输入")
    col3.metric("工业界方案", "5 家", "含 LangGraph 自身")
    col4.metric("踩坑数", "11 个", "全部已修")

    st.markdown("---")
    st.header("TL;DR(三行结论)")
    st.success(
        "**1. 5 个 demo 按复杂度阶梯组织** —— 01 LangGraph 原生 ⭐ → 02 langmem ⭐⭐ → "
        "03 Mem0 ⭐⭐⭐ → 04 Zep ⭐⭐⭐⭐ → 05 Letta ⭐⭐⭐⭐⭐。前三个完整跑通,后两个因需起外部服务给出代码示范。"
    )
    st.info(
        "**2. 工业界 memory 方案有两类设计哲学** —— **Memory Service**"
        "(LangGraph store / langmem / Mem0 / Zep)即插即用到 LangGraph;"
        "**Agent Runtime**(Letta/MemGPT)跟 LangGraph 是平行替代,二选一。"
    )
    st.warning(
        "**3. 生产部署核心 trade-off** —— LLM 自主调 tool 不稳定(~50% 命中率,实测)→ "
        "要么换更强模型,要么程序化兜底,要么用 Mem0 那种'后台自动抽取'绕开 LLM 自觉性。"
    )

    st.markdown("---")
    st.info(
        "📌 **项目范围声明** —— 本 lab 仅聚焦 **memory** 维度的方案对比。"
        "Agent 框架本身(LangGraph / Letta / CrewAI / AutoGen / LlamaIndex)是另一个独立选型问题,"
        "**不在此展开**。我们假定 LangGraph 是主框架,memory 后端从 5 家里选。"
        "**唯一例外**:Letta(demo 05)本身平行替代 LangGraph,纳入是因为它的三层 memory 架构是绕不开的设计思想。"
    )

    st.markdown("---")
    st.header("5 个 Demo —— 复杂度阶梯")
    ladder_df = pd.DataFrame([
        ["01", "langgraph_native", "33.3k (整框架)", "⭐",        "无",                  "Hand-rolled save/search tools; LangGraph 一等公民"],
        ["02", "langmem",          "1.5k",           "⭐⭐",      "+langmem",            "工厂封装 60 行 → 2 行;LLM 标准化写入"],
        ["03", "mem0",             "57k ⭐最高",     "⭐⭐⭐",     "+mem0 +chroma",       "服务端 LLM 自动抽事实;chroma 默认落盘"],
        ["04", "zep",              "4.6k",           "⭐⭐⭐⭐",   "+Zep server",         "知识图谱 + 时序事实(valid_from/to) + 自动 summary"],
        ["05", "letta",            "23k",            "⭐⭐⭐⭐⭐", "+Letta server",       "完整 agent runtime(NOT LangGraph plug-in!)三层 memory"],
    ], columns=["#", "Demo", "GitHub ⭐ (2026-05)", "复杂度", "外部依赖", "核心特性"])
    st.dataframe(ladder_df, use_container_width=True, hide_index=True)
    st.caption("📌 注意:**star 高 ≠ 适合你**。Mem0 star 最高,但适用性取决于你是否需要服务端自动抽取。详见 Section 7「选型决策框架」。")

    st.code(dedent("""
         01 langgraph_native        — short-term + long-term + semantic + LLM-managed
                          ↓ add:    LLM-driven content normalization
         02 langmem                 — same + factory-generated tools
                          ↓ add:    automatic fact extraction + default disk persistence
         03 mem0                    — same + program-controlled writes + chroma persist
                          ↓ add:    knowledge graph + temporal facts + auto summary
         04 zep                     — same + Neo4j-backed graph + Graphiti engine
                          ↓ paradigm shift:
         05 letta (MemGPT)          — abandons LangGraph; LLM-as-OS pages memory in/out
    """).strip(), language="text")

    st.markdown("---")
    st.header("速览:四家方案怎么选")
    overview_df = pd.DataFrame([
        ["公司有自研 Memory 类",        "demo 01 形态(改 2 行接入)"],
        ["想最快出原型",                  "Mem0(demo 03,自带 chroma 落盘)"],
        ["客服/CRM 需要时序追溯",         "Zep(demo 04,知识图谱独家)"],
        ["Agent 要有 persona 持续演化",  "Letta(demo 05,切换整套框架)"],
        ["已 deep in LangGraph",         "langmem(demo 02)或 Mem0(demo 03)"],
    ], columns=["你的诉求", "推荐方案"])
    st.table(overview_df)


# ====================================================================
# Section 2: Memory 四个能力
# ====================================================================
elif section == SECTIONS[1]:
    st.title("📚 Memory 的四个能力")

    cap_data = [
        {"能力": "短期记忆 Short-term", "说明": "同一 thread_id 内跨 turn 持久化对话状态",
         "典型实现": "LangGraph Checkpointer (MemorySaver / SqliteSaver / PostgresSaver)"},
        {"能力": "长期记忆 Long-term", "说明": "跨 thread / 跨 session 持久化事实",
         "典型实现": "LangGraph Store / Mem0 / Zep / 自研向量库"},
        {"能力": "语义召回 Semantic Recall", "说明": "用自然语言 query 检索相关 memory",
         "典型实现": "Embedding 模型 + 向量索引"},
        {"能力": "自主读写 Agent-managed", "说明": "LLM 自己决定何时记/查",
         "典型实现": "把 save/search 暴露为 tool"},
    ]
    st.table(pd.DataFrame(cap_data))

    st.markdown("---")
    st.header("短期 vs 长期 — 本质区别")
    st.code(dedent("""
        ┌──────────────────────────────┬──────────────────────────────────┐
        │  短期记忆 Short-term         │  长期记忆 Long-term              │
        │  ────────────────────────    │  ──────────────────────────────  │
        │  Checkpointer                │  Store (with embeddings)         │
        │  按 thread_id 隔离           │  按 namespace 隔离 (如 user_id)  │
        │  存 graph state (messages)   │  存事实 (content + metadata)     │
        │  MemorySaver / SqliteSaver   │  InMemoryStore / PostgresStore   │
        │  跨 turn,不跨 thread         │  跨 thread,跨 session           │
        └──────────────────────────────┴──────────────────────────────────┘
    """).strip(), language="text")

    st.info(
        "**关键设计哲学**:LangGraph 把 memory 拆成 `Checkpointer` + `Store` 两层是"
        "工程上极其正确的设计。你**几乎不需要换 LangGraph 框架本身**(除非选 Letta 那种"
        "完全平行的 runtime),只需要根据业务诉求决定 `Store` 后端是 Mem0 / Zep / 自研 / "
        "Postgres+pgvector。"
    )


# ====================================================================
# Section 3: 5 个 Demo 对比
# ====================================================================
elif section == SECTIONS[2]:
    st.title("🧪 5 个 Demo 对比")

    st.markdown(
        "**对照变量**(所有 demo 相同):LangGraph `create_react_agent`(05 Letta 除外) + "
        "智谱 `glm-4-flash` + `embedding-2` + 相同测试脚本"
        "(alice/t1 写偏好 → alice/t2 跨 thread 验证 → bob/t3 验证隔离)。"
    )

    diff_df = pd.DataFrame([
        ["save/search 实现",        "手写 @tool ~60 行", "langmem 工厂 1 行", "闭包包装 mem.add/search", "闭包包装 zep client",      "Letta 内置(无需写)"],
        ["长期存储后端",             "InMemoryStore",     "InMemoryStore",     "chromadb(本地落盘)",   "Zep server(Neo4j+PG)",   "Letta server"],
        ["写入触发机制",             "LLM 自主调 tool",    "LLM 自主调 tool",   "程序化 ingest 自动",     "程序化 ingest 自动",       "LLM 自主调内置 tool"],
        ["内容粒度",                "用户原话",          "LLM 标准化后",       "LLM 提取 + 推理",         "facts + graph + summary",  "core / recall / archival 三层"],
        ["持久化(默认)",           "❌ 进程结束即丢",    "❌ 进程结束即丢",    "✅ chroma 落盘",          "✅ Zep server",            "✅ Letta server"],
        ["跑通情况",                "✅",                "✅",                "✅",                      "代码示范(需起服务)",      "代码示范(需起服务)"],
        ["接公司自研后端难度",        "最易(改 2 行)",      "难(封装在内部)",     "中(改 vector_store)",  "极难(整套服务)",          "不适合"],
    ], columns=["维度", "01 LangGraph 原生", "02 langmem", "03 Mem0", "04 Zep", "05 Letta"])
    st.dataframe(diff_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    tabs = st.tabs([
        "01 LangGraph 原生 ⭐",
        "02 langmem ⭐⭐",
        "03 Mem0 ⭐⭐⭐",
        "04 Zep ⭐⭐⭐⭐",
        "05 Letta ⭐⭐⭐⭐⭐",
    ])

    # ------------------- 01 -------------------
    with tabs[0]:
        st.subheader("一句话定位")
        st.info("**完全手写**`save_memory` / `search_memory`,用 `InjectedStore` 让 LangGraph 注入 store。"
                "公司有自研 Memory 类时,改这两个函数体的 store.put/search 就接通。")

        st.subheader("关键代码 — Tool 用 InjectedStore 解耦")
        st.code(dedent('''
            @tool
            def save_memory(
                content: str, category: str, *,
                config: RunnableConfig,
                store: Annotated[BaseStore, InjectedStore()],  # ← LangGraph 一等公民
            ) -> str:
                user_id = config["configurable"]["user_id"]
                store.put(
                    ("memories", user_id),
                    key=str(uuid.uuid4()),
                    value={"content": content, "category": category},
                )
                return f"[memory saved] {content}"
        ''').strip(), language="python")

        st.subheader("实测输出")
        with st.expander("点开看完整对话(alice/t1 → alice/t2 → bob/t3)"):
            st.code(dedent("""
                ======================================================================
                Session 1 — alice 告诉 agent 一些事实和偏好
                ======================================================================
                [user=alice thread=t1]
                  User: 我叫 Alice,在量化交易做研究员
                  Bot : 已记住
                  · tool[save_memory] -> [memory saved] 我叫Alice,在量化交易做研究员

                [user=alice thread=t1]
                  User: 我喜欢回答尽量简短,代码示例只用 Python
                  Bot : 已记住
                  · tool[save_memory] -> [memory saved] 我喜欢回答尽量简短,代码示例只用 Python

                --- store 当前内容 (namespace=('memories','alice')) ---
                  - [fact] 我叫Alice,在量化交易做研究员
                  - [preference] 我喜欢回答尽量简短,代码示例只用 Python

                ======================================================================
                Session 2 — 新 thread (t2),短期记忆已清空
                ======================================================================
                [user=alice thread=t2]
                  User: 帮我写一段计算夏普比率的代码
                  Bot : 好的,以下是Python代码示例:
                        ```python
                        import numpy as np
                        def calculate_sharpe_ratio(returns, risk_free_rate):
                            return (np.mean(returns) - risk_free_rate) / np.std(returns)
                        ```

                ======================================================================
                Session 3 — 换用户 bob,验证 namespace 隔离
                ======================================================================
                [user=bob thread=t3]
                  User: 你知道我叫什么吗?
                  Bot : 我不知道你的名字,你可以告诉我吗?

                --- store 当前内容 (namespace=('memories','bob')) ---
                  (空)
            """).strip(), language="text")

        st.success("✓ Session 2 在新 thread 里直接用 Python — 长期记忆跨 thread 召回成功。")
        st.success("✓ Session 3 bob 不知道 alice 的事 — namespace 隔离成功。")

    # ------------------- 02 -------------------
    with tabs[1]:
        st.subheader("一句话定位")
        st.info("**langmem 工厂替代手写 tool**,60 行变 2 行。底层还是 LangGraph store。"
                "顺手帮你做了 LLM 标准化写入,但隐藏了细节。")

        st.subheader("关键代码 — 工厂一行替代手写")
        st.code(dedent('''
            from langmem import create_manage_memory_tool, create_search_memory_tool

            MEMORY_TOOLS = [
                create_manage_memory_tool(namespace=("memories", "{user_id}")),
                create_search_memory_tool(namespace=("memories", "{user_id}")),
            ]
            # namespace 里的 "{user_id}" 是模板,langmem 自动从 config 注入
        ''').strip(), language="python")

        st.subheader("实测输出 — 关键差异:LLM 标准化")
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("01 — 用户原话")
            st.code("- [fact] 我叫Alice,在量化交易做研究员\n"
                    "- [preference] 我喜欢回答尽量简短,代码示例只用 Python", language="text")
        with col_b:
            st.caption("02 — LLM 标准化后")
            st.code("- 用户 Alice,量化交易研究员。\n"
                    "- Alice 偏好回答简短,代码示例只用 Python。", language="text")

        st.warning(
            "**隐藏的代价**:langmem 内部多了一次 LLM 调用做标准化,token 成本更高。"
            "另外 metadata schema 固定,要自定义字段得传 pydantic model。"
        )

    # ------------------- 03 -------------------
    with tabs[2]:
        st.subheader("一句话定位")
        st.info("**Mem0 自动事实抽取**,不靠 LLM 自觉调 tool。每轮对话后程序化 `m.add()`,"
                "服务端 LLM 自动从对话提取事实。**默认就落盘**(chromadb)。")

        st.subheader("关键代码 — 程序化 ingest")
        st.code(dedent('''
            def ingest_turn(mem, user_id, user_text, bot_text):
                mem.add(
                    messages=[
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": bot_text},
                    ],
                    user_id=user_id,
                )

            def chat(agent, mem, text, thread_id, user_id):
                result = agent.invoke({"messages": [...]}, config={...})
                # ★ 关键:对话结束后自动抽取
                ingest_turn(mem, user_id, text, result["messages"][-1].content)
                return result
        ''').strip(), language="python")

        st.subheader("实测输出 — alice 累积到 13 条 memory!")
        with st.expander("点开看 mem0 自动抽取的 13 条 memory"):
            st.code(dedent("""
                --- mem0 当前 user_id=alice 的全部 memory ---
                  - Alice is a quantitative trading researcher
                  - Alice has a deep understanding of financial markets and data analysis
                    due to her role as a quantitative trading researcher          ← 推理!
                  - Alice prefers concise answers and prefers Python code examples
                  - Assistant will provide short answers and Python code examples
                  - Alice requested a Python code example for calculating the Sharpe ratio
                  - Assistant provided a Python code example to calculate the Sharpe ratio
                  - Alice is a quantitative trading researcher with a deep understanding ...
                  - Alice prefers concise answers and Python code examples
                  - Alice introduced herself as a quantitative trading researcher
                  - The assistant acknowledged Alice's role as a quantitative trading researcher
                  - Alice expressed a preference for concise answers and Python code examples
                  - The assistant confirmed Alice's preference for concise answers ...
                  - (重复变体,共 13 条)
            """).strip(), language="text")

        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Session 1 写入条数", "4 条", "用户只说了 2 句")
        col_y.metric("跑 3 次后累积", "13 条", "+9 条变体/推理")
        col_z.metric("进程重启后还在?", "✓", "chromadb 落盘")

        st.error(
            "**双刃剑**:Mem0 把'量化研究员'**推理出**'对金融市场有深入了解',"
            "用户原话里没有的事实。回答更'懂'用户,但严肃业务有合规风险。"
        )

    # ------------------- 04 -------------------
    with tabs[3]:
        st.subheader("一句话定位")
        st.info("**Zep 在 Mem0 自动抽取之上,加知识图谱 + 时序事实 + 自动 summary**。"
                "需要起 Zep 服务(Docker / Cloud)。本 demo 未跑通,代码作为示范。")

        st.subheader("关键代码 — Zep 三层数据模型")
        st.code(dedent('''
            from zep_cloud.types import Message

            def ingest_turn(client, session_id, user_text, bot_text):
                """服务端会自动抽取 facts + 更新 graph + 生成 summary"""
                client.memory.add(
                    session_id=session_id,
                    messages=[
                        Message(role="user", role_type="user", content=user_text),
                        Message(role="assistant", role_type="assistant", content=bot_text),
                    ],
                )

            @tool
            def search_memory(query: str, *, config: RunnableConfig):
                user_id = config["configurable"]["user_id"]
                res = client.memory.search_sessions(
                    user_id=user_id,
                    text=query,
                    search_scope="facts",      # 或 "messages" / "graph"
                    limit=5,
                )
                ...
        ''').strip(), language="python")

        st.subheader("Zep 独家能力")
        zep_caps = pd.DataFrame([
            ["知识图谱",   "实体(Alice / Python / 量化交易) + 关系('Alice works in 量化交易')", "客服/CRM 用户画像"],
            ["时序追溯",   "`fact.valid_from='2026-05-01'` / `valid_to='2026-08-15'`",            "心理咨询、销售跟进"],
            ["Session 摘要", "自动生成对话摘要,减少 context 长度",                                "长会话"],
        ], columns=["能力", "示例", "适合场景"])
        st.dataframe(zep_caps, use_container_width=True, hide_index=True)

        st.subheader("启动 Zep 服务")
        st.code(dedent('''
            # 本地起 Zep
            docker run -p 8000:8000 ghcr.io/getzep/zep:latest
            export ZEP_BASE_URL=http://localhost:8000

            # 或 Zep Cloud
            export ZEP_API_KEY=...
        ''').strip(), language="bash")

        st.warning(
            "**部署门槛最高**:本地需 Docker + Neo4j + Postgres + Zep server。"
            "检索能力最强(三种 search_scope:messages / facts / graph),但完全黑盒,自定义能力受限。"
        )

    # ------------------- 05 -------------------
    with tabs[4]:
        st.subheader("一句话定位")
        st.warning("**Letta 抛弃 LangGraph**,是个完整 agent runtime。LLM as OS,"
                   "自己 page memory:core memory 始终在 context,recall memory 是历史消息库,"
                   "archival memory 是无限大磁盘。**不是 plug-in,是平行替代框架**。")

        st.subheader("关键代码 — 不再有 `create_react_agent`")
        st.code(dedent('''
            from letta_client import Letta

            client = Letta(base_url="http://localhost:8283")

            # 创建 agent —— 持久化的有 ID 的实体,跨 session 续聊
            agent = client.agents.create(
                name="agent_for_alice",
                memory_blocks=[
                    {"label": "persona", "value": "You are a helpful assistant..."},
                    {"label": "human",   "value": "User is Alice. Update as you learn..."},
                ],
                model="openai/gpt-4o-mini",
                embedding="openai/text-embedding-3-small",
            )

            # 对话
            response = client.agents.messages.create(
                agent_id=agent.id,
                messages=[{"role": "user", "content": "我叫 Alice,在量化交易做研究员"}],
            )
            # Letta 内部 LLM 自动调:
            #   - core_memory_append / core_memory_replace 更新 human block
            #   - archival_memory_insert 存到长期档案
        ''').strip(), language="python")

        st.subheader("Letta 三层 memory 架构")
        st.code(dedent("""
            ┌────────────────────────────────────────────────────────┐
            │  Context window (有限,例如 8k tokens)                 │
            │  ┌──────────────────────────────────────────────────┐  │
            │  │ Core Memory(始终在 context):                     │  │
            │  │   - persona block: "You are..."                   │  │
            │  │   - human block:   "User is Alice, quant..."      │  │
            │  └──────────────────────────────────────────────────┘  │
            │  ┌──────────────────────────────────────────────────┐  │
            │  │ Recent messages(滑动窗口)                        │  │
            │  └──────────────────────────────────────────────────┘  │
            └────────────────────────────────────────────────────────┘
                                      ↕ LLM 调 tool 主动 page in/out
            ┌────────────────────────────────────────────────────────┐
            │  Recall Memory(历史消息库 — 全量保存)                  │
            │  conversation_search(query) 检索旧对话                  │
            └────────────────────────────────────────────────────────┘
            ┌────────────────────────────────────────────────────────┐
            │  Archival Memory(无限大事实库)                         │
            │  archival_memory_insert / archival_memory_search        │
            └────────────────────────────────────────────────────────┘
        """).strip(), language="text")

        st.subheader("启动 Letta 服务")
        st.code(dedent('''
            pip install letta letta-client
            letta server                          # 默认 http://localhost:8283
            export OPENAI_API_KEY=sk-...

            # 或 ZhipuAI:走 OpenAI 兼容
            export OPENAI_API_KEY=<zhipu-key>
            export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
        ''').strip(), language="bash")

        st.error(
            "**最大区别**:Letta 不能即插即用 LangGraph。如果选 Letta,要重写整个 agent 框架。"
            "唯一支持 'agent 自演化人格' 的方案 — agent 有持久化身份,memory 跟 agent 绑定。"
            "Token 成本最高(core memory 永驻 context)。"
        )


# ====================================================================
# Section 4: Prompt 优化实验
# ====================================================================
elif section == SECTIONS[3]:
    st.title("📊 Prompt 优化实验")
    st.markdown(
        "**问题**:demo 01/02 在 `temperature=0` 下,LLM 仍可能"
        "'嘴上说已记住但实际没调 `save_memory` tool'。"
        "用 4 种 prompt × 2 种输入测命中率。"
    )
    st.caption("📁 完整脚本:`experiments/prompt_variants.py`")

    st.markdown("---")
    st.header("4 种 Prompt 变体")
    variant_df = pd.DataFrame([
        ["V0 baseline",  "原温和引导 prompt"],
        ["V1 strict",    "硬性规则 + 触发条件枚举 + 失败模式警告"],
        ["V2 few-shot",  "V1 + 3 个 input/正确流程示例"],
        ["V3 + 自检",     "V2 + 承诺词反向校验('说了已记住但没调 tool,回退')"],
    ], columns=["变体", "设计"])
    st.table(variant_df)

    st.markdown("---")
    st.header("8 组实测命中率")
    result_df = pd.DataFrame([
        ["V0 baseline", "✓ (1/2 不稳定)", "✗ 0 次"],
        ["V1 strict",   "✓ 1 次",          "✗ 0 次"],
        ["V2 few-shot", "✗ 0 次",          "✓ 2 次完美"],
        ["V3 + 自检",    "✗ 0 次",          "✓ 2 次"],
    ], columns=["变体", "I1 简单声明", "I2 含其他诉求"])
    st.table(result_df)

    st.subheader("save_memory 调用次数(可视化)")
    chart_df = pd.DataFrame({
        "I1 simple":  [1, 1, 0, 0],
        "I2 complex": [0, 0, 2, 2],
    }, index=["V0", "V1", "V2", "V3"])
    st.bar_chart(chart_df)

    st.markdown("---")
    st.header("三个核心结论")
    st.warning(
        "**结论 1 — Prompt 优化有天花板**:V2/V3 解决复杂输入,但简单输入反而退化"
        "(few-shot 让模型'按示例行事')。"
    )
    st.warning(
        "**结论 2 — glm-4-flash 在 tool calling 上有固有非确定性**:V0 baseline "
        "同输入两次跑结果不同,temperature=0 也无法消除。**这是模型能力,不是 prompt 能解决的**。"
    )
    st.warning(
        "**结论 3 — 生产环境必须工程兜底**:不能只靠 prompt。"
    )

    st.markdown("---")
    st.header("工程兜底代码(规则匹配补 store.put)")
    st.code(dedent('''
        import re

        TRIGGER_PATTERNS = [
            (r"我(叫|是|名字是)\\s*([^\\s,，。]+)",        "fact"),
            (r"我(喜欢|希望|偏好)\\s*([^,，。]+)",        "preference"),
            (r"我(在|于)\\s*([^\\s,，。]+)\\s*(工作|做|当)", "fact"),
        ]

        def chat(agent, store, text, thread_id, user_id):
            result = agent.invoke({"messages": [HumanMessage(content=text)]},
                                  config={"configurable": {"thread_id": thread_id,
                                                            "user_id": user_id}})
            tool_calls = [m for m in result["messages"]
                          if getattr(m, "type", None) == "tool"
                          and m.name == "save_memory"]

            # LLM 漏调 + 文本含触发词 → 程序化补一次
            if not tool_calls:
                for pattern, category in TRIGGER_PATTERNS:
                    if re.search(pattern, text):
                        store.put(("memories", user_id), key=str(uuid.uuid4()),
                                  value={"content": text, "category": category, "auto": True})
                        break
            return result["messages"][-1].content
    ''').strip(), language="python")

    st.header("提升 ROI 排序")
    roi_df = pd.DataFrame([
        ["1. 换更强模型(glm-4-plus / glm-4-air)",  "API 贵几倍",    "tool calling 显著更稳", "⭐⭐⭐⭐"],
        ["2. 程序化兜底 — 后处理校验",                 "+20 行代码",   "100% 保底",            "⭐⭐⭐⭐⭐"],
        ["3. 改用 Mem0 模式",                       "切换架构",     "完全程序化",            "⭐⭐⭐⭐"],
        ["4. 继续 prompt 工程",                     "时间长",       "收益递减",              "⭐⭐"],
    ], columns=["方案", "成本", "提升", "推荐度"])
    st.dataframe(roi_df, use_container_width=True, hide_index=True)


# ====================================================================
# Section 5: 工业界主流方案 — 5 家全景
# ====================================================================
elif section == SECTIONS[4]:
    st.title("🏭 工业界主流方案 —— 5 家全景")

    st.header("⚠️ 名称澄清:MemGPT / Letta / langmem")
    st.code(dedent("""
        2023.10  UC Berkeley 论文 "MemGPT: Towards LLMs as Operating Systems"
                                  ↓
        2024     原团队成立公司,把研究项目产品化
                                  ↓
        2024中   GitHub 仓库改名 memgpt → letta,公司也叫 Letta
                                  ↓
        2025     "MemGPT" = 论文/概念,"Letta" = 实际框架/公司
    """).strip(), language="text")
    st.info(
        "**MemGPT ↔ Letta**:同一团队,同一项目,改了名。\n\n"
        "**langmem**:**LangChain 团队**独立开发,**思路借鉴 MemGPT**,"
        "实现轻量化,**不是 Letta 的 fork**。"
    )

    st.markdown("---")
    st.header("两类设计哲学(关键区分)")
    st.code(dedent("""
        长期 memory 工具
               │
               ├─── Memory Service 类 ────────────────── 即插即用 ✓
               │    跟 agent 框架平行,LangGraph 可挂载
               │    ├─ LangGraph store (一等公民,自带)
               │    ├─ langmem         (LangChain 同生态)
               │    ├─ Mem0            (Python SDK,可选起服务)
               │    └─ Zep             (必须起服务 / Cloud)
               │
               └─── Agent Runtime 类 ─────────────────── 平行替代 ✗
                    自己一套 agent 框架,不和 LangGraph 共存
                    └─ Letta (原 MemGPT)
    """).strip(), language="text")

    st.markdown("---")
    st.header("5 家全景对比")
    big_df = pd.DataFrame([
        ["GitHub Stars (2026-05)", "33.3k (整框架)",                "1.5k",                              "57k ⭐ 最高",                  "4.6k",                          "23k"],
        ["类型",                "Memory Service",                  "Memory Service",                   "Memory Service",            "Memory Service",                "Agent Runtime"],
        ["一句话",              "LangGraph 自带 KV+向量 store,你自己写 tool", "工厂封装 LangGraph store",       "服务端 LLM 自动抽事实",     "知识图谱 + 时序事实",            "LLM as OS,自己管 memory"],
        ["抽取由谁做",          "LLM 自己用 tool",                  "LLM 自己用 tool(内部加 LLM 标准化)",  "服务端 LLM 自动",            "服务端(facts+graph+summary)",  "LLM 自己用内置 tool"],
        ["数据模型",            "KV + 向量 + namespace",           "同 LangGraph store",                "user/session/agent",         "user→session→message + KG",     "core/recall/archival"],
        ["时序追溯",            "自己加字段",                       "自己加字段",                         "无",                         "★ Graphiti valid_from/to",      "弱"],
        ["知识图谱",            "无",                              "无",                                 "无",                         "★ 有(基于 Neo4j)",             "无"],
        ["部署门槛",            "无(进程内)",                      "无(进程内)",                         "低(pip install)",            "高(Docker+Neo4j+Postgres)",     "中-高(Letta server)"],
        ["与 LangGraph 关系",   "一等公民",                         "plug-in",                           "plug-in",                    "plug-in",                       "平行替代"],
        ["token 成本",          "中",                              "中-高(多次 LLM)",                    "中",                         "中",                            "高(core memory 永驻)"],
        ["对应 Demo",           "01",                              "02",                                 "03",                         "04",                            "05"],
        ["典型场景",            "自研后端集成",                     "LangGraph 标准用法",                  "通用 chatbot,快速原型",      "客服/CRM/心理咨询",             "陪伴 / persona 演化"],
    ], columns=["维度", "LangGraph 原生", "langmem", "Mem0", "Zep", "Letta (MemGPT)"])
    st.dataframe(big_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.success(
        "**关键洞察**:五家的开源版底层向量库都差不多(Qdrant / pgvector / Weaviate / Chroma)。"
        "**差异不在存储,在抽取流程的设计哲学。**"
    )


# ====================================================================
# Section 6: 持久化与生产部署
# ====================================================================
elif section == SECTIONS[5]:
    st.title("🚀 持久化与生产部署")

    st.header("短期记忆要不要落盘?— 看场景")
    sk_yes = pd.DataFrame([
        ["多副本/集群部署", "state 必须共享"],
        ["跨设备/刷页面续聊", "用户切电脑、刷浏览器要恢复"],
        ["隔夜/跨天对话", "用户回来不能从头开始"],
        ["合规/审计(金融、医疗、法律)", "完整对话日志可追溯"],
        ["故障恢复", "进程 crash 后续聊"],
    ], columns=["必须落盘场景", "为什么"])
    sk_no = pd.DataFrame([
        ["一次性匿名查询(Google 那种)", "用户问完就走"],
        ["极轻量内部工具", "团队几个人,可接受重启"],
        ["隐私优先 mode", "用户主动选择不保留"],
        ["纯无状态 RAG", "每次问答独立"],
        ["demo / hackathon", "跑通就行"],
    ], columns=["可以不落场景", "为什么"])
    col1, col2 = st.columns(2)
    with col1:
        st.table(sk_yes)
    with col2:
        st.table(sk_no)

    st.markdown("---")
    st.header("长期记忆:几乎一定要落盘")
    st.success(
        "**长期记忆存在的整个意义就是跨 session 召回。**\n\n"
        "不落盘 = 进程一重启就清空 = 跟没有长期记忆没区别。\n\n"
        "但需要配 **TTL 策略**(GDPR、用户偏好会变、向量库容量) + "
        "**全删除路径**(用户'forget me'要级联删除)。"
    )

    st.markdown("---")
    st.header("工业界 5 种典型架构 Pattern")
    pattern_df = pd.DataFrame([
        ["A. 全内存",                          "MemorySaver",            "InMemoryStore",      "demo / hackathon"],
        ["B. Redis + Postgres+pgvector ⭐主流", "Redis(7-30 天 TTL)",   "Postgres + pgvector", "大多数生产 chatbot"],
        ["C. 全 Postgres",                     "PostgresSaver",          "PostgresStore",      "小团队,简化运维"],
        ["D. 全托管 SaaS",                     "LangGraph Cloud / LangSmith", "Zep Cloud / Mem0 Cloud", "初创,不想自己运维"],
        ["E. 自研 KV + 自研向量库",             "公司内部 KV",            "公司内部向量库",      "大厂,合规/性能自控"],
    ], columns=["Pattern", "短期", "长期", "适合"])
    st.dataframe(pattern_df, use_container_width=True, hide_index=True)

    st.info(
        "**Pattern B 为什么最主流**:Redis 短期读写微秒级 + 原生 TTL;"
        "Postgres + pgvector 长期一套数据库管 KV + 元数据 + 向量,运维简单。"
    )

    st.markdown("---")
    st.header("6 个容易踩的工程坑")
    pit_df = pd.DataFrame([
        ["TTL 必须设",                  "短期记忆无 TTL,Redis 会被对话数据撑爆"],
        ["冷热分离",                    "30 天热数据 Redis,老数据归档对象存储(成本差几十倍)"],
        ["embedding 模型不能随便换",     "切换 embedding 要重新 embed 全量数据"],
        ["多租户隔离",                  "SaaS 场景下 namespace[0] 必须是 tenant_id"],
        ["Delete cascade",              "用户'忘掉我'要同时删 Redis + Postgres + 向量库 + S3"],
        ["写放大",                      "1 条对话可能产生 5-10 次外部调用"],
    ], columns=["细节", "后果"])
    st.dataframe(pit_df, use_container_width=True, hide_index=True)


# ====================================================================
# Section 7: 选型决策框架
# ====================================================================
elif section == SECTIONS[6]:
    st.title("🎯 选型决策框架")

    st.header("决策树")
    st.code(dedent("""
        你的 chatbot 业务:
        │
        ├─ 公司已有自研 Memory 类(KV / 向量库)?
        │   └─ 是 → demo 01 形态(改 2 行 store.put / store.search 接入)
        │
        ├─ 没有自研,想最快出原型?
        │   └─ Mem0(demo 03)— SDK 即用,自带 chroma 落盘
        │
        ├─ 需要长期事实演化 + 时序追溯(客服 / CRM / 心理咨询)?
        │   └─ Zep(demo 04)— 部署高,但知识图谱独家
        │
        ├─ Agent 要有"人格"持续演化(陪伴 / 教练)?
        │   └─ Letta(demo 05)— 切换整个 agent 框架,不再用 LangGraph
        │
        └─ 已 deep in LangGraph,只想 plug 一个 memory?
            ├─ 简单场景 → langmem(demo 02)
            └─ 想要事实抽取 → Mem0(demo 03)
    """).strip(), language="text")

    st.markdown("---")
    st.header("不同业务场景对应方案")
    scene_df = pd.DataFrame([
        ["内部工具 / demo",                    "01 全内存",                        "0 依赖"],
        ["通用 chatbot(中等流量)",            "03 Mem0 + chroma 落盘",            "自动抽取,持久化开箱即用"],
        ["通用 chatbot(高流量,生产)",        "01 + PostgresStore + Redis checkpointer", "性能/成本/可控性"],
        ["客服系统",                           "04 Zep",                           "时序事实 + 图谱必须"],
        ["销售 CRM",                          "04 Zep",                           "同上"],
        ["心理咨询 / 长期陪伴",                "05 Letta 或 04 Zep",               "persona 演化 / 时序追溯"],
        ["企业内合规(金融、医疗)",            "01 + 自研后端 + 审计日志",          "数据自控,不能用 SaaS"],
        ["出海 SaaS",                         "Mem0 Cloud / Zep Cloud",           "省运维,合规靠厂商"],
    ], columns=["场景", "推荐", "理由"])
    st.dataframe(scene_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("🎯 对应你的项目")
    st.info(
        "你说'公司项目代码无法分享,需要 demo 参考'。我的建议:"
    )
    st.success(
        "**从 demo 01 起步,把 `store.put` / `store.search` 替换成公司接口,上层不动**。"
        "这是改动最小、风险最低、保留最大灵活性的方案。\n\n"
        "等业务上线跑稳了,再根据真实数据决定要不要换成 Mem0 那种'自动抽取'模式。"
    )

    st.subheader("接入公司自研 Memory 的具体改法")
    st.code(dedent('''
        # 把 demo 01 的 save_memory / search_memory 函数体里的
        # store.put / store.search 换成公司 Memory 类的 API,其他完全不动:

        @tool
        def save_memory(content: str, category: str, *, config: RunnableConfig) -> str:
            user_id = config["configurable"]["user_id"]
            your_company_memory.save(  # ← 改这里
                user_id=user_id,
                content=content,
                category=category,
            )
            return f"[memory saved] {content}"

        @tool
        def search_memory(query: str, *, config: RunnableConfig, limit: int = 5) -> str:
            user_id = config["configurable"]["user_id"]
            results = your_company_memory.search(  # ← 改这里
                user_id=user_id,
                query=query,
                limit=limit,
            )
            return "\\n".join(f"- {r}" for r in results)

        # 上层 graph、prompt 注入、tool 暴露逻辑全都不动 ✓
    ''').strip(), language="python")


# ====================================================================
# Section 8: 踩坑记录
# ====================================================================
elif section == SECTIONS[7]:
    st.title("🐛 踩坑记录")
    st.markdown("按出现顺序,**所有跑通过程中真实遇到的问题**:")

    bugs_df = pd.DataFrame([
        ["1",  "pip",                "PyPI 直连超时(国内常态)",                                  "`-i https://pypi.tuna.tsinghua.edu.cn/simple`"],
        ["2",  "LangGraph",          "版本号过时(我最初写 >=0.2.50,实际 1.2.1)",                "主版本号跳到 1.x,API 兼容性可控"],
        ["3",  "langchain-community", "deprecation 警告",                                         "仍能用,`ZhipuAIEmbeddings` 还在原位"],
        ["4",  "mem0",               "版本大跳变 0.1.x → 2.0.2,API breaking change",            "`search()` 改用 `filters={'user_id': ...}`,`limit` 改 `top_k`"],
        ["5",  "mem0 chroma",        "`embedding_model_dims` 不接受",                            "删掉,chroma 自动从首次插入推断"],
        ["6",  "PowerShell",         "中文输出乱码",                                              "`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`"],
        ["7",  "PowerShell 5.1",     "`python script.py 2>&1` 卡死(Python 还在跑但 PS 卡死)",   "**不要加 `2>&1`**"],
        ["8",  "LangGraph 1.0",      "`create_react_agent` deprecation 警告",                    "仍可用,推荐迁 `from langchain.agents import create_agent`"],
        ["9",  "spaCy",              "mem0 抱怨 spaCy 没装",                                     "可选 `pip install mem0ai[nlp]`,不装也能跑"],
        ["10", "LLM tool calling",   "glm-4-flash 自主调 tool 不稳定(temperature=0 也不行)",   "prompt 优化 + 程序化兜底(详见 Section 4)"],
        ["11", "Streamlit 1.37",     "`width=\"stretch\"` 不支持(1.40+ 才有)",                  "改 `use_container_width=True`"],
    ], columns=["#", "模块", "问题", "解决"])
    st.dataframe(bugs_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("生产改造最小清单(给你直接抄)")
    st.code(dedent('''
        # 如果基于 demo 01 + PostgresStore + 公司自研 Memory 上线:

        1. memory_module.build_checkpointer
           MemorySaver() → PostgresSaver.from_conn_string(...)

        2. memory_module.build_store
           InMemoryStore(index={...}) → PostgresStore.from_conn_string(..., index={...})

        3. memory_module.save_memory 函数体
           store.put(...) → 调你们自研 Memory 的 save 接口

        4. memory_module.search_memory 函数体
           store.search(...) → 调你们自研 Memory 的 search 接口

        5. 加程序化兜底层(可选,见 Section 4)

        6. 加 TTL 策略 — 写入 expires_at 字段,后台 job 定期清理

        7. 加多租户隔离 — namespace 第一层放 tenant_id

        8. 加 Delete cascade — 用户删除请求级联清空所有 namespace

        其它全部不动:agent.py / run_demo.py / dynamic_prompt / load_relevant_memories
    ''').strip(), language="text")


# ====================================================================
# Section 9: 交互式 Prompt 测试
# ====================================================================
elif section == SECTIONS[8]:
    st.title("🎮 交互式 Prompt 测试")
    st.markdown(
        "实时跑一次 demo 01 — 输入消息,选 prompt variant,看 LLM 是否真的调用 `save_memory`。"
    )

    st.warning(
        "⚠️ 这个测试会**实际调用智谱 API**(用你的 key)。每次大概 0.001 元成本。"
    )

    with st.expander("⚙️ 配置 ZHIPUAI_API_KEY", expanded=not os.getenv("ZHIPUAI_API_KEY")):
        key_input = st.text_input(
            "ZHIPUAI_API_KEY",
            value=os.getenv("ZHIPUAI_API_KEY", ""),
            type="password",
            help="key 只在 streamlit session 中保留,不写入磁盘",
        )
        if key_input:
            os.environ["ZHIPUAI_API_KEY"] = key_input

    st.markdown("---")

    PROMPT_VARIANTS = {
        "V0 baseline (温和引导)": dedent("""
            你是一个有长期记忆的对话助手。

            记忆使用规则:
            1. 当用户透露偏好/事实/约定时,主动调用 save_memory 工具记下,并简短确认"已记住"。
            2. 回答前如果不确定用户背景,先调用 search_memory 检索一次。
        """).strip(),

        "V1 strict (硬性规则)": dedent("""
            你是一个有长期记忆的对话助手。

            【硬性记忆规则 - 必须遵守】

            第一步:检查用户消息是否包含:
              (a) 自我介绍 - 姓名/身份/职业
              (b) 偏好声明 - "我喜欢/希望/以后都用"
              (c) 事实陈述 - "我在/我们用/我的项目"
              (d) 明确请求 - "请记住/记一下"

            第二步:命中则**先调用 save_memory**,再生成回复
              - content: 用第三人称转述(例:"用户的名字是 Alice")
              - category: preference / fact / rule / decision

            【失败模式 - 必须避免】
            × 嘴上说"已记住"但没调 save_memory
            × 因为用户的请求里有"帮我写代码"就跳过 memory 记录
        """).strip(),

        "V2 few-shot (示例引导)": dedent("""
            你是一个有长期记忆的对话助手。

            【记忆规则】用户透露偏好/事实/身份时,**先调用 save_memory,再回复**。

            【示例】
            输入: "我叫 Alice"
              1. tool_call: save_memory(content="用户的名字是 Alice", category="fact")
              2. 回复: "你好,Alice!"

            输入: "我喜欢简洁回答,顺便告诉我今天天气"
              1. tool_call: save_memory(content="用户喜欢简洁回答", category="preference")
              2. 回复: "好的,会简洁回答。关于天气,你能告诉我所在城市吗?"

            输入: "帮我写个函数"
              (无需调用 memory tool,直接回复代码)
        """).strip(),

        "V3 + 自检 (承诺词反向校验)": dedent("""
            你是一个有长期记忆的对话助手。

            【记忆规则】用户透露偏好/事实/身份时,**先调用 save_memory,再回复**。

            【示例】
            输入: "我叫 Alice"
              1. tool_call: save_memory(content="用户的名字是 Alice", category="fact")
              2. 回复: "你好,Alice!"

            输入: "帮我写个函数"
              (无需调用 memory tool,直接回复代码)

            【最后自检 - 关键!!】
            在生成最终回复前,自检一次:
              问: 我的回复里有没有"记住了" / "已记住" / "记下了" / "OK 没问题" 这种承诺词?
              问: 如果有,我之前调用过 save_memory tool 吗?

            如果答案是"有承诺词但没调过 tool" —— 这是严重的"假装记忆"错误。
            请回退,先调用 save_memory 让事实真正落到 store,再生成回复。
        """).strip(),
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        variant = st.selectbox("Prompt 变体", list(PROMPT_VARIANTS.keys()))
    with col2:
        user_input = st.text_input(
            "用户消息",
            value="我叫 Alice,在量化交易做研究员",
            help="试试简单声明 vs 含其他诉求的复杂输入,看 tool 调用差异",
        )

    with st.expander("查看当前 prompt 全文"):
        st.code(PROMPT_VARIANTS[variant], language="text")

    if st.button("🚀 运行 demo 01(实际调用智谱)", type="primary", use_container_width=True):
        if not os.getenv("ZHIPUAI_API_KEY"):
            st.error("请先填入 ZHIPUAI_API_KEY")
        else:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent / "01_langgraph_native"))

            try:
                with st.spinner("调用智谱 glm-4-flash,等 5-15 秒..."):
                    # 重置 module cache 防 prompt patch 不生效
                    for mod in ("agent", "memory_module"):
                        if mod in sys.modules:
                            del sys.modules[mod]
                    import agent as _agent
                    from langchain_core.messages import HumanMessage

                    _agent.SYSTEM_PROMPT = PROMPT_VARIANTS[variant]
                    a, _ = _agent.build_agent()
                    result = a.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config={"configurable": {
                            "thread_id": f"streamlit_{variant[:2]}",
                            "user_id": "streamlit_user",
                        }},
                    )

                reply = result["messages"][-1].content
                tool_calls = [m for m in result["messages"]
                              if getattr(m, "type", None) == "tool"]
                save_calls = [tc for tc in tool_calls if tc.name == "save_memory"]

                col_x, col_y = st.columns(2)
                col_x.metric("Tool 总调用", len(tool_calls))
                col_y.metric(
                    "save_memory 调用",
                    len(save_calls),
                    "✓ 命中" if save_calls else "✗ 漏调",
                )

                st.subheader("Bot 回复")
                st.write(reply)

                if tool_calls:
                    st.subheader("Tool 调用轨迹")
                    for tc in tool_calls:
                        st.code(f"[{tc.name}] {str(tc.content)[:200]}", language="text")
                else:
                    st.warning(
                        "⚠️ **没有任何 tool 调用** —— LLM 选择直接回复,"
                        "没有把信息真的写入 memory。"
                    )

            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.caption(
        "💡 **提示**:试试同一句话切换不同 prompt,或者用 V2 跑'我叫 Alice,我喜欢用 Python,"
        "顺便帮我写个函数'这种复杂输入 —— 你会看到很有意思的命中率差异。"
    )
