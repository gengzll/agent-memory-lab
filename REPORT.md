# Agent Memory 实践报告

> LangGraph 环境下,**5 种 memory 集成方式**的对比、实测与生产选型指南 —— 从最简单(手写 tool)到最复杂(完整 agent runtime)。

---

## TL;DR(三行)

1. **5 个 demo 按复杂度阶梯组织**:01 LangGraph 原生 ⭐ → 02 langmem ⭐⭐ → 03 Mem0 ⭐⭐⭐ → 04 Zep ⭐⭐⭐⭐ → 05 Letta ⭐⭐⭐⭐⭐。前三个完整跑通,后两个因需起外部服务给出代码示范。
2. **工业界 memory 方案有两类设计哲学**:**Memory Service**(LangGraph store / langmem / Mem0 / Zep)即插即用到 LangGraph;**Agent Runtime**(Letta/MemGPT)跟 LangGraph 是平行替代,二选一。
3. **生产部署核心 trade-off**:LLM 自主调 tool 不稳定(~50% 命中率,实测)→ 要么换更强模型,要么程序化兜底,要么用 Mem0 那种"后台自动抽取"绕开 LLM 自觉性。

---

## 📌 项目范围声明

> **本项目仅聚焦 "memory" 这一维度的方案对比**。Agent 框架本身(LangGraph / Letta / CrewAI / AutoGen / LlamaIndex)是另一个独立的选型问题,本 repo **不展开**。我们假定 **LangGraph 是你的主框架**,memory 后端从 5 家里选。
>
> **唯一例外**:Letta(demo 05)本身是个**完整的 agent runtime**,会替代 LangGraph。我们仍把它纳入,是因为它的"三层 memory"架构(core/recall/archival)是 memory 领域绕不开的设计思想,即便最终不选它,理解它也有价值。
>
> 如果你的主框架不是 LangGraph(用了 LlamaIndex / CrewAI / AutoGen),它们各自自带 memory 子系统,**直接用框架内置**通常比迁移到本 repo 的方案更顺手 — 我们不在这里展开。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [Memory 的四个能力](#2-memory-的四个能力)
3. [5 个 Demo —— 从简到繁](#3-5-个-demo--从简到繁)
4. [实测结果对比](#4-实测结果对比)
5. [Prompt 优化实验](#5-prompt-优化实验)
6. [工业界主流方案对比 —— 5 家全景](#6-工业界主流方案对比--5-家全景)
7. [持久化与生产部署](#7-持久化与生产部署)
8. [选型决策框架](#8-选型决策框架)
9. [踩坑记录](#9-踩坑记录)
10. [附录](#10-附录)

---

## 1. 背景与目标

**业务诉求**:在对话 robot 中接入 memory 模块,让 agent 能记住用户偏好、跨 session 召回事实、跨多用户隔离数据。

**技术约束**:
- 主框架 LangGraph(已选定)
- LLM 用国内厂商(本次实测用智谱 `glm-4-flash`;代码现已抽出 [`llm_factory.py`](llm_factory.py),通过 `OPENAI_BASE_URL` / `OPENAI_MODEL` 即可切到 OpenAI / DeepSeek 等任何兼容 endpoint)
- 公司项目代码无法直接分享 — 需要可参考的 demo

**目标**:
- 验证多种 memory 后端在 LangGraph 下的可行性
- 量化各方案的代码量、行为差异、token 成本、持久化能力
- 给出明确的生产选型建议

---

## 2. Memory 的四个能力

| 能力 | 说明 | 典型实现 |
|---|---|---|
| **短期记忆** Short-term | 同一 `thread_id` 内跨 turn 持久化对话状态(messages、tool 调用历史) | LangGraph `Checkpointer`(MemorySaver / SqliteSaver / PostgresSaver) |
| **长期记忆** Long-term | 跨 thread / 跨 session 持久化"事实"(用户偏好、历史结论) | LangGraph `Store` / Mem0 / Zep / 自研向量库 |
| **语义召回** Semantic Recall | 用自然语言 query 检索相关 memory | Embedding 模型 + 向量索引 |
| **自主读写** Agent-managed | LLM 自己决定何时记、记什么、何时查 | 把 `save_memory` / `search_memory` 暴露为 tool |

### 短期 vs 长期 — 本质区别

```
┌──────────────────────────────┬──────────────────────────────────┐
│  短期记忆 Short-term         │  长期记忆 Long-term              │
│  ────────────────────────    │  ──────────────────────────────  │
│  Checkpointer                │  Store (with embeddings)         │
│  按 thread_id 隔离           │  按 namespace 隔离 (如 user_id)  │
│  存 graph state (messages)   │  存事实 (content + metadata)     │
│  MemorySaver / SqliteSaver   │  InMemoryStore / PostgresStore   │
│  跨 turn,不跨 thread         │  跨 thread,跨 session           │
└──────────────────────────────┴──────────────────────────────────┘
```

> **关键设计哲学**:LangGraph 把 memory 拆成 `Checkpointer` + `Store` 两层是工程上极其正确的设计。你**几乎不需要换 LangGraph 框架本身**,只需要根据业务诉求决定 `Store` 后端是 Mem0 / Zep / 自研 / Postgres+pgvector。Letta 是少数例外(完整替代框架)。

---

## 3. 5 个 Demo —— 从简到繁

### 设计原则:能力逐步累加

```
01 langgraph_native    短期 + 长期 + 语义 + 自主读写(LangGraph 一等公民,手写胶水)
        ↓ + 自动 LLM 标准化写入
02 langmem             同上 + 工厂封装(60 行 → 2 行,LangChain 同生态)
        ↓ + 自动事实抽取(不再依赖 LLM 自觉)+ 默认落盘
03 mem0                同上 + 程序化 ingest + chroma 落盘 + 推理事实
        ↓ + 知识图谱 + 时序事实 + 自动 summary
04 zep                 同上 + Neo4j 图谱 + Graphiti 引擎
        ↓ 范式转变(不是"更多层",是不同框架)
05 letta (MemGPT)      抛弃 LangGraph;LLM as OS,自己 page memory
                       core memory 始终在 context,archival 无限大磁盘
```

### 3.1 Demo 01: LangGraph 原生 ⭐

📁 [`01_langgraph_native/`](01_langgraph_native/)

**一句话**:完全手写 `save_memory` / `search_memory`,用 `InjectedStore` 让 LangGraph 注入 store。公司有自研 Memory 类时,改这两个函数体的 `store.put` / `store.search` 就接通。

**核心代码 — Tool 用 InjectedStore 解耦**:
```python
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
```

**Store 实际写入**:
```
- [fact] 我叫Alice,在量化交易做研究员
- [preference] 我喜欢回答尽量简短,代码示例只用 Python
```

**特点**:
- 用户原话原文存储,带自定义 category 标签
- ❌ 默认不落盘(`InMemoryStore`,生产换 `PostgresStore`)
- ⚠️ 命中率不稳(~50%,LLM 自主调 tool 的固有问题)

---

### 3.2 Demo 02: langmem ⭐⭐

📁 [`02_langmem/`](02_langmem/)

**一句话**:`langmem` 工厂替代手写 tool,60 行变 2 行。底层还是 LangGraph store。顺手帮你做了 LLM 标准化写入,但隐藏了细节。

**核心代码 — 工厂一行替代手写**:
```python
from langmem import create_manage_memory_tool, create_search_memory_tool

MEMORY_TOOLS = [
    create_manage_memory_tool(namespace=("memories", "{user_id}")),
    create_search_memory_tool(namespace=("memories", "{user_id}")),
]
# namespace 里的 "{user_id}" 是模板,langmem 自动从 config 注入
```

**Store 实际写入(对比 01)**:
```
01 原话:     我叫Alice,在量化交易做研究员
02 标准化:   用户 Alice,量化交易研究员。          ← LLM 加工后存
```

**特点**:
- 一个 `manage_memory` tool 支持 create / update / delete 三动作
- ⚠️ 隐藏代价:langmem 内部多一次 LLM 调用做标准化,token 成本更高
- ⚠️ Metadata schema 固定,要自定义字段得传 pydantic model
- ❌ 同样不落盘(默认 InMemoryStore)

---

### 3.3 Demo 03: Mem0 ⭐⭐⭐

📁 [`03_mem0/`](03_mem0/)

**一句话**:不靠 LLM 自觉调 tool,每轮对话后程序化 `m.add()`,服务端 LLM 自动从对话提取事实。**默认就落盘**(chromadb)。

**核心代码 — 程序化 ingest**:
```python
def ingest_turn(mem, user_id, user_text, bot_text):
    """每轮对话后强制调用,mem0 内部 LLM 自动抽取事实"""
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
```

**Store 实际写入 — alice 累积到 13 条(含推理)**:
```
- Alice is a quantitative trading researcher
- Alice has a deep understanding of financial markets and data analysis
  due to her role as a quantitative trading researcher          ← LLM 推理!
- Alice prefers concise answers and prefers Python code examples
- Assistant will provide short answers and Python code examples
- ... (重复变体共 13 条)
```

**特点**:
- ✅ **默认落盘**:chromadb 本地文件 + `~/.mem0/history.db`,实测进程重启后仍能读到
- ✅ 不依赖 LLM 自觉调 tool,命中率 100%
- ⚠️ 抽取偏英文(mem0 内部 prompt)
- ⚠️ 去重不完美(同一事实多次写入产生变体)
- ⚠️ 双刃剑:把"量化研究员"**推理出**"对金融市场有深入了解" — 用户原话没有的事实

---

### 3.4 Demo 04: Zep ⭐⭐⭐⭐

📁 [`04_zep/`](04_zep/)

**一句话**:在 Mem0 的"自动抽取"之上,加 **知识图谱**(实体+关系)+ **时序事实**(`valid_from` / `valid_to`)+ **自动 session summary**。需要起 Zep 服务。

**核心代码 — Zep 三层数据模型**:
```python
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
```

**Zep 独家能力**:
- **知识图谱**:实体(Alice / Python / 量化交易) + 关系("Alice works in 量化交易")— 适合客服/CRM 这种"用户画像"业务
- **时序追溯**:`fact.valid_from = "2026-05-01"` / `valid_to = "2026-08-15"` — 适合心理咨询、销售跟进
- **session summary**:自动生成对话摘要,减少 context 长度

**特点**:
- ✅ 检索能力最强(三种 search_scope:messages / facts / graph)
- ⚠️ 完全黑盒,自定义能力受限
- 💡 我们这版**未跑通**(需起服务),代码作为示范

> ⚠️ **Vendor status (as of 2026-05)**:Zep **Community Edition(self-host docker)已被官方废弃** —— 见 https://github.com/getzep/zep 的 README:"Zep Community Edition is no longer supported and has been deprecated"。
>
> **当前推荐部署路径**:
> - **Zep Cloud(推荐)**:无 Docker,本 demo 现有代码直接可用。注册并申请 key:https://www.getzep.com → `export ZEP_API_KEY=...`
> - **Graphiti self-host(开源替代)**:Zep 把图谱引擎独立开源,可本地 docker compose 跑 Neo4j。仓库:https://github.com/getzep/graphiti。**但 API 与 Zep client 不同**(`graphiti.add_episode` vs `client.memory.add`),要让本 demo 代码跑起来需重写 `memory_module.py`
> - ~~`docker run ghcr.io/getzep/zep:latest`~~ — 历史命令,EOL,不推荐

---

### 3.5 Demo 05: Letta (MemGPT) ⭐⭐⭐⭐⭐

📁 [`05_letta/`](05_letta/)

**一句话**:**抛弃 LangGraph**,Letta 是个完整 agent runtime。LLM as OS,自己 page memory:**core memory** 始终在 context,**recall memory** 是历史消息库,**archival memory** 是无限大磁盘。

**核心代码 — 不再有 `create_react_agent`**:
```python
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
```

**Letta 的三层 memory 架构**:
```
┌────────────────────────────────────────────────────────┐
│  Context window (有限,例如 8k tokens)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Core Memory(始终在 context):                    │  │
│  │   - persona block: "You are..."                   │  │
│  │   - human block:   "User is Alice, quant..."      │  │
│  │   - (可选其他 blocks)                              │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Recent messages(滑动窗口)                       │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
                          ↕ LLM 调 tool 主动 page in/out
┌────────────────────────────────────────────────────────┐
│  Recall Memory(历史消息库 — 全量保存)                │
│  conversation_search(query) 检索旧对话                 │
└────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────┐
│  Archival Memory(无限大事实库)                        │
│  archival_memory_insert / archival_memory_search       │
└────────────────────────────────────────────────────────┘
```

**特点**:
- ❌ **不能即插即用到 LangGraph** — 完全替代框架
- ✅ 唯一支持 "agent 自演化人格" 的方案 — agent 有持久化身份,memory 跟 agent 绑定
- ✅ Context 管理最精细:LLM 自己决定什么 page in,什么放 archival
- ⚠️ Token 成本高(core memory 永驻 context)
- 💡 我们这版**未跑通**,代码作为示范

> ⚠️ **Vendor status (as of 2026-05)**:Letta 官方主文档以 **Letta Cloud**(https://app.letta.com)为主推荐;GitHub README 和 docs 主目录已不强调本地 server 命令。
>
> **当前推荐部署路径**:
> - **Letta Cloud(推荐)**:零本地资源。注册拿 token:https://app.letta.com/api-keys → `export LETTA_API_TOKEN=...` + `export LETTA_BASE_URL=https://api.letta.com`。本 demo 现有代码改这两个 env 即可
> - **本地 server**:`pip install letta && letta server`(默认端口 8283)— pip 包仍提供,但具体启动参数和环境变量**以 https://docs.letta.com 为准**(主文档已不强调,可能在 0.x 版本间有变化)

---

## 4. 实测结果对比

### 4.1 测试脚本(三个 demo 完全相同)

```python
# Session 1 — alice 写入身份+偏好
chat("我叫 Alice,在量化交易做研究员", thread_id="t1", user_id="alice")
chat("我喜欢回答尽量简短,代码示例只用 Python", thread_id="t1", user_id="alice")

# Session 2 — 新 thread,短期记忆清零
chat("帮我写一段计算夏普比率的代码", thread_id="t2", user_id="alice")

# Session 3 — 换用户 bob,验证 namespace 隔离
chat("你知道我叫什么吗?", thread_id="t3", user_id="bob")
```

### 4.2 三个 Demo 横向对比(全部跑通)

| 维度 | 01 手写 | 02 langmem | 03 Mem0 |
|---|---|---|---|
| Session 1 写入条数 | 2 条(用户原话) | 2 条(LLM 标准化) | **4-13 条**(累积 + 推理) |
| Session 2 跨 thread 用 Python | ✓ | ✓ | ✓ |
| Session 3 bob 看不到 alice | ✓ | ✓ | ✓ |
| 是否真有 tool 调用 | 显式 `· tool[save_memory]` | 显式 `· tool[manage_memory]` | **无 tool 行,但 memory 已写入** |
| 进程重启 memory 还在 | ❌ | ❌ | ✅(实测) |

### 4.3 三个核心观察

**观察 1 — Mem0 的"推理事实"是双刃剑**

Mem0 把"量化研究员"自动**推理出**"对金融市场有深入了解"这种用户原话里**没有**的事实。
- 优点:回答更"懂"用户,长期对话越用越聪明
- 风险:推理可能产生不实事实(在严肃业务里有合规问题)

**观察 2 — Mem0 持久化是开箱即用,其他要自己配**

| Demo | 长期记忆后端 | 重启后还在? |
|---|---|---|
| 01 langgraph_native | `InMemoryStore` | ❌ 进程结束即丢 |
| 02 langmem | `InMemoryStore` | ❌ 进程结束即丢 |
| 03 mem0 | chromadb + `~/.mem0/history.db` | ✅ 实测:新进程仍读出 13 条 |

**观察 3 — Mem0 去重不完美**

13 条 alice memory 里至少有 4 对重复变体(向量相似度高但不完全相同)。生产环境需要 reflection / consolidation 策略。

---

## 5. Prompt 优化实验

📁 [`experiments/prompt_variants.py`](experiments/prompt_variants.py)

**问题**:demo 01/02 在 `temperature=0` 下,LLM 仍可能"嘴上说'已记住'但实际没调 `save_memory` tool"。

**实验设计**:4 种 prompt × 2 种输入 × N 次重复,统计 `save_memory` 的实际调用情况。每个变体在上一个基础上累加,看哪一步带来最大提升。

### 5.1 4 种 Prompt 变体 — 累加式设计

| 变体 | 增加了什么 | 关键意图 |
|---|---|---|
| **V0 baseline** | (起点)简短陈述 3 条记忆使用规则 | 看不加任何强化、模型"自由发挥"时的真实命中率 |
| **V1 strict** | 把"何时调 save_memory"展开成**触发条件清单**(a/b/c/d 四类)+ 三步执行流程 + **失败模式警告**("× 嘴上说已记住但没调 tool") | 把规则变硬性指令,堵漏调 |
| **V2 few-shot** | V1 + **4 个 input/正确流程示例**(简单声明 / 含任务的复合声明 / 纯任务无需记 / 多事实复合) | 让模型按示例"临摹"调用形式 |
| **V3 + 自检** | V2 + **承诺词反向校验**:回复前自查"我有没有写'已记住'但其实没调 tool" | 堵"假装记忆"这个最常见的失败模式 |

变体的具体 prompt 文本和"设计意图"字段都在 [`experiments/prompt_variants.py`](experiments/prompt_variants.py) 的 `V0/V1/V2/V3 + V*_DESIGN` 常量里,Streamlit 也能直接看到全文。

### 5.1.5 2 种输入 — 简单 vs 含干扰任务

| 标签 | 完整输入文本 | 含几条待存事实 | 设计目的 |
|---|---|---|---|
| **I1 简单输入** | `我叫 Alice,在量化交易做研究员` | 1 条(姓名 + 职业算一条事实) | 没有任何其他诉求争夺注意力,**理论上应 100% 命中** |
| **I2 复杂输入** | `我叫 Alice,我喜欢用 Python。顺便帮我写一个计算列表平均值的函数。` | 2 条(姓名、编程偏好)+ 1 个**无关编程任务** | 故意混入任务诉求,看 LLM 会不会"被任务带跑"漏调 save_memory。这是生产场景常见模式 |

读数据时记住:**I1 期望 1 次 save_memory 调用,I2 期望 2 次**。

### 5.2 实测命中率(N=5,glm-4-flash,temperature=0)

> **数据收集方式**:每个 (变体 × 输入) 组合重复跑 N=5 次。
> - `mean ± std` — 该组 N 次中 save_memory 实际调用次数的平均值和标准差
> - `hit_rate` — N 次中**至少调用 1 次**的比率(衡量"漏调"严重程度)
> - `perfect_rate` — N 次中**调用次数 = 期望值**的比率(衡量"全调对")

| 变体 | I1 mean ± std | I1 hit / perfect | I2 mean ± std | I2 hit / perfect |
|---|---|---|---|---|
| V0 baseline   | **0.80 ± 0.40** | 80% / 80% | 0.00 ± 0.00 | 0% / 0% |
| V1 strict     | **1.00 ± 0.00** | **100% / 100%** | 0.00 ± 0.00 | 0% / 0% |
| V2 few_shot   | 0.00 ± 0.00 | 0% / 0% | **1.60 ± 0.80** † | 80% / 80% † |
| V3 selfcheck  | 0.00 ± 0.00 | 0% / 0% | **2.00 ± 0.00** | **100% / 100%** |

† V2_few_shot × I2 的 5 次跑里有 1 次智谱 API `Connection error` 被记为 0 次;扣除该次后实际是 `2.00 ± 0.00 / 100% / 100%`。

可视化记忆口诀:
- **V0/V1 阵营** "simple-friendly":简单 input 还行,复杂 input 全军覆没
- **V2/V3 阵营** "complex-master":复杂 input 完美,简单 input 反而退化到 0

### 5.3 三个核心结论

**1. Prompt 优化有反转 trade-off,不是单调改进** ⚠️

累加优化反而带来 simple/complex 之间的能力反转 —— V1→V2 救活了 I2 (0 → 1.6),但同时把 I1 从 1.0 砸到 0;V3 在 I2 上达成完美(2.00 ± 0.00),I1 仍是 0。

机制:V2 加的 4 个 few-shot 示例**全是含任务的复合声明**(`"我喜欢简洁回答,顺便告诉我今天天气"`、`"我叫 Bob,我用 Rust...顺便帮我看下这段代码"`),模型学到的隐含规则成了"**看到任务才存事实**",于是纯自我介绍式的 I1 被判定为"不需要存"。这是 few-shot 的 prompt distribution shift 经典坑。

**2. glm-4-flash 在 tool calling 上有固有非确定性**

直接证据:`V0_baseline × I1_simple` 的 `std=0.40`(5 次跑出 4 次 1 / 1 次 0)。`temperature=0` 不能消除 —— 这是模型 router/采样层面的随机性,不是 prompt 能解决的。`V0_baseline × I2_complex` 全是 0、`V3_selfcheck × I2_complex` 全是 2(std=0.00),说明这种非确定性会被强 prompt **掩盖**(不是消除),但弱 prompt 下立刻暴露。

**3. 生产环境必须工程兜底,prompt 单点不可靠**

没有任何单一 prompt 同时拿下 I1 和 I2。最强的 V3 在 I2 完美,I1 也是 0。生产里你不可能要求用户都按 V3 的训练分布说话 —— 工程兜底(规则匹配补 save / 用 Mem0 那种 LLM 后处理抽取)是必经路。

**额外洞察**:如果你的业务 input 分布**确定是 I2-like 复合诉求**,V3 + glm-4-flash 已能稳定到 100%,可以直接上;如果是 I1-like 纯声明,V1 stricter 是更稳的选择。**没有一个 prompt 能两个都覆盖**就是这次实验最强的工程结论。

### 5.4 工程兜底代码(规则匹配补 store.put)

```python
import re

TRIGGER_PATTERNS = [
    (r"我(叫|是|名字是)\s*([^\s,，。]+)",         "fact"),
    (r"我(喜欢|希望|偏好)\s*([^,，。]+)",         "preference"),
    (r"我(在|于)\s*([^\s,，。]+)\s*(工作|做|当)", "fact"),
]

def chat(agent, store, text, thread_id, user_id):
    result = agent.invoke({"messages": [HumanMessage(content=text)]},
                          config={"configurable": {"thread_id": thread_id,
                                                    "user_id": user_id}})
    tool_calls = [m for m in result["messages"]
                  if getattr(m, "type", None) == "tool" and m.name == "save_memory"]

    # LLM 漏调 + 文本含触发词 → 程序化补一次
    if not tool_calls:
        for pattern, category in TRIGGER_PATTERNS:
            if re.search(pattern, text):
                store.put(("memories", user_id), key=str(uuid.uuid4()),
                          value={"content": text, "category": category, "auto": True})
                break
    return result["messages"][-1].content
```

### 5.5 提升 ROI 排序

| 方案 | 成本 | 提升 | 推荐度 |
|---|---|---|---|
| 1. 换更强模型(`glm-4-plus` / `glm-4-air`) | API 贵几倍 | tool calling 显著更稳 | ⭐⭐⭐⭐ |
| 2. 程序化兜底 — 后处理校验 | +20 行代码 | 100% 保底 | ⭐⭐⭐⭐⭐ |
| 3. 改用 Mem0 模式 | 切换架构 | 完全程序化 | ⭐⭐⭐⭐ |
| 4. 继续 prompt 工程 | 时间长 | 收益递减 | ⭐⭐ |

---

## 6. 工业界主流方案对比 —— 5 家全景

### 6.1 名称澄清:MemGPT / Letta / langmem

```
2023.10  UC Berkeley 论文 "MemGPT: Towards LLMs as Operating Systems"
                          ↓
2024     原团队成立公司,把研究项目产品化
                          ↓
2024中   GitHub 仓库改名 memgpt → letta,公司也叫 Letta
                          ↓
2025     "MemGPT" = 论文/概念,"Letta" = 实际框架/公司
```

- **MemGPT ↔ Letta**:同一团队,同一项目,改了名
- **langmem**:**LangChain 团队**独立开发,**思路借鉴 MemGPT**,实现轻量化,**不是 Letta 的 fork**

### 6.2 两类设计哲学(关键区分)

```
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
```

### 6.3 5 家全景对比

| 维度 | LangGraph 原生 | langmem | Mem0 | Zep | Letta (MemGPT) |
|---|---|---|---|---|---|
| **GitHub Stars** (2026-05) | [33.3k](https://github.com/langchain-ai/langgraph) (整个框架) | [1.5k](https://github.com/langchain-ai/langmem) | [**57k**](https://github.com/mem0ai/mem0) ⭐最高 | [4.6k](https://github.com/getzep/zep) | [23k](https://github.com/letta-ai/letta) |
| **Vendor Status** (2026-05) | 主流活跃 | LangChain 子项目活跃 | 主流活跃 | ⚠️ **CE 废弃,只推 Cloud**(或用 Graphiti) | ⚠️ 主推 Cloud,本地 server 不强调 |
| **类型** | Memory Service | Memory Service | Memory Service | Memory Service | **Agent Runtime** |
| **一句话** | 自带 KV+向量 store,你自己写 tool | 工厂封装 LangGraph store | 服务端 LLM 自动抽事实 | 知识图谱 + 时序事实 | LLM as OS,自己管 memory |
| **抽取由谁做** | LLM 自己用 tool | LLM 自己用 tool(langmem 内部加 LLM 标准化) | 服务端 LLM 每轮自动 | 服务端(facts+graph+summary) | LLM 自己用内置 tool |
| **数据模型** | KV + 向量 + namespace | 同 LangGraph store | user/session/agent | user→session→message + **KG** | core/recall/archival |
| **时序追溯** | 自己加字段 | 自己加字段 | 无 | ★ Graphiti `valid_from/to` | 弱 |
| **知识图谱** | 无 | 无 | 无 | ★ 有(Neo4j) | 无 |
| **部署门槛** | 无(进程内) | 无(进程内) | 低(pip install) | **高**(Docker + Neo4j) | 中-高(Letta server) |
| **与 LangGraph 关系** | 一等公民 | plug-in | plug-in | plug-in | **平行替代** |
| **token 成本** | 中 | 中-高(多次 LLM) | 中 | 中 | 高(core memory 永驻) |
| **对应 demo** | [01](01_langgraph_native/) | [02](02_langmem/) | [03](03_mem0/) | [04](04_zep/) | [05](05_letta/) |
| **典型场景** | 自研后端集成 | LangGraph 标准用法 | 通用 chatbot,快速原型 | 客服/CRM/心理咨询 | 陪伴 / persona 演化 |

### 6.4 关键洞察

**这五家的开源版底层向量库都差不多**(Qdrant / pgvector / Weaviate / Chroma)。差异在**抽取流程的设计哲学**:

| 项 | 抽取流程 |
|---|---|
| LangGraph 原生 | 完全由你控制,LLM 调你写的 tool |
| langmem | LangChain 团队对 LangGraph 路线的工厂化封装,加 LLM 内部标准化 |
| Mem0 | 服务端 LLM 自动,固定流程 |
| Zep | 服务端 LLM 自动 + 知识图谱建模 |
| Letta | LLM 自管(prompt + tool call),OS 化 |

---

## 7. 持久化与生产部署

### 7.1 短期记忆要不要落盘?— 看场景

| 必须落盘 | 可以不落 |
|---|---|
| 多副本/集群部署(state 必须共享) | 一次性匿名查询(Google 那种) |
| 跨设备/刷页面续聊 | 极轻量内部工具 |
| 隔夜/跨天对话 | 隐私优先 mode |
| 合规/审计(金融、医疗、法律) | 纯无状态 RAG |
| 故障恢复 | demo / hackathon |

### 7.2 长期记忆:几乎一定要落盘

> 长期记忆存在的整个意义就是**跨 session 召回**。不落盘 = 进程一重启就清空 = 跟没有长期记忆没区别。

但需要配 **TTL 策略**(GDPR、用户偏好会变、向量库容量) + **全删除路径**(用户"forget me"要级联删除)。

### 7.3 工业界 5 种典型架构 Pattern

| Pattern | 短期 | 长期 | 适合谁 |
|---|---|---|---|
| **A. 全内存** | MemorySaver | InMemoryStore | demo / hackathon |
| **B. Redis + Postgres+pgvector ⭐ 主流** | Redis(7-30 天 TTL) | Postgres + pgvector | 大多数生产 chatbot |
| **C. 全 Postgres** | PostgresSaver | PostgresStore | 小团队,简化运维 |
| **D. 全托管 SaaS** | LangGraph Cloud / LangSmith | Zep Cloud / Mem0 Cloud | 初创,不想自己运维 |
| **E. 自研 KV + 自研向量库** | 公司内部 KV | 公司内部向量库 | 大厂,合规/性能自控 |

### 7.4 6 个容易踩的工程坑

| 细节 | 后果 |
|---|---|
| TTL 必须设 | 短期记忆无 TTL → Redis 撑爆 |
| 冷热分离 | 30 天热数据 Redis,老数据归档对象存储(成本差几十倍) |
| embedding 模型不能随便换 | 换 embedding 要重新 embed 全量 |
| 多租户隔离 | SaaS 场景 `namespace[0]` 必须是 `tenant_id` |
| Delete cascade | 用户"忘掉我"要同时删 Redis + Postgres + 向量库 + S3 |
| 写放大 | 1 条对话可能产生 5-10 次外部调用 |

---

## 8. 选型决策框架

### 8.1 决策树

```
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
    ├─ 简单 → langmem(demo 02,几行接入)
    └─ 自动抽取 → Mem0(demo 03)
```

### 8.2 不同业务场景对应方案

| 场景 | 推荐 | 理由 |
|---|---|---|
| 内部工具 / demo | 01 全内存 | 0 依赖 |
| 通用 chatbot(中等流量) | 03 Mem0 + chroma 落盘 | 自动抽取,持久化开箱即用 |
| 通用 chatbot(高流量,生产) | 01 + PostgresStore + Redis checkpointer | 性能/成本/可控性 |
| 客服系统 | 04 Zep **Cloud**(CE 已废弃)或 Graphiti self-host | 时序事实 + 图谱必须 |
| 销售 CRM | 04 Zep Cloud 或 Graphiti | 同上 |
| 心理咨询 / 长期陪伴 | 05 Letta **Cloud** 或 04 Zep Cloud | persona 演化 / 时序追溯 |
| 企业内合规(金融、医疗) | 01 + 自研后端 + 审计日志 | 数据自控,不能用 SaaS |
| 出海 SaaS | Mem0 Cloud / Zep Cloud | 省运维,合规靠厂商 |

### 8.3 接入公司自研 Memory 的具体改法(基于 demo 01)

```python
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
    return "\n".join(f"- {r}" for r in results)

# 上层 graph、prompt 注入、tool 暴露逻辑全都不动 ✓
```

---

## 9. 踩坑记录

| # | 模块 | 问题 | 解决 |
|---|---|---|---|
| 1 | pip | PyPI 直连超时 | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 2 | LangGraph | 版本号过时(最初写 >=0.2.50,实际 1.2.1) | 主版本号跳到 1.x,核心 import 路径未变 |
| 3 | langchain-community | deprecation 警告 | 仍能用,`ZhipuAIEmbeddings` 还在原位 |
| 4 | mem0 | **版本大跳变 0.1.x → 2.0.2** | `search()` 改用 `filters={'user_id': ...}`,`limit` 改 `top_k` |
| 5 | mem0 chroma | `embedding_model_dims` 不接受 | 删掉,chroma 自动推断 |
| 6 | PowerShell | 中文输出乱码 | `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` |
| 7 | PowerShell 5.1 | `python script.py 2>&1` 卡死 | **不要加 `2>&1`** |
| 8 | LangGraph 1.0 | `create_react_agent` deprecation 警告 | 仍可用,推荐迁移 |
| 9 | spaCy | mem0 抱怨 spaCy 没装 | 可选 `pip install mem0ai[nlp]` |
| 10 | LLM tool calling | glm-4-flash 自主调 tool 不稳定 | prompt 优化 + 程序化兜底(详见 §5) |
| 11 | Streamlit 1.37 | `width="stretch"` 不支持(1.40+ 才有) | 改 `use_container_width=True` |

---

## 10. 附录

### 10.1 文件目录

```
agent-memory-lab/
├── README.md                  ← 项目入口
├── REPORT.md                  ← 本报告
├── app.py                     ← Streamlit 可视化
├── requirements.txt           ← 合并依赖
├── .gitignore
│
├── 01_langgraph_native/       ⭐ 跑通
│   ├── memory_module.py       (短期/长期/语义/自主读写四件套)
│   ├── agent.py
│   ├── run_demo.py
│   └── requirements.txt
│
├── 02_langmem/                ⭐⭐ 跑通
│   ├── memory_module.py       (langmem 工厂替代手写)
│   ├── agent.py
│   ├── run_demo.py
│   └── requirements.txt
│
├── 03_mem0/                   ⭐⭐⭐ 跑通(自动持久化)
│   ├── memory_module.py       (mem0 + chroma + 智谱兼容)
│   ├── agent.py
│   ├── run_demo.py
│   └── requirements.txt
│
├── 04_zep/                    ⭐⭐⭐⭐ 代码示范(需起服务)
│   ├── memory_module.py
│   ├── agent.py
│   ├── run_demo.py
│   └── requirements.txt
│
├── 05_letta/                  ⭐⭐⭐⭐⭐ 代码示范(需起服务,平行替代框架)
│   ├── memory_module.py       (Letta client + 三层 memory 配置)
│   ├── agent.py
│   ├── run_demo.py
│   └── requirements.txt
│
└── experiments/               ← 辅助实验
    ├── prompt_variants.py     ← 4 prompt × 2 input 命中率分析
    └── mem0_persistence.py    ← 跨进程持久化验证
```

### 10.2 依赖版本(2026-05-26 实测)

```
langgraph         1.2.1
langchain-core    1.4.0
langchain-openai  1.2.2
langchain-community 0.4.2
zhipuai           2.1.5.20250825
langmem           0.0.30
mem0ai            2.0.2
chromadb          1.5.9
streamlit         1.37.1
```

### 10.3 复跑命令模板

```powershell
# Windows PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# 智谱(默认 provider,跟报告里实测一致):
$env:OPENAI_API_KEY = "<zhipu-key>"
# 切换 provider(任何 OpenAI 兼容服务都行):
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
# $env:OPENAI_MODEL = "gpt-4o-mini"
# $env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

python "01_langgraph_native\run_demo.py"          # 或 02 / 04
python "03_mem0\run_demo.py" --reset              # mem0 加 --reset 清空旧记忆
```

> 历史兼容:旧的 `ZHIPUAI_API_KEY` 仍能用 —— `llm_factory.get_api_key()` 在 `OPENAI_API_KEY` 未设时自动 fallback。

### 10.4 生产改造最小清单(基于 demo 01)

> ⚠️ **持久化 backend 不在 langgraph 主包里,要单独装**:
> ```bash
> # SQLite checkpointer
> pip install langgraph-checkpoint-sqlite
> # Postgres checkpointer + store(生产推荐)
> pip install langgraph-checkpoint-postgres langgraph-store-postgres
> ```

1. `build_checkpointer`: `MemorySaver()` → `PostgresSaver.from_conn_string(...)`
2. `build_store`: `InMemoryStore(index={...})` → `PostgresStore.from_conn_string(..., index={...})`
3. `save_memory` 函数体: `store.put(...)` → 调你们自研 Memory 的 save 接口
4. `search_memory` 函数体: `store.search(...)` → 调你们自研 Memory 的 search 接口
5. 加程序化兜底层(见 §5.4)
6. 加 TTL 策略 — 写入 `expires_at` 字段,后台 job 定期清理
7. 加多租户隔离 — `namespace[0]` 放 `tenant_id`
8. 加 Delete cascade — 用户删除请求级联清空

其它全部不动:`agent.py` / `run_demo.py` / `dynamic_prompt` / `load_relevant_memories` 这些上层代码原封保留。

---

## 结语

**LangGraph 把 memory 拆成 `Checkpointer` + `Store` 两层是工程上极其正确的设计**。短期内你几乎不需要换 LangGraph 框架本身(除非选 Letta 那种完全平行的 runtime),只需根据业务诉求决定 `Store` 后端是 Mem0 / Zep / 自研 / Postgres+pgvector。

**对你这种"公司自研 Memory 类需要接入"的场景**:从 demo 01 起步,把 `store.put` / `store.search` 替换成公司接口,上层完全不动,是改动最小、风险最低、保留最大灵活性的方案。等业务上线跑稳了,再根据真实数据决定要不要换成 Mem0 那种"自动抽取"模式。

---

*报告生成于 2026-05-26,实测环境 Windows 11 + Python 3.12.7 + 智谱 `glm-4-flash`。*
