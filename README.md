# Agent Memory 实践报告

> LangGraph 环境下,**LangGraph 原生 / langmem / Mem0 / Zep / Letta** 五种 memory 方案的对比、实测与生产选型指南。

---

## TL;DR(三行)

1. **跑通了三种集成方式**:LangGraph 原生手写、langmem 工厂、Mem0 自动抽取。模型用智谱 `glm-4-flash`,embedding 用 `embedding-2`。
2. **工业界 memory 方案分两类**:**Memory Service**(Mem0、Zep、langmem)— 即插即用;**Agent Runtime**(Letta/MemGPT)— 跟 LangGraph 是平行替代关系,二选一。
3. **生产部署的核心 trade-off**:LLM 自主调 tool 不稳定(~50% 命中率,实测数据)→ 要么换更强模型,要么程序化兜底,要么用 Mem0 那种"后台自动抽取"绕开 LLM 自觉性。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [Memory 的四个能力](#2-memory-的四个能力)
3. [三个 demo 的设计与差异](#3-三个-demo-的设计与差异)
4. [实测结果对比](#4-实测结果对比)
5. [Prompt 优化实验:让 LLM 稳定调用 memory tool](#5-prompt-优化实验)
6. [工业界主流方案对比](#6-工业界主流方案对比)
7. [持久化与生产部署](#7-持久化与生产部署)
8. [选型决策框架](#8-选型决策框架)
9. [踩坑记录](#9-踩坑记录)
10. [附录:核心代码与文件结构](#10-附录)

---

## 1. 背景与目标

**业务诉求**:在对话 robot 中接入 memory 模块,让 agent 能记住用户偏好、跨 session 召回事实、跨多用户隔离数据。

**技术约束**:
- 主框架 LangGraph(已选定)
- LLM 用国内厂商(本次用智谱 `glm-4-flash`)
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

**短期 vs 长期的本质区别**:

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

---

## 3. 三个 demo 的设计与差异

### 3.1 共同点(对照变量)

| 项 | 取值 |
|---|---|
| Agent 框架 | LangGraph `create_react_agent` |
| LLM | 智谱 `glm-4-flash`(走 OpenAI 兼容协议) |
| Embedding | 智谱 `embedding-2`(1024 维) |
| 短期记忆 | `MemorySaver`(内存) |
| 测试脚本 | alice/t1 写偏好 → alice/t2 跨 thread 验证 → bob/t3 验证隔离 |

### 3.2 差异(实验变量)

| 维度 | 01 LangGraph 原生 | 02 langmem | 03 Mem0 |
|---|---|---|---|
| `save_memory` / `search_memory` 实现 | 手写 `@tool`,~60 行 | langmem 工厂,1 行 | 闭包包装 `mem.add` / `mem.search` |
| 长期 store 后端 | `InMemoryStore` | `InMemoryStore`(langmem 直接基于 LangGraph store) | `chromadb`(本地落盘) |
| 写入触发机制 | **LLM 自主调 tool** | **LLM 自主调 tool** | **程序化** `ingest_turn()` 每轮自动调 |
| 内容粒度 | 用户原话 | LLM 内部标准化后 | LLM 提取 + **推理** |
| 持久化(默认) | ❌ 进程结束即丢 | ❌ 进程结束即丢 | ✅ chroma 自动落盘 |
| 接公司自研后端的难度 | **最易**(改 2 行 `store.put` / `store.search`) | 难(封装在 langmem 内部) | 改 vector_store 配置 |

### 3.3 核心代码骨架对比

**demo 01 — 手写 tool**:

```python
@tool
def save_memory(
    content: str, category: str, *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],  # ← LangGraph 一等公民
) -> str:
    user_id = config["configurable"]["user_id"]
    store.put(("memories", user_id), key=str(uuid.uuid4()),
              value={"content": content, "category": category})
    return f"[memory saved] {content}"
```

**demo 02 — langmem 工厂**:

```python
MEMORY_TOOLS = [
    create_manage_memory_tool(namespace=("memories", "{user_id}")),
    create_search_memory_tool(namespace=("memories", "{user_id}")),
]
# 60 行手写 → 2 行工厂调用,语义保留
```

**demo 03 — Mem0 闭包**:

```python
def make_memory_tools(mem: Memory):
    @tool
    def search_memory(query, *, config, limit=5):
        user_id = config["configurable"]["user_id"]
        results = mem.search(query=query,
                             filters={"user_id": user_id}, top_k=limit)
        ...

# 关键:不靠 LLM 自觉,每轮对话后程序化 ingest
def ingest_turn(mem, user_id, user_text, bot_text):
    mem.add(messages=[{"role": "user", "content": user_text},
                      {"role": "assistant", "content": bot_text}],
            user_id=user_id)
```

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

### 4.2 三个 demo 的 Store 实际内容对比

**demo 01 — 用户原话 + 你自定义的 category**:
```
- [fact] 我叫Alice,在量化交易做研究员
- [preference] 我喜欢回答尽量简短,代码示例只用 Python
```

**demo 02 — LLM 标准化后的中文事实**:
```
- 用户 Alice,量化交易研究员。
- Alice 偏好回答简短,代码示例只用 Python。
```

**demo 03 — LLM 提取 + 推理出的英文事实(累积 13 条)**:
```
- Alice is a quantitative trading researcher
- Alice has a deep understanding of financial markets and data analysis
  due to her role as a quantitative trading researcher          ← 推理!
- Alice prefers concise answers and prefers Python code examples
- Assistant will provide short answers and Python code examples
- Alice requested a Python code example for calculating the Sharpe ratio
- Assistant provided a Python code example to calculate the Sharpe ratio
- Alice is a quantitative trading researcher with a deep understanding
- Alice prefers concise answers and Python code examples
- ... (重复变体,共 13 条)
```

### 4.3 三个 demo 横向对比

| 维度 | 01 手写 | 02 langmem | 03 Mem0 |
|---|---|---|---|
| Session 1 写入条数(实际) | 2 条(用户原话) | 2 条(LLM 标准化) | **4-13 条**(累积 + 推理) |
| Session 2 跨 thread 用 Python 回答 | ✓ | ✓ | ✓ |
| Session 3 bob 看不到 alice | ✓ | ✓ | ✓ |
| tool 调用是否真发生 | 显式 `· tool[save_memory] -> ...` 行 | 显式 `· tool[manage_memory] -> ...` 行 | **无 tool 行,但 memory 已写入** |
| 进程重启后 memory 是否还在 | ❌ | ❌ | ✅(实证:新进程 `m.get_all` 仍能读到) |

### 4.4 三个核心观察

**观察 1 — Mem0 的"推理事实"是双刃剑**

Mem0 把"量化研究员"自动**推理出**"对金融市场有深入了解"这种**用户原话里没有的事实**。
- 优点:回答更"懂"用户,长期对话越用越聪明
- 风险:推理可能产生不实事实(在严肃业务里有合规问题)

**观察 2 — Mem0 持久化是"开箱即用",其他要自己配**

| Demo | 长期记忆后端 | 重启后还在? |
|---|---|---|
| 01 langgraph_native | `InMemoryStore` | ❌ 进程结束即丢 |
| 02 langmem | `InMemoryStore` | ❌ 进程结束即丢 |
| 03 mem0 | chromadb + `~/.mem0/history.db` | ✅ 实证:新进程仍读出 13 条 |

**观察 3 — Mem0 去重不完美**

同一事实经过多次 LLM 提取产生表述略不同的变体(向量相似度高但不完全相同),被当作多条留下。
13 条 alice memory 里至少有 4 对重复变体。

---

## 5. Prompt 优化实验

**问题**:demo 01/02 在 `temperature=0` 下,LLM 仍可能"嘴上说'已记住',但实际没调 `save_memory` tool"。

**实验设计**:4 种 prompt × 2 种输入,看 `save_memory` 是否真被调用。

### 5.1 Prompt 变体

| 变体 | 设计 |
|---|---|
| V0 baseline | 原温和引导 prompt |
| V1 strict | 硬性规则 + 触发条件枚举 + 失败模式警告 |
| V2 few-shot | V1 + 3 个 input/正确流程 示例 |
| V3 + 自检 | V2 + 承诺词反向校验("如果想说'记住了'但没调 tool,回退") |

### 5.2 实测结果

| 变体 | I1 `我叫 Alice,在量化交易做研究员` | I2 `我叫 Alice,我喜欢用 Python。顺便帮我写函数` | 命中率 |
|---|---|---|---|
| V0 baseline | 跑两次:1次✓,1次✗(**不稳定**) | ✗ 0 次调用 | ~50% |
| V1 strict | ✓ 1 次 | ✗ 0 次 | 1/2 |
| V2 few-shot | ✗ 0 次 | ✓ **2 次**(完美) | 1/2 |
| V3 + 自检 | ✗ 0 次 | ✓ 2 次 | 1/2 |

### 5.3 结论

1. **Prompt 优化有效但有天花板** — V2/V3 解决了复杂输入,但简单输入反而退化(few-shot 让模型"按示例行事")
2. **glm-4-flash 在 tool calling 上有固有非确定性** — temperature=0 也无法消除,**这是模型能力问题,不是 prompt 能解决的**
3. **生产环境必须工程兜底**

### 5.4 工程兜底代码

```python
import re

TRIGGER_PATTERNS = [
    (r"我(叫|是|名字是)\s*([^\s,，。]+)", "fact"),
    (r"我(喜欢|希望|偏好)\s*([^,，。]+)", "preference"),
    (r"我(在|于)\s*([^\s,，。]+)\s*(工作|做|当)", "fact"),
]

def chat(agent, store, text, thread_id, user_id):
    result = agent.invoke({"messages": [HumanMessage(content=text)]},
                          config={"configurable": {"thread_id": thread_id,
                                                    "user_id": user_id}})
    # 检查 LLM 实际是否调了 save_memory
    tool_calls = [m for m in result["messages"]
                  if getattr(m, "type", None) == "tool" and m.name == "save_memory"]

    # 没调 + 用户文本含触发词 → 程序化补一次
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
| 2. 程序化兜底 — 后处理校验 | 加 20 行代码 | 100% 保底 | ⭐⭐⭐⭐⭐ |
| 3. 改用 Mem0 模式 | 切换架构 | LLM 不参与抽取,完全程序化 | ⭐⭐⭐⭐ |
| 4. 继续 prompt 工程 | 时间长 | 收益递减,有天花板 | ⭐⭐ |

---

## 6. 工业界主流方案对比

### 6.1 两类设计哲学(关键区分)

```
长期 memory 工具
       │
       ├─── Memory Service 类 ────────────────── 即插即用 ✓
       │    跟 agent 框架平行,LangGraph 可挂载
       │    ├─ Mem0     (Python SDK,可选起服务)
       │    ├─ Zep      (必须起服务 / Cloud)
       │    └─ langmem  (LangChain 同生态)
       │
       └─── Agent Runtime 类 ─────────────────── 平行替代 ✗
            自己一套 agent 框架,不和 LangGraph 共存
            └─ Letta (原 MemGPT)
```

### 6.2 名称澄清:MemGPT / Letta / langmem

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
- **langmem**:**LangChain 团队**独立开发,**思路借鉴 MemGPT**(LLM 自管 memory),实现轻量化,**不是 Letta 的 fork**

### 6.3 四家全景对比

| 维度 | Letta (MemGPT) | Mem0 | Zep | langmem |
|---|---|---|---|---|
| **一句话** | LLM as OS,自己管 memory | Memory as Service,后台抽 | 知识图谱 + 时序事实库 | LangChain 出的 memory 工具包 |
| **是 Agent Runtime?** | ✅ | ❌ | ❌ | ❌ |
| **抽取由谁做** | LLM 自己用 tool | 服务端 LLM 自动 | 服务端(facts + graph + summary) | LLM 自己用 tool(工厂封装好) |
| **数据模型** | core / recall / archival | user / session / agent | user → session → message + **knowledge graph** | LangGraph store(KV + 向量) |
| **时序追溯** | 弱 | 无 | ★ Graphiti 提供 `valid_from / valid_to` | 无 |
| **知识图谱** | 无 | 无 | ★ 有(基于 Neo4j) | 无 |
| **部署门槛** | 中(起 Letta server) | 低(`pip install`) | **高**(Docker + Neo4j + Postgres) | 极低 |
| **与 LangGraph 关系** | 平行替代 | plug-in | plug-in | plug-in(同生态) |
| **token 成本** | 高(core memory 永驻 context) | 中 | 中 | 中 |
| **典型场景** | 陪伴 / persona 演化 | 通用 chatbot,快速原型 | 客服 / CRM / 心理咨询 | LangGraph 标准用法 |

### 6.4 关键洞察

**这四家的开源版底层向量库都差不多**(Qdrant / pgvector / Weaviate / Chroma)。差异不在存储,在**抽取流程的设计哲学**:

| 项 | 抽取流程 |
|---|---|
| Letta | 完全交给 LLM(prompt + tool call) |
| Mem0 | 交给 mem0 服务端 LLM,固定流程 |
| Zep | 交给 zep 服务端 LLM,加一层图谱建模 |
| langmem | LangChain 团队对 Letta 路线的轻量化封装 |

**这也对应我们 demo 的设计**:

| Demo | 对应业界产品 |
|---|---|
| demo 01 手写 tool | ≈ Letta 简化版(去掉 server) |
| demo 02 langmem | = langmem 本身 |
| demo 03 Mem0 | = Mem0 本身 |
| demo 04 Zep(未跑) | = Zep 本身 |

---

## 7. 持久化与生产部署

### 7.1 短期记忆要不要落盘?— 看场景

| 必须落盘 | 可以不落 |
|---|---|
| 多副本/集群部署(state 必须共享) | 一次性匿名查询(像 Google 那种) |
| 用户跨设备/刷页面续聊 | 极轻量内部工具 |
| 隔夜/跨天对话 | 隐私优先模式 |
| 合规/审计(金融、医疗、法律) | 纯无状态 RAG |
| 故障恢复 | demo / hackathon |

### 7.2 长期记忆几乎一定要落盘

> 长期记忆存在的整个意义就是**跨 session 召回**。不落盘 = 进程一重启就清空 = 跟没有长期记忆没区别。

但需要配**TTL 策略**(GDPR、用户偏好会变、向量库容量) + **全删除路径**(用户"forget me"要级联删除)。

### 7.3 工业界典型架构

| Pattern | 短期 | 长期 | 适合谁 |
|---|---|---|---|
| **A. 全内存** | MemorySaver | InMemoryStore | demo / hackathon |
| **B. Redis + Postgres+pgvector** ⭐ **主流** | Redis(7-30 天 TTL) | Postgres + pgvector | 大多数生产 chatbot |
| **C. 全 Postgres** | PostgresSaver | PostgresStore | 小团队,不想多维护一套基础设施 |
| **D. 全托管 SaaS** | LangGraph Cloud / LangSmith | Zep Cloud / Mem0 Cloud | 初创,不想自己运维 |
| **E. 自研 KV + 自研向量库** | 公司内部 KV | 公司内部向量库 | 大厂,合规/性能/成本自控 |

### 7.4 Pattern B 为什么最主流

- **Redis 短期**:读写微秒级、原生 TTL、集群成熟 — 对话状态需要的就是低延迟随机访问
- **Postgres + pgvector 长期**:事务保证、SQL 灵活查询、`pgvector` 扩展提供向量索引 — 一套数据库管 KV + 元数据 + 向量,运维简单

### 7.5 容易踩的工程坑

| 细节 | 后果 |
|---|---|
| TTL 必须设 | 短期记忆若无 TTL,Redis 会被对话数据撑爆 |
| 冷热分离 | 30 天热数据 Redis,老数据归档对象存储(成本差几十倍) |
| embedding 模型不能随便换 | 长期记忆落盘后,换 embedding 要重新 embed 全量数据 |
| 多租户隔离 | SaaS 场景下 `namespace[0]` 必须是 `tenant_id`,否则数据串户是事故 |
| Delete cascade | 用户要求"忘掉我",必须同时删 Redis + Postgres + 向量库 + S3 归档 |
| 写放大 | 长期记忆 LLM 提取 + embedding,1 条对话可能产生 5-10 次外部调用 |

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
│   └─ Mem0(SDK 即用,自带 chroma 落盘,但接受英文事实 + 推理)
│
├─ 需要长期事实演化 + 时序追溯(客服 / CRM / 心理咨询)?
│   └─ Zep(部署门槛高,但知识图谱独家)
│
├─ Agent 要有"人格"持续演化(陪伴 / 教练)?
│   └─ Letta(切换整个 agent 框架,不再用 LangGraph)
│
└─ 已 deep in LangGraph,只想 plug 一个 memory?
    ├─ 简单场景 → langmem(demo 02,几行接入)
    └─ 想要事实抽取 → Mem0(demo 03)
```

### 8.2 不同业务场景对应方案

| 场景 | 推荐 | 理由 |
|---|---|---|
| 内部工具 / demo | demo 01 全内存 | 0 依赖,跑通即可 |
| 通用 chatbot(中等流量) | demo 03 Mem0 + chroma 落盘 | 自动抽取,持久化开箱即用 |
| 通用 chatbot(高流量,生产) | demo 01 + PostgresStore + Redis checkpointer | 性能/成本/可控性 |
| 客服系统 | Zep(本地 Docker 或 Cloud) | 时序事实 + 图谱必须 |
| 销售 CRM | Zep | 同上 |
| 心理咨询 / 长期陪伴 | Letta 或 Zep | persona 演化 / 时序追溯 |
| 企业内合规要求(金融、医疗) | demo 01 + 自研后端 + 完整审计日志 | 数据自控,不能用 SaaS |
| 出海 SaaS | Mem0 Cloud / Zep Cloud | 省运维,数据合规靠厂商 |

### 8.3 对应你的项目

> 你说"公司项目代码无法分享,需要一个 demo 参考"。我的判断:

| 你的诉求 | 推荐 |
|---|---|
| 公司有自研 Memory 类,只要"LangGraph 怎么接" | **demo 01 的形态** |
| 后续要落盘 | **demo 01 + PostgresStore**(代码我可以现成给) |
| 想试"自动抽取"是否好用 | **demo 03 mem0** 直接 ingest |

接入公司自研 Memory 的具体改法:

```python
# 把 demo 01 的 save_memory / search_memory 函数体里的 store.put / store.search
# 换成你们公司 Memory 类的 API,其他完全不动:

@tool
def save_memory(content: str, category: str, *, config: RunnableConfig) -> str:
    user_id = config["configurable"]["user_id"]
    your_company_memory.save(user_id=user_id, content=content, category=category)  # ← 改这里
    return f"[memory saved] {content}"

@tool
def search_memory(query: str, *, config: RunnableConfig, limit: int = 5) -> str:
    user_id = config["configurable"]["user_id"]
    results = your_company_memory.search(user_id=user_id, query=query, limit=limit)  # ← 改这里
    return "\n".join(f"- {r}" for r in results)
```

上层 graph、prompt 注入、tool 暴露逻辑全都不动。

---

## 9. 踩坑记录

按出现顺序,所有跑通过程中真实遇到的问题:

| # | 模块 | 问题 | 解决 |
|---|---|---|---|
| 1 | pip | PyPI 直连超时(国内常态) | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 2 | LangGraph | 版本号过时(我最初写 `>=0.2.50`,实际 1.2.1) | 主版本号已跳到 1.x,API 兼容性可控,核心 import 路径未变 |
| 3 | langchain-community | deprecation 警告 | 仍能用,`ZhipuAIEmbeddings` 还在原位 |
| 4 | mem0 | **版本大跳变** 0.1.x → 2.0.2,API 有 breaking change | `search()`/`get_all()` 不再接受 `user_id` 参数,改用 `filters={"user_id": ...}`,`limit` 改 `top_k` |
| 5 | mem0 chroma config | `embedding_model_dims` 不接受 | 删掉,chroma 自动从首次插入推断维度 |
| 6 | PowerShell | 中文输出乱码 | `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` + `$env:PYTHONIOENCODING="utf-8"` |
| 7 | PowerShell 5.1 | `python script.py 2>&1` 把 stderr 行包装成 NativeCommandError,导致 Python 进程虽然在跑但 PowerShell 这边卡死 | **不要加 `2>&1`** |
| 8 | LangGraph 1.0 | `create_react_agent` deprecation 警告 | 仍可用,推荐迁移到 `from langchain.agents import create_agent` |
| 9 | spaCy | mem0 抱怨 spaCy 没装 | 可选 `pip install mem0ai[nlp]`,不装也能跑(走 fallback) |
| 10 | LLM tool calling | glm-4-flash 自主调 tool 不稳定(temperature=0 也不行) | prompt 优化(实验 5)+ 程序化兜底 |

---

## 10. 附录

### 10.1 文件目录

```
D:\work\Memory\
├── REPORT.md                          ← 本报告
├── requirements.txt                   ← 全部依赖合并清单
│
├── 01_langgraph_native\               ✅ 跑通
│   ├── memory_module.py               (短期/长期/语义/自主读写四件套)
│   ├── agent.py                       (create_react_agent + ZhipuAI 配置)
│   ├── run_demo.py                    (alice/t1+t2,bob/t3 三段验证)
│   ├── quick_test_output_variability.py  (4 种 prompt × 2 种输入对比)
│   └── requirements.txt
│
├── 02_langmem\                        ✅ 跑通
│   ├── memory_module.py               (langmem 工厂替代手写 tool)
│   ├── agent.py                       (与 01 几乎相同)
│   ├── run_demo.py                    (与 01 相同的测试)
│   └── requirements.txt
│
├── 03_mem0\                           ✅ 跑通(自动持久化)
│   ├── memory_module.py               (mem0 + chroma + 智谱兼容协议)
│   ├── agent.py                       (用 ingest_turn 程序化写入)
│   ├── run_demo.py                    (chat() 后自动 ingest)
│   ├── check_persistence.py           (新进程验证落盘)
│   └── requirements.txt
│
└── 04_zep\                            未跑(需要起 zep 服务)
    ├── memory_module.py               (zep_cloud client + 三层数据模型)
    ├── agent.py
    ├── run_demo.py
    └── requirements.txt
```

### 10.2 依赖版本(截至 2026-05-25 实测)

```
langgraph         1.2.1
langchain-core    1.4.0
langchain-openai  1.2.2
langchain-community 0.4.2  (deprecated 但还能用)
zhipuai           2.1.5.20250825
langmem           0.0.30
mem0ai            2.0.2
chromadb          1.5.9
```

### 10.3 关键 API 索引

| Demo | 关键 import |
|---|---|
| 01 | `from langgraph.prebuilt import create_react_agent, InjectedStore`<br>`from langgraph.store.memory import InMemoryStore`<br>`from langgraph.checkpoint.memory import MemorySaver` |
| 02 | `from langmem import create_manage_memory_tool, create_search_memory_tool` |
| 03 | `from mem0 import Memory`<br>`Memory.from_config({...})` |
| 04 | `from zep_cloud.client import Zep`<br>`from zep_cloud.types import Message` |

### 10.4 复跑命令(每个 demo 都一样)

```powershell
# Windows PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:ZHIPUAI_API_KEY = "<your-key>"
python "D:\work\Memory\01_langgraph_native\run_demo.py"   # 或 02 / 03
```

### 10.5 接到生产的最小改造清单

如果你决定基于 **demo 01 + PostgresStore + 公司自研 Memory** 上线,改动是:

1. **`memory_module.py:build_checkpointer`** — `MemorySaver()` → `PostgresSaver.from_conn_string(...)`
2. **`memory_module.py:build_store`** — `InMemoryStore(index={...})` → `PostgresStore.from_conn_string(..., index={...})`
3. **`memory_module.py:save_memory` 函数体** — `store.put(...)` → 调你们自研 Memory 的 save 接口
4. **`memory_module.py:search_memory` 函数体** — `store.search(...)` → 调你们自研 Memory 的 search 接口
5. **加程序化兜底层**(可选,见 5.4) — 规则匹配触发词,LLM 漏调时补一次 store.put
6. **加 TTL 策略**(看业务) — 在 `save_memory` 里写入 `expires_at` 字段,后台 job 定期清理
7. **加多租户隔离** — `namespace` 第一层放 `tenant_id`,所有 `user_id` 在其下
8. **加 Delete cascade** — 用户删除请求要级联清空所有 namespace 下数据

其它全部不动:`agent.py` / `run_demo.py` / `dynamic_prompt` / `load_relevant_memories` 这些上层代码原封保留。

---

## 结语

**这次实测给出的最关键结论**:

> LangGraph 把 memory 拆成 `Checkpointer` + `Store` 两层是工程上极其正确的设计。短期内你**几乎不需要换 LangGraph 框架本身**,只需要根据业务诉求决定 `Store` 后端是 Mem0 / Zep / 自研 / Postgres+pgvector。

具体到你的项目:**从 demo 01 起步,把 `store.put` / `store.search` 替换成公司接口,上层不动**,是改动最小、风险最低、保留最大灵活性的方案。等业务上线跑稳了,再根据真实数据决定要不要换成 Mem0 那种"自动抽取"模式。

---

*报告生成于 2026-05-26,实测环境 Windows + Python 3.12.7 + 智谱 `glm-4-flash`。*
