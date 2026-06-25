# agent-memory-lab

> A hands-on lab comparing **5 memory integration patterns** for LangGraph-based chatbot agents — from the simplest (hand-rolled tools) to the most sophisticated (full agent runtime).

## 🎯 Why this repo

Production chatbots need memory. The ecosystem offers many options — LangGraph store, langmem, Mem0, Zep, Letta/MemGPT — but how do they actually compare?

This repo provides **5 demos** (3 fully tested end-to-end on your laptop, 2 code templates that need external services to run) + a **prompt-engineering experiment**, with:

- Exact code differences between the 5 approaches
- Empirical results (what gets stored, what gets recalled, what gets lost)
- Real-world trade-offs (persistence, token cost, deployment overhead)
- A decision framework for choosing the right one

## 📌 Scope of this lab

> **This repo focuses on the *memory* dimension only.** Agent framework selection (LangGraph vs Letta vs CrewAI vs AutoGen vs LlamaIndex) is a separate orthogonal question we do **not** unpack here. We assume **LangGraph is your main framework** and explore which memory backend fits — with one explicit exception: Letta (demo 05) is itself a full agent runtime that *replaces* LangGraph, included because its three-tier memory architecture is too important to skip.

## 📚 The demos — from simple to complex

| # | Demo | GitHub ⭐ | Complexity | External deps | Core idea |
|---|---|---|---|---|---|
| 01 | [`01_langgraph_native`](01_langgraph_native/) | [33.3k](https://github.com/langchain-ai/langgraph) (whole framework) | ⭐ | none | Hand-rolled `save_memory` / `search_memory` tools; `InjectedStore` + `RunnableConfig` 解耦 |
| 02 | [`02_langmem`](02_langmem/) | [1.5k](https://github.com/langchain-ai/langmem) | ⭐⭐ | +langmem | Factory tools replace 60 lines with 2; LLM standardizes content before write |
| 03 | [`03_mem0`](03_mem0/) | [**57k**](https://github.com/mem0ai/mem0) | ⭐⭐⭐ | +mem0 +chroma | **Server-side LLM auto-extracts** facts every turn; persists to disk by default |
| 04 | [`04_zep`](04_zep/) | [4.6k](https://github.com/getzep/zep) | ⭐⭐⭐⭐ | +Zep server | Knowledge graph + temporal facts (`valid_from` / `valid_to`) + auto session summary |
| 05 | [`05_letta`](05_letta/) | [23k](https://github.com/letta-ai/letta) | ⭐⭐⭐⭐⭐ | +Letta server | **Full agent runtime** (NOT a LangGraph plug-in!) — three-tier memory: core / recall / archival |
| 06 | [`06_mem0_standalone`](06_mem0_standalone/) | (same mem0) | ⭐⭐ | +mem0 +openai | **mem0 without LangGraph** — raw `openai` SDK loop + `mem.add()` / `mem.search()`; proves mem0 is a standalone lib, not a framework component |

> Star counts as of 2026-05. Note **mem0 has the most stars by far** — but that's not the same as "most appropriate for your case"; see [REPORT § 8](REPORT.md#8-选型决策框架) for selection criteria.

> **Demo 06 is an orthogonal cut, not a step up the ladder.** It re-implements demo 03's mem0 usage with *zero* LangGraph — answering "如果我自己调 LLM endpoint,不用任何 agent 框架,还能用 mem0 吗?" (yes). The `build_memory` / `load_relevant_memories` / `ingest_turn` functions are byte-for-byte identical to demo 03 — only the LangGraph tool-binding glue is dropped.

### Capability ladder

```
 01 langgraph_native        — short-term + long-term + semantic + LLM-managed
                  ↓ add:    LLM-driven content normalization
 02 langmem                 — same + factory-generated tools (less boilerplate)
                  ↓ add:    automatic fact extraction (no longer relies on LLM self-call)
                            + default disk persistence
 03 mem0                    — same + program-controlled writes + chroma persist
                  ↓ add:    knowledge graph (entity + relations)
                            + temporal facts (when did this become true)
                            + auto session summary
 04 zep                     — same + Neo4j-backed graph + Graphiti engine
                  ↓ paradigm shift (not "more layers" — different framework):
 05 letta (MemGPT)          — abandons LangGraph; LLM-as-OS pages memory in/out
                              core memory always in context, archival is infinite disk

 ── orthogonal cut (not "more layers" — fewer) ──
 06 mem0_standalone         — demo 03's mem0, minus LangGraph: raw openai SDK
                              loop. Shows mem0 ≠ framework component; you keep
                              mem.add()/search() and write your own chat loop.
```

Read [REPORT.md](REPORT.md) for full implementation details, empirical data, and decision framework.

## 🚀 Quick start

### 1. Install deps

```bash
pip install -r requirements.txt
```

For offline / mainland China:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Set API key

All demos route through `llm_factory.py` (repo root) — one set of env vars, every demo follows.

**Default — ZhipuAI `glm-4-flash` + `embedding-2`** (only `OPENAI_API_KEY` needed):

```powershell
$env:OPENAI_API_KEY = "<your-zhipu-key>"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

**Switch provider** — override `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_EMBEDDING_MODEL`:

```powershell
# Official OpenAI
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# DeepSeek (LLM) + ZhipuAI (embedding — DeepSeek 没有 embedding service)
$env:OPENAI_API_KEY = "<deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:OPENAI_MODEL = "deepseek-chat"
$env:ZHIPUAI_API_KEY = "<zhipu-key>"     # embedding 仍走智谱
$env:OPENAI_EMBEDDING_MODEL = "embedding-2"
```

> **历史兼容**: 老的 `ZHIPUAI_API_KEY` 仍能用 —— 当 `OPENAI_API_KEY` 未设置时,`llm_factory` 会自动 fallback。Demo 05 (Letta) 自带 LLM,不走这套工厂。

### 3. Run any demo

```powershell
python "01_langgraph_native/run_demo.py"   # ⭐ simplest, no external service
python "02_langmem/run_demo.py"            # ⭐⭐ langmem factory tools (现在支持 update/delete)
python "03_mem0/run_demo.py"               # ⭐⭐⭐ Mem0 self-host (auto chroma persist)
python "06_mem0_standalone/run_demo.py"    # ⭐⭐ mem0 WITHOUT LangGraph (raw openai SDK)
python "06_mem0_standalone/run_demo.py" --interactive   # 自己跟它聊
```

**持久化** — Demo 01 / 02 / 03 / 06 现在默认都把记忆按 `user_id` 落盘,**跨进程恢复**。Demo 04 走 Zep server 的 session 数据,Demo 05 走 Letta 自己的数据库,无需本地落盘。

| Demo | 持久化方式 | 落盘位置(默认) | 跨进程清理命令 |
|---|---|---|---|
| **01 langgraph_native** | `PersistentInMemoryStore`(JSON,继承 InMemoryStore) | `./01_langgraph_native/store.json` | `--reset` 全清 / `--reset-user alice` 单用户 / `--no-persist` 关闭持久化 |
| **02 langmem** | 同上 | `./02_langmem/store.json` | 同上 |
| **03 mem0** | chroma 向量库本地落盘 | `./mem0_chroma/` | `--reset` 全清 / `--reset-user alice` 单用户 |
| **06 mem0_standalone** | chroma(同 03,但无 LangGraph) | `./06_mem0_standalone/mem0_chroma_standalone/` | `--reset` 全清 / `--reset-user alice` 单用户 |

所有 `store.json` / `mem0_chroma*/` 已经写进 `.gitignore`,不会被推到 GitHub。

```powershell
# 看 alice 跨进程能不能恢复
python "01_langgraph_native/run_demo.py"            # 第一次跑,alice 的记忆落到 store.json
python "01_langgraph_native/run_demo.py"            # 第二次跑,alice 的记忆从 store.json 恢复
python "01_langgraph_native/run_demo.py" --reset    # 全清重来
```

For 04 (Zep) and 05 (Letta), see their own files — both need an external service.

> ⚠️ **Vendor status (as of 2026-05)** — recommendations have shifted to Cloud:
> - **Zep**: Community Edition self-host docker is **deprecated** (per official GitHub README). Use **Zep Cloud** (`export ZEP_API_KEY=...`, existing demo code works as-is), or use the open-source **[Graphiti](https://github.com/getzep/graphiti)** for self-hosted graph memory (but demo 04 code would need rewriting since Graphiti's API differs from Zep client).
> - **Letta**: official docs guide users to **Letta Cloud** (`export LETTA_API_TOKEN=...`). Local `letta server` from the pip package may still work but is no longer featured in the main docs — verify with [docs.letta.com](https://docs.letta.com) before relying on it.

### 4. Interactive report (Streamlit)

```bash
streamlit run app.py
```

Browse 9 sections including: side-by-side demo comparison, prompt experiments, industry product comparison, and an interactive prompt tester. Section 9 lets you **switch provider in the UI** — paste key, override `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_EMBEDDING_MODEL`, then run live against any OpenAI-compatible endpoint.

## 🧪 Bonus: prompt engineering experiment

`experiments/prompt_variants.py` runs **4 prompt variants × 2 inputs** to measure how reliably `glm-4-flash` actually calls `save_memory`.

**Spoiler**: even `temperature=0` doesn't help. Model-level non-determinism caps prompt optimization around **50% hit rate**. See REPORT § 5 for the full data and the recommended programmatic fallback.

## 📂 Repo layout

```
agent-memory-lab/
├── README.md                          ← you are here
├── REPORT.md                          ← detailed empirical report
├── app.py                             ← Streamlit interactive report
├── llm_factory.py                     ← 共享 LLM/embedding 工厂(env-var 驱动 provider)
├── persistent_store.py                ← Demo 01/02 共用的 JSON 持久化 store
├── requirements.txt                   ← combined deps
│
├── 01_langgraph_native/               ⭐ hand-rolled tools(JSON 持久化)
├── 02_langmem/                        ⭐⭐ factory tools(JSON 持久化)
├── 03_mem0/                           ⭐⭐⭐ auto fact extraction(chroma 持久化)
├── 04_zep/                            ⭐⭐⭐⭐ knowledge graph + temporal
├── 05_letta/                          ⭐⭐⭐⭐⭐ full agent runtime
├── 06_mem0_standalone/                ⭐⭐ mem0 without LangGraph(raw openai SDK)
│
└── experiments/                       ← supporting experiments
    ├── prompt_variants.py             ← 4 prompts × 2 inputs hit-rate analysis
    └── mem0_persistence.py            ← cross-process persistence verification
```

## 🔑 Tested versions

| Package | Version |
|---|---|
| Python | 3.12.7 |
| langgraph | 1.2.1 |
| langchain-core | 1.4.0 |
| langchain-openai | 1.2.2 |
| langchain-community | 0.4.2 |
| langmem | 0.0.30 |
| mem0ai | 2.0.2 |
| chromadb | 1.5.9 |
| zhipuai | 2.1.5 |

(All tested 2026-05-26 on Windows 11 + Python 3.12.7.)

## 📜 License

MIT
