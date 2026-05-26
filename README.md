# agent-memory-lab

> A hands-on lab comparing **5 memory integration patterns** for LangGraph-based chatbot agents — from the simplest (hand-rolled tools) to the most sophisticated (full agent runtime).

## 🎯 Why this repo

Production chatbots need memory. The ecosystem offers many options — LangGraph store, langmem, Mem0, Zep, Letta/MemGPT — but how do they actually compare?

This repo provides **5 runnable demos** + a **prompt-engineering experiment**, with:

- Exact code differences between the 5 approaches
- Empirical results (what gets stored, what gets recalled, what gets lost)
- Real-world trade-offs (persistence, token cost, deployment overhead)
- A decision framework for choosing the right one

## 📚 The 5 demos — from simple to complex

| # | Demo | Complexity | External deps | Core idea |
|---|---|---|---|---|
| 01 | [`01_langgraph_native`](01_langgraph_native/) | ⭐ | none | Hand-rolled `save_memory` / `search_memory` tools; `InjectedStore` + `RunnableConfig` 解耦 |
| 02 | [`02_langmem`](02_langmem/) | ⭐⭐ | +langmem | Factory tools replace 60 lines with 2; LLM standardizes content before write |
| 03 | [`03_mem0`](03_mem0/) | ⭐⭐⭐ | +mem0 +chroma | **Server-side LLM auto-extracts** facts every turn; persists to disk by default |
| 04 | [`04_zep`](04_zep/) | ⭐⭐⭐⭐ | +Zep server | Knowledge graph + temporal facts (`valid_from` / `valid_to`) + auto session summary |
| 05 | [`05_letta`](05_letta/) | ⭐⭐⭐⭐⭐ | +Letta server | **Full agent runtime** (NOT a LangGraph plug-in!) — three-tier memory: core / recall / archival |

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

### 2. Set API key (we use ZhipuAI `glm-4-flash` via OpenAI-compatible)

```powershell
# Windows PowerShell
$env:ZHIPUAI_API_KEY = "<your-key>"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 3. Run any demo

```powershell
python "01_langgraph_native/run_demo.py"   # ⭐ simplest, no external service
python "02_langmem/run_demo.py"            # ⭐⭐ langmem factory tools
python "03_mem0/run_demo.py"               # ⭐⭐⭐ Mem0 self-host (auto chroma persist)
```

For 04 (Zep) and 05 (Letta), see their own files — both need an external service running. Quick links:
- Zep self-host: `docker run -p 8000:8000 ghcr.io/getzep/zep:latest`
- Letta self-host: `pip install letta && letta server`

### 4. Interactive report (Streamlit)

```bash
streamlit run app.py
```

Browse 9 sections including: side-by-side demo comparison, prompt experiments, industry product comparison, and an interactive prompt tester (paste your key, run live).

## 🧪 Bonus: prompt engineering experiment

`experiments/prompt_variants.py` runs **4 prompt variants × 2 inputs** to measure how reliably `glm-4-flash` actually calls `save_memory`.

**Spoiler**: even `temperature=0` doesn't help. Model-level non-determinism caps prompt optimization around **50% hit rate**. See REPORT § 5 for the full data and the recommended programmatic fallback.

## 📂 Repo layout

```
agent-memory-lab/
├── README.md                          ← you are here
├── REPORT.md                          ← detailed empirical report
├── app.py                             ← Streamlit interactive report
├── requirements.txt                   ← combined deps
│
├── 01_langgraph_native/               ⭐ hand-rolled tools
├── 02_langmem/                        ⭐⭐ factory tools
├── 03_mem0/                           ⭐⭐⭐ auto fact extraction + persist
├── 04_zep/                            ⭐⭐⭐⭐ knowledge graph + temporal
├── 05_letta/                          ⭐⭐⭐⭐⭐ full agent runtime
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
