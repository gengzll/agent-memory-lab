"""
两个 session 的演示 —— 证明长期记忆能跨 thread + 跨进程生效。

Session 1 (thread_id=t1, user_id=alice)
  用户透露身份 + 偏好,期望 agent 调用 save_memory 写入 store。

Session 2 (thread_id=t2, user_id=alice)   ← 新 thread,短期记忆清零!
  用户提个会用到偏好的问题,期望:
    - dynamic_prompt 自动把 t1 写入的 memory 召回拼到 system
    - 或 agent 主动调用 search_memory
  → 回答应当遵循 alice 在 t1 透露的偏好。

持久化:
  默认会把每个 user 的 memory 落到 ./01_langgraph_native/store.json,
  下次跑 demo / 你自己的脚本会自动从 JSON 恢复(按 user_id namespace 隔离)。

跑法:
    pip install -r requirements.txt
    # 智谱(默认):
    $env:OPENAI_API_KEY = "<zhipu-key>"
    # OpenAI:
    $env:OPENAI_API_KEY = "sk-..."
    $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
    $env:OPENAI_MODEL = "gpt-4o-mini"
    $env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    python run_demo.py                       # 增量(沿用上次记忆)
    python run_demo.py --reset               # 清空整个 store.json 后从零开始
    python run_demo.py --reset-user alice    # 只清 alice,保留 bob
    python run_demo.py --no-persist          # 关闭持久化(纯内存,旧行为)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from agent import build_agent
from llm_factory import get_api_key
from persistent_store import reset_user

DEFAULT_PERSIST_PATH = str(Path(__file__).resolve().parent / "store.json")


def chat(agent, text: str, thread_id: str, user_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    reply = result["messages"][-1].content
    print(f"\n[user={user_id} thread={thread_id}]")
    print(f"  User: {text}")
    print(f"  Bot : {reply}")
    # 打印 tool 调用轨迹,看 agent 有没有真的写/查 memory
    tool_calls = [
        m for m in result["messages"]
        if getattr(m, "type", None) == "tool"
    ]
    if tool_calls:
        for tc in tool_calls:
            print(f"  · tool[{tc.name}] -> {tc.content[:120]}")
    return reply


def dump_store(store, user_id: str) -> None:
    print(f"\n--- store 当前内容 (namespace=('memories','{user_id}')) ---")
    try:
        items = store.search(("memories", user_id), query="", limit=20)
    except Exception:
        items = store.search(("memories", user_id), limit=20)
    if not items:
        print("  (空)")
        return
    for it in items:
        print(f"  - [{it.value.get('category', '?')}] {it.value['content']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph native memory demo")
    p.add_argument(
        "--reset", action="store_true",
        help="清空整个 store.json 后从零开始(所有用户记忆都会丢)",
    )
    p.add_argument(
        "--reset-user", metavar="USER_ID",
        help="只清空指定 user_id 的记忆(保留其他用户)",
    )
    p.add_argument(
        "--no-persist", action="store_true",
        help="关闭持久化(纯内存,进程退出即丢失)",
    )
    p.add_argument(
        "--persist-path", default=DEFAULT_PERSIST_PATH,
        help=f"持久化 JSON 文件路径,默认 {DEFAULT_PERSIST_PATH}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    get_api_key()  # 没配 key 在此抛错,信息更明确

    persist_path = None if args.no_persist else args.persist_path

    if args.reset and persist_path:
        path = Path(persist_path)
        if path.exists():
            path.unlink()
            print(f"[store] reset=True,已删除 {persist_path}")

    agent, store = build_agent(persist_path=persist_path)

    if args.reset_user and not args.reset:
        n = reset_user(store, args.reset_user)
        print(f"[store] 已清空 user_id={args.reset_user} 的 {n} 条记忆")

    if persist_path:
        print(f"[store] 持久化文件: {persist_path}\n")

    print("=" * 70)
    print("Session 1 — alice 告诉 agent 一些事实和偏好")
    print("=" * 70)
    chat(agent, "我叫 Alice,在量化交易做研究员", thread_id="t1", user_id="alice")
    chat(agent, "我喜欢回答尽量简短,代码示例只用 Python", thread_id="t1", user_id="alice")

    dump_store(store, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新 thread (t2),短期记忆已清空")
    print("        看长期记忆是否能跨 thread 生效:回答应当用 Python + 简短风格")
    print("=" * 70)
    chat(agent, "帮我写一段计算夏普比率的代码", thread_id="t2", user_id="alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换个用户 bob,同一个 agent 实例")
    print("        看 namespace 隔离:bob 应当看不到 alice 的 memory")
    print("=" * 70)
    chat(agent, "你知道我叫什么吗?", thread_id="t3", user_id="bob")

    dump_store(store, "bob")


if __name__ == "__main__":
    main()
