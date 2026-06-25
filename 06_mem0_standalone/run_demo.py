"""
三个 session 演示 (mem0 standalone 版) —— 证明不靠框架也能跨 session 记忆

跟 03_mem0 同样的剧情,但**完全没有 LangGraph**:
  Session 1 (alice): 透露身份 + 偏好 → mem0 自动抽取入库
  Session 2 (alice): 新对话,不传任何 history(短期记忆为空!)
                     → 能答对偏好,只能是因为 mem0 的长期记忆召回生效
  Session 3 (bob):   换用户 → 验证 user_id 隔离,bob 看不到 alice 的记忆

Session 2 不传 history 是关键:LangGraph 版靠 checkpointer 也可能记住,
这里没有 checkpointer,所以"记得"必然来自 mem0 长期记忆 —— 证明更干净。

跑法:
    $env:OPENAI_API_KEY = "<key>"          # 默认智谱;OpenAI/DeepSeek 见 ../llm_factory.py
    python run_demo.py                      # 增量(沿用上次落盘的记忆)
    python run_demo.py --reset              # 清空整个 chroma 目录后重跑
    python run_demo.py --reset-user alice   # 只清 alice,保留 bob
    python run_demo.py --interactive        # 进入交互模式,自己跟它聊
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import build_client, chat
from llm_factory import get_api_key, get_model
from memory_module import (
    DEFAULT_PERSIST_DIR,
    build_memory,
    reset_user_memory,
)


def run_turn(client, mem, text: str, user_id: str) -> str:
    """跑一轮并打印(无 history,凸显长期记忆召回)。"""
    print(f"\n[user={user_id}]")
    print(f"  User: {text}")
    reply = chat(client, mem, text, user_id, verbose=True)
    print(f"  Bot : {reply}")
    return reply


def dump_memory(mem, user_id: str) -> None:
    print(f"\n--- mem0 当前 user_id={user_id} 的全部 memory ---")
    try:
        items = mem.get_all(filters={"user_id": user_id})
        items = items.get("results", []) if isinstance(items, dict) else items
    except Exception as e:
        print(f"  (读取失败: {e})")
        return
    if not items:
        print("  (空)")
        return
    for it in items:
        print(f"  - {it.get('memory', it)}")


def interactive(client, mem) -> None:
    """交互模式 —— 自己跟它聊,体验'纯手撸 + mem0'的真实用法。"""
    print("\n进入交互模式。命令:")
    print("  /user <id>   切换当前用户")
    print("  /dump        查看当前用户的全部记忆")
    print("  /exit        退出")
    user_id = "alice"
    history: list[dict] = []  # 这就是你自己管理的短期记忆
    while True:
        try:
            text = input(f"\n[{user_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue
        if text == "/exit":
            print("再见。")
            break
        if text == "/dump":
            dump_memory(mem, user_id)
            continue
        if text.startswith("/user "):
            user_id = text.split(maxsplit=1)[1].strip()
            history = []  # 换人清空短期历史(长期记忆仍按 user_id 召回)
            print(f"已切换到 user={user_id}")
            continue

        reply = chat(client, mem, text, user_id, history=history, verbose=True)
        print(f"Bot: {reply}")
        # 维护短期记忆:把这轮加进 history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mem0 standalone demo (no LangGraph)")
    p.add_argument("--reset", action="store_true",
                   help="清空整个 chroma 目录后从零开始")
    p.add_argument("--reset-user", metavar="USER_ID",
                   help="只清空指定 user_id 的记忆")
    p.add_argument("--interactive", action="store_true",
                   help="进入交互模式,自己跟它聊")
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR,
                   help=f"chroma 落盘目录,默认 {DEFAULT_PERSIST_DIR}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    get_api_key()  # 没配 key 在此抛错
    print(f"模型: {get_model()}    持久化目录: {args.persist_dir}")

    client = build_client()
    mem = build_memory(persist_dir=args.persist_dir, reset=args.reset)

    if args.reset_user and not args.reset:
        n = reset_user_memory(mem, args.reset_user)
        print(f"[mem0] 已清空 user_id={args.reset_user} 的 {n} 条记忆")

    if args.interactive:
        interactive(client, mem)
        return

    print("\n" + "=" * 70)
    print("Session 1 — alice 告诉助手一些事实和偏好(mem0 自动抽取入库)")
    print("=" * 70)
    run_turn(client, mem, "我叫 Alice,在量化交易做研究员", "alice")
    run_turn(client, mem, "我喜欢回答尽量简短,代码示例只用 Python", "alice")
    dump_memory(mem, "alice")

    print("\n" + "=" * 70)
    print("Session 2 — 新对话,不传任何 history(短期记忆为空!)")
    print("        能答对偏好 → 只可能来自 mem0 长期记忆召回")
    print("=" * 70)
    run_turn(client, mem, "帮我写一段计算夏普比率的代码", "alice")

    print("\n" + "=" * 70)
    print("Session 3 — 换用户 bob,验证 user_id 隔离")
    print("=" * 70)
    run_turn(client, mem, "你知道我叫什么吗?", "bob")
    dump_memory(mem, "bob")


if __name__ == "__main__":
    main()
