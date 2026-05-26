"""新进程里查 alice 的 memory 还在不在 —— 验证 03 是否真的持久化"""
import sys
sys.path.insert(0, r"D:\work\Memory\03_mem0")
from memory_module import build_memory

m = build_memory()
r = m.get_all(filters={"user_id": "alice"})
items = r.get("results", []) if isinstance(r, dict) else r
print(f"alice memory count: {len(items)}")
for it in items[:10]:
    print(f"  - {it.get('memory', it)}")
