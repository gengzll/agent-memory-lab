"""
PersistentInMemoryStore —— LangGraph InMemoryStore 的文件持久化版

行为:
  - 跟 InMemoryStore 一样:put / get / delete / search(含向量召回) / list_namespaces
  - 启动时如果 persist_path 文件存在,把里面的 (namespace, key, value) 全部 load 进来
  - 每次 put / delete 后把当前全量内容写回 persist_path(原子写:先写 .tmp 再 rename)
  - namespace 用 user_id 做第二层(沿用 ("memories", user_id) 约定)即可天然做用户隔离

适用范围:
  小型 demo / 单机单进程场景。每次写都全量 dump,数据量 < 几千条很爽,几万条会慢。
  生产环境换 PostgresStore(pgvector 索引)或自研后端。

文件格式:
  JSON 数组,每个 item 含 namespace(list)、key、value(dict),例如
  [
    {"namespace": ["memories", "alice"], "key": "uuid-1",
     "value": {"content": "用户喜欢 Python", "category": "preference", ...}},
    ...
  ]
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore


class PersistentInMemoryStore(InMemoryStore):
    """文件持久化版 InMemoryStore。

    Args:
        persist_path: 持久化 JSON 文件路径。None 时退化为纯内存(与 InMemoryStore 等价)。
        index: 透传给父类,用于配置 embedding 向量索引(同 InMemoryStore)。
    """

    def __init__(
        self,
        persist_path: str | os.PathLike[str] | None = None,
        *,
        index: dict | None = None,
    ) -> None:
        # InMemoryStore 1.2.x: 只接受 index kwarg
        if index is not None:
            super().__init__(index=index)
        else:
            super().__init__()
        self.persist_path: Path | None = Path(persist_path) if persist_path else None
        self._loading = False  # _load 期间静默,避免每条 put 触发一次 _save
        if self.persist_path is not None and self.persist_path.exists():
            self._load_from_disk()

    # -------- 持久化点:put / delete 后写盘 --------

    def put(self, namespace, key, value, *args, **kwargs):  # type: ignore[override]
        result = super().put(namespace, key=key, value=value, *args, **kwargs)
        if not self._loading:
            self._save_to_disk()
        return result

    def delete(self, namespace, key, *args, **kwargs):  # type: ignore[override]
        result = super().delete(namespace, key=key, *args, **kwargs)
        if not self._loading:
            self._save_to_disk()
        return result

    # -------- I/O --------

    def _load_from_disk(self) -> None:
        """从 persist_path 读 JSON,把每条 (namespace, key, value) 放回内存。

        load 期间 _loading=True,super().put 不会再触发 _save 形成循环。
        """
        assert self.persist_path is not None
        try:
            raw = self.persist_path.read_text(encoding="utf-8")
            data: list[dict[str, Any]] = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError) as e:
            print(f"[persistent_store] 警告:读取 {self.persist_path} 失败 ({e}),从空状态开始")
            return

        self._loading = True
        try:
            for entry in data:
                ns = tuple(entry["namespace"])
                super().put(ns, key=entry["key"], value=entry["value"])
        finally:
            self._loading = False

    def _save_to_disk(self) -> None:
        """全量 dump 当前所有 item 到 persist_path。原子写。"""
        if self.persist_path is None:
            return
        items: list[dict[str, Any]] = []
        try:
            for ns in self.list_namespaces():
                for it in self.search(ns, limit=100_000):
                    items.append({
                        "namespace": list(ns),
                        "key": it.key,
                        "value": it.value,
                    })
        except Exception as e:
            print(f"[persistent_store] 警告:dump 失败 ({e})")
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写:先写 tmp 再 rename,避免写到一半被读到
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=self.persist_path.name + ".",
            suffix=".tmp",
            dir=str(self.persist_path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.persist_path)
        except Exception:
            # 清理 tmp
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -------- 工具方法 --------

    def reset(self) -> None:
        """清空内存内容并删除持久化文件。"""
        # 删掉所有 namespace 下的所有 key
        for ns in list(self.list_namespaces()):
            for it in self.search(ns, limit=100_000):
                super().delete(ns, key=it.key)
        if self.persist_path is not None and self.persist_path.exists():
            self.persist_path.unlink()


def reset_user(store: PersistentInMemoryStore, user_id: str) -> int:
    """清空单个用户的全部记忆(只动 ("memories", user_id) 这个 namespace)。

    返回删除条数。其他用户的记忆不受影响。
    """
    ns = ("memories", user_id)
    items = store.search(ns, limit=100_000)
    for it in items:
        store.delete(ns, key=it.key)
    return len(items)
