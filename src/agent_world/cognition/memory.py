"""
Memory Store - NPC 记忆管理系统

支持：
  - 追加记忆条目（时间戳 + 事件 + 重要性 + 关联NPC）
  - 检索记忆（按时间/重要性/关键词）
  - 摘要生成（对大量记忆进行压缩）
  - 滑动窗口（Recent Context）
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json


class MemoryEntry(BaseModel):
    """记忆条目"""
    event: str                           # 事件描述
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = 0.5              # 重要性 0.0~1.0
    related_npc_ids: list[str] = Field(default_factory=list)  # 关联 NPC
    location: str = ""                   # 发生地点


class MemoryStore:
    """
    NPC 的记忆存储与管理。
    
    设计要点：
    - 所有记忆按时间顺序追加
    - 滑动窗口控制 Recent Context 大小
    - 支持按条件检索
    """

    # 滑动窗口大小（影响 "最近上下文" 的容量）
    MAX_RECENT = 10

    def __init__(self, entries: list[MemoryEntry] | None = None):
        self._entries: list[MemoryEntry] = entries or []

    # === 追加 ===

    def add(
        self,
        event: str,
        importance: float = 0.5,
        related_npcs: list[str] | None = None,
        location: str = "",
    ) -> MemoryEntry:
        """追加一条记忆"""
        entry = MemoryEntry(
            event=event,
            timestamp=datetime.now(),
            importance=importance,
            related_npc_ids=related_npcs or [],
            location=location,
        )
        self._entries.append(entry)
        return entry

    # === 查询 ===

    def get_all(self) -> list[MemoryEntry]:
        """获取全部记忆（按时间倒序）"""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)

    def get_recent(self, n: int = 5) -> list[MemoryEntry]:
        """获取最近 n 条记忆（滑动窗口）"""
        sorted_entries = self.get_all()
        return sorted_entries[:n]

    def get_by_importance(self, threshold: float = 0.5) -> list[MemoryEntry]:
        """获取重要性 >= threshold 的记忆"""
        return [e for e in self._entries if e.importance >= threshold]

    def search(self, keyword: str) -> list[MemoryEntry]:
        """搜索包含关键词的记忆"""
        return [
            e for e in self._entries
            if keyword.lower() in e.event.lower()
        ]

    def get_by_location(self, location: str) -> list[MemoryEntry]:
        """获取发生在特定地点的记忆"""
        return [e for e in self._entries if e.location == location]

    def get_related(self, npc_id: str) -> list[MemoryEntry]:
        """获取与某 NPC 相关的记忆"""
        return [e for e in self._entries if npc_id in e.related_npc_ids]

    # === 摘要 ===

    def summarize(self, max_memory: int = 20) -> str:
        """
        对记忆进行摘要。
        
        如果记忆条目 <= max_memory，全部返回；
        否则按重要性 + 时间采样，保留关键记忆。
        
        返回格式："; ".join([事件1, 事件2, ...])
        """
        if len(self._entries) <= max_memory:
            return "; ".join(e.event for e in self.get_all())

        # 重要记忆优先
        important = self.get_by_importance(0.6)
        # 剩余按时间均匀采样
        remaining_slots = max_memory - len(important)
        others = [e for e in self._entries if e not in important]
        sampled = others[-remaining_slots:] if remaining_slots > 0 else []

        combined = important + sampled
        combined.sort(key=lambda e: e.timestamp, reverse=True)
        return "; ".join(e.event for e in combined[:max_memory])

    def recent_context_string(self, n: int = 5) -> str:
        """生成 Recent Context 字符串（给 LLM 用）"""
        recent = self.get_recent(n)
        if not recent:
            return "最近没有发生什么事。"
        return "; ".join(f"{e.event}" for e in recent)

    # === 统计 ===

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # === 序列化 ===

    def to_dict(self) -> list[dict]:
        return [
            {
                "event": e.event,
                "timestamp": e.timestamp.isoformat(),
                "importance": e.importance,
                "related_npc_ids": e.related_npc_ids,
                "location": e.location,
            }
            for e in self._entries
        ]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "MemoryStore":
        """从字典列表恢复"""
        entries = [
            MemoryEntry(
                event=d["event"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                importance=d.get("importance", 0.5),
                related_npc_ids=d.get("related_npc_ids", []),
                location=d.get("location", ""),
            )
            for d in data
        ]
        return cls(entries)

    # === 裁剪 ===

    def prune(self, max_entries: int = 200):
        """
        裁剪记忆，防止无限增长。
        
        保留：
        - 重要性 >= 0.7 的记忆（全部保留）
        - 最新的一些记忆（时间序取后 max_entries 条）
        """
        if len(self._entries) <= max_entries:
            return

        high_importance = [e for e in self._entries if e.importance >= 0.7]
        recent = sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:max_entries]

        # 合并去重
        seen = set()
        merged = []
        for e in high_importance + recent:
            if id(e) not in seen:
                seen.add(id(e))
                merged.append(e)

        self._entries = sorted(merged, key=lambda e: e.timestamp, reverse=True)