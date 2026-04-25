"""
Memory Manager - 基于 NPC 属性/状态/标签的记忆管理

控制记忆的：
- 重要性权重
- 保留期限
- 遗忘速率
- 检索优先级
"""

from typing import Optional
from agent_world.models.npc import NPC, MemoryEntry


class MemoryManager:
    """记忆管理器"""

    # 标签对记忆重要性的影响
    TAG_IMPORTANCE_WEIGHTS = {
        "greedy": {"trade": 2.0, "gold": 2.0, "wealth": 1.8},
        "social": {"conversation": 2.0, "chat": 1.8, "greeting": 1.5},
        "hardworking": {"work": 2.0, "farm": 1.8, "mine": 1.8},
        "curious": {"explore": 2.0, "discovery": 1.8, "new": 1.5},
        "lazy": {"rest": 2.0, "idle": 1.5},
        "peaceful": {"conflict": 0.5, "trade": 1.2, "socialize": 1.5},
        "ambitious": {"goal": 2.0, "achievement": 1.8, "reputation": 1.5},
        "cautious": {"danger": 2.0, "safety": 1.8, "warning": 1.5},
    }

    # 心情对记忆保留的影响
    MOOD_RETENTION = {
        "happy": 1.2,      # 愉快时记忆更持久
        "neutral": 1.0,
        "sad": 0.8,        # 悲伤时记忆衰减更快
        "angry": 0.7,      # 愤怒时记忆不稳定
        "anxious": 0.6,    # 焦虑时记忆混乱
    }

    # 属性影响记忆容量和精度
    def get_memory_capacity(self, npc: NPC) -> int:
        """根据属性计算记忆容量"""
        base = 100
        intelligence_bonus = (npc.attributes.get("intelligence", 5) - 5) * 5
        willpower_bonus = (npc.attributes.get("willpower", 5) - 5) * 3
        return base + intelligence_bonus + willpower_bonus

    def get_memory_precision(self, npc: NPC) -> float:
        """根据感知属性计算记忆精度（0-1）"""
        perception = npc.attributes.get("perception", 5)
        return 0.5 + (perception / 10.0)  # 5->1.0, 10->1.5

    def calculate_importance(self, npc: NPC, memory: MemoryEntry) -> float:
        """计算记忆的重要性权重"""
        importance = 1.0

        # 标签权重
        for tag in npc.tags:
            if tag in self.TAG_IMPORTANCE_WEIGHTS:
                tag_weights = self.TAG_IMPORTANCE_WEIGHTS[tag]
                for key, weight in tag_weights.items():
                    if key in memory.event.lower():
                        importance *= weight

        # 心情权重
        mood = npc.npc_status.get("mood", "neutral")
        if mood in self.MOOD_RETENTION:
            importance *= self.MOOD_RETENTION[mood]

        # 压力权重（高压力降低重要记忆权重）
        stress = npc.npc_status.get("stress", 0)
        if stress > 50:
            importance *= 0.8
        elif stress > 80:
            importance *= 0.5

        # 疲劳权重
        fatigue = npc.npc_status.get("fatigue", 0)
        if fatigue > 70:
            importance *= 0.7

        return max(0.1, importance)  # 最小权重 0.1

    def should_remember(self, npc: NPC, memory: MemoryEntry) -> bool:
        """根据属性/状态决定是否保留某条记忆"""
        importance = self.calculate_importance(npc, memory)

        # 低重要性且记忆已满时遗忘
        if importance < 0.5 and len(npc.memory) >= npc.memory_limit:
            return False

        # 压力过高时遗忘不重要的社交记忆
        stress = npc.npc_status.get("stress", 0)
        if stress > 70 and importance < 0.8:
            if any(word in memory.event.lower() for word in ["chat", "talk", "greeting"]):
                return False

        return True

    def get_optimal_memory_count(self, npc: NPC) -> int:
        """根据状态计算最优记忆数量"""
        capacity = self.get_memory_capacity(npc)
        motivation = npc.npc_status.get("motivation", "normal")

        if motivation == "high":
            return int(capacity * 0.9)
        elif motivation == "low":
            return int(capacity * 0.5)
        else:
            return int(capacity * 0.7)

    def prune_memories(self, npc: NPC) -> list[MemoryEntry]:
        """修剪记忆，保留最重要的"""
        if len(npc.memory) <= self.get_optimal_memory_count(npc):
            return []

        # 计算每条记忆的重要性
        memories_with_importance = [
            (m, self.calculate_importance(npc, m)) for m in npc.memory
        ]

        # 按重要性排序
        memories_with_importance.sort(key=lambda x: x[1], reverse=True)

        # 保留最重要的
        keep_count = self.get_optimal_memory_count(npc)
        kept = [m for m, _ in memories_with_importance[:keep_count]]
        removed = [m for m, _ in memories_with_importance[keep_count:]]

        npc.memory = kept
        return removed

    def add_memory(self, npc: NPC, event: str, importance: float = 1.0) -> None:
        """添加记忆，自动修剪"""
        entry = MemoryEntry(event=event, importance=importance)
        npc.memory.append(entry)

        # 检查是否需要修剪
        if len(npc.memory) > npc.memory_limit:
            self.prune_memories(npc)

    def update_status(self, npc: NPC, delta_energy: int = 0, delta_stress: int = 0, mood_change: Optional[str] = None) -> None:
        """更新 NPC 状态"""
        if delta_energy != 0:
            npc.energy = max(0, min(100, npc.energy + delta_energy))

        if delta_stress != 0:
            npc.npc_status["stress"] = max(0, min(100, npc.npc_status.get("stress", 0) + delta_stress))

        if mood_change:
            npc.npc_status["mood"] = mood_change

        # 根据能量更新疲劳度
        if npc.energy < 20:
            npc.npc_status["fatigue"] = min(100, npc.npc_status.get("fatigue", 0) + 5)
        elif npc.energy > 80:
            npc.npc_status["fatigue"] = max(0, npc.npc_status.get("fatigue", 0) - 2)
