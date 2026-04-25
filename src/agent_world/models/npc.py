# NPC Data Model

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class NPCRole(str, Enum):
    """NPC 职业类型"""
    MERCHANT = "merchant"       # 商人
    GUARD = "guard"             # 守卫
    FARMER = "farmer"           # 农民
    SCHOLAR = "scholar"         # 学者
    ARTISAN = "artisan"         # 工匠
    MINER = "miner"             # 矿工
    HEALER = "healer"           # 治疗师
    BARBARIAN = "barbarian"     # 野蛮人
    SAGE = "sage"               # 贤者
    WANDERER = "wanderer"      # 流浪者


class NPCStatus(str, Enum):
    """NPC 当前状态"""
    IDLE = "idle"              # 空闲
    WORKING = "working"        #工作中
    TRAVELING = "traveling"    # 移动中
    SOCIALIZING = "socializing" # 社交中
    RESTING = "resting"        # 休息中


class Position(BaseModel):
    """NPC 位置"""
    zone_id: str = "village_square"
    x: float = 0.0
    y: float = 0.0


class MemoryEntry(BaseModel):
    """NPC 记忆条目"""
    event: str
    timestamp: datetime
    importance: float = Field(ge=0.0, le=1.0, default=0.5)  # 重要性 0-1
    related_npc_ids: list[str] = Field(default_factory=list)
    location: str | None = None  # 事件发生地点
    goal: str | None = None      # 关联的目标


class NPC(BaseModel):
    """NPC 数据模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    role: NPCRole

    # 等级与属性
    level: int = Field(ge=1, le=100, default=1)
    attributes: dict = Field(default_factory=lambda: {
        "strength": 10,
        "intelligence": 10,
        "charisma": 10,
        "endurance": 10,
        "wisdom": 10
    })

    # 背包与位置
    inventory: list[str] = Field(default_factory=list)
    position: Position = Field(default_factory=Position)

    # 记忆与关系
    memory: list[MemoryEntry] = Field(default_factory=list)
    relationships: dict = Field(default_factory=dict)  # {npc_id: affinity (-100 to 100)}

    # 状态与时间
    status: NPCStatus = NPCStatus.IDLE
    vitality: float = Field(default=100.0, ge=0.0, le=100.0)  # 活力/能量 0-100
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def drain_energy(self, amount: float):
        """消耗活力"""
        self.vitality = max(0.0, self.vitality - amount)
        if self.vitality < 20:
            self.status = NPCStatus.RESTING

    def restore_energy(self, amount: float):
        """恢复活力"""
        self.vitality = min(100.0, self.vitality + amount)

    def add_memory(self, event: str, importance: float = 0.5, related_npcs: list[str] | None = None, location: str | None = None, goal: str | None = None):
        """添加记忆"""
        entry = MemoryEntry(
            event=event,
            timestamp=datetime.now(),
            importance=importance,
            related_npc_ids=related_npcs or [],
            location=location or self.position.zone_id,
            goal=goal,
        )
        self.memory.append(entry)
        self.updated_at = datetime.now()

    def update_relationship(self, npc_id: str, delta: int):
        """更新与某 NPC 的关系值"""
        if npc_id not in self.relationships:
            self.relationships[npc_id] = 0
        self.relationships[npc_id] = max(-100, min(100, self.relationships[npc_id] + delta))
        self.updated_at = datetime.now()