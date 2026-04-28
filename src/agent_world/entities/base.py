"""
WorldObject Base - 实体基类 + 状态机 + 交互接口

所有可交互物体的基类。
定义了物体的状态、属性、以及与 NPC 交互的接口。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


# === 枚举定义 ===

class ObjectState(str, Enum):
    """物体状态"""
    AVAILABLE = "available"      # 可用（空闲，无人占用）
    OCCUPIED = "occupied"        # 已被占用
    BROKEN = "broken"            # 损坏
    DEPLETED = "depleted"        # 资源耗尽（如矿脉挖空）
    LOCKED = "locked"            # 锁定（如酒馆关门）


class ObjectType(str, Enum):
    """物体类型"""
    STALL = "stall"              # 商人摊位
    FARM_PLOT = "farm_plot"      # 农田
    ORE_VEIN = "ore_vein"        # 矿脉
    BAR_COUNTER = "bar_counter"  # 酒馆吧台
    LIBRARY_DESK = "library_desk"  # 图书馆书桌
    TEMPLE_ALTAR = "temple_altar"  # 神庙祭坛
    BARRACKS_EQUIPMENT = "barracks_equipment"  # 兵营训练设施
    FOREST_HUNTING_GROUND = "forest_hunting_ground"  # 森林狩猎场


# === 交互结果 ===

class InteractionResult(BaseModel):
    """交互结果"""
    success: bool
    description: str                    # 描述发生了什么
    loot: list[str] = Field(default_factory=list)   # 获得的物品
    state_change: ObjectState | None = None  # 状态变化
    next_available_at: datetime | None = None  # 下次可用时间（如果被占用）


class Affordance(BaseModel):
    """
    物体提供的交互能力（它能做什么）。
    
    例如 FarmPlot 提供: plant, water, harvest
    """
    action: str            # 动作名称
    description: str       # 动作描述
    energy_cost: float = 0  # 消耗体力
    duration_ticks: int = 1  # 持续时间（tick 数）


# === WorldObject 基类 ===

class WorldObject(ABC):
    """
    所有可交互物体的抽象基类。
    
    设计要点：
    - 每个物体有唯一 ID、类型、状态、所在 Zone
    - 物体有可交互的 Actions（Affordances）
    - NPC 使用物体时，物体记录当前占用者
    """

    def __init__(
        self,
        object_id: str | None = None,
        object_type: ObjectType | None = None,
        zone_id: str = "",
        name: str = "",
        state: ObjectState = ObjectState.AVAILABLE,
    ):
        self.id = object_id or str(uuid4())
        self.object_type = object_type
        self.zone_id = zone_id
        self.name = name
        self.state = state
        self.current_user: str | None = None  # 当前占用者 NPC ID
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    # === 抽象方法（子类必须实现）===

    @abstractmethod
    def get_affordances(self) -> list[Affordance]:
        """返回该物体能提供的所有交互能力"""
        ...

    @abstractmethod
    def can_interact(self, npc_id: str, action: str) -> tuple[bool, str]:
        """
        检查 NPC 是否能执行该动作。
        
        Returns:
            (can_interact, reason)
        """
        ...

    @abstractmethod
    def interact(self, npc_id: str, action: str) -> InteractionResult:
        """
        执行交互。
        
        Returns:
            InteractionResult
        """
        ...

    # === 通用方法 ===

    def is_available(self) -> bool:
        """是否可用（空闲）"""
        return self.state == ObjectState.AVAILABLE and self.current_user is None

    def occupy(self, npc_id: str):
        """占用该物体"""
        self.current_user = npc_id
        self.state = ObjectState.OCCUPIED
        self.updated_at = datetime.now()

    def release(self):
        """释放该物体"""
        self.current_user = None
        if self.state == ObjectState.OCCUPIED:
            self.state = ObjectState.AVAILABLE
        self.updated_at = datetime.now()

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "id": self.id,
            "object_type": self.object_type.value if self.object_type else None,
            "zone_id": self.zone_id,
            "name": self.name,
            "state": self.state.value,
            "current_user": self.current_user,
            "affordances": [a.model_dump() for a in self.get_affordances()],
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id[:8]} state={self.state.value} user={self.current_user}>"