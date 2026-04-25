"""
Entities Module - 世界实体系统

提供可交互的 World Objects，NPC 的行为围绕"找可交互物体"进行。

Files:
  - base.py        : WorldObject 基类 + 状态机 + 交互接口
  - world_objects.py: 具体物体实现（Stall, FarmPlot, OreVein, BarCounter...）
"""

from .base import WorldObject, ObjectState, ObjectType, InteractionResult, Affordance
from .world_objects import (
    Stall,
    FarmPlot,
    OreVein,
    BarCounter,
    LibraryDesk,
    TempleAltar,
    BarracksEquipment,
    ForestHuntingGround,
    WorldObjectManager,
)
from .manager import get_entity_manager, init_entity_manager

__all__ = [
    "WorldObject",
    "ObjectState",
    "ObjectType",
    "InteractionResult",
    "Affordance",
    "Stall",
    "FarmPlot",
    "OreVein",
    "BarCounter",
    "LibraryDesk",
    "TempleAltar",
    "BarracksEquipment",
    "ForestHuntingGround",
    "WorldObjectManager",
    "get_entity_manager",
    "init_entity_manager",
]