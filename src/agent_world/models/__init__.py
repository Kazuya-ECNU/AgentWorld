"""
NPC 模型扩展 - 物理属性与记忆标签系统

新增：
- PhysicalAttributes: 记忆力/体能/健康
- PersonaTags: 记忆驱动的标签（工作标签/出生地/喜好等）
- init_npcs(): 多样化 NPC 初始化
"""

from .npc import NPC, NPCRole, NPCStatus, Position
from .world import World, WorldTime

__all__ = ["NPC", "NPCRole", "NPCStatus", "Position", "World", "WorldTime"]