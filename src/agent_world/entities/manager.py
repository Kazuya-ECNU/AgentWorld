"""
Shared Entity Manager - 全局共享实体管理器

解决 API 和 NPC Engine 之间共享实体状态的问题。
"""

from .world_objects import WorldObjectManager

# 全局共享实体管理器
_shared_manager: WorldObjectManager | None = None

def get_entity_manager() -> WorldObjectManager:
    """获取全局共享的实体管理器"""
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = WorldObjectManager()
    return _shared_manager

def init_entity_manager(zones: list[dict]):
    """初始化全局实体管理器"""
    global _shared_manager
    manager = get_entity_manager()
    if not manager.all():  # 尚未初始化
        manager.init_default_world(zones)
