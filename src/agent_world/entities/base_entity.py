"""
实体基类 —— 所有可交互实体（NPC/物品/zone）继承此类。
每个实体拥有一组属性 + 一组公开接口。
"""

from __future__ import annotations
from typing import Any
from ..models.interaction import EntityAttribute, EntityInterface


class Entity:
    """
    实体基类。
    实体 = 属性集合 + 接口集合 + 拓扑连接信息。
    """

    def __init__(self, entity_id: str, name: str, entity_type: str = "generic"):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type    # "npc" / "object" / "zone"
        self.attributes: dict[str, EntityAttribute] = {}
        self.interfaces: dict[str, EntityInterface] = {}
        self.connected_entity_ids: set[str] = set()   # 拓扑连接的其他实体

    # ─── 属性管理 ───

    def set_attr(self, name: str, value: Any, **kwargs):
        if name in self.attributes:
            self.attributes[name].value = value
        else:
            self.attributes[name] = EntityAttribute(
                name=name, value=value, **kwargs
            )

    def get_attr(self, name: str) -> Any:
        attr = self.attributes.get(name)
        return attr.value if attr else None

    def modify_attr(self, name: str, delta: float) -> float:
        """数值型属性增减，返回实际增加值"""
        attr = self.attributes.get(name)
        if attr is None or not isinstance(attr.value, (int, float)):
            return 0.0
        new_val = attr.value + delta
        if attr.min_value is not None:
            new_val = max(attr.min_value, new_val)
        if attr.max_value is not None:
            new_val = min(attr.max_value, new_val)
        actual = new_val - attr.value
        attr.value = new_val
        return actual

    def attrs_to_prompt(self) -> str:
        """格式化当前属性供 LLM 使用"""
        lines = [f"  {self.name} ({self.entity_type}) 属性："]
        for a in self.attributes.values():
            lines.append(f"    {a.name} = {a.value}  ({a.description})")
        return "\n".join(lines)

    # ─── 接口管理 ───

    def add_interface(self, iface: EntityInterface):
        self.interfaces[iface.interface_id] = iface

    def get_interface(self, iface_id: str) -> EntityInterface | None:
        return self.interfaces.get(iface_id)

    def interfaces_to_prompt(self) -> str:
        lines = [f"  {self.name} 公开接口："]
        for iface in self.interfaces.values():
            lines.append(iface.to_prompt_block())
        return "\n".join(lines)

    # ─── 拓扑连接 ───

    def connect_to(self, other_entity_id: str):
        self.connected_entity_ids.add(other_entity_id)

    def disconnect_from(self, other_entity_id: str):
        self.connected_entity_ids.discard(other_entity_id)

    # ─── 序列化 ───

    def to_prompt_block(self) -> str:
        """本 tick 给 LLM 看的完整实体信息"""
        parts = [
            f"【{self.name}】（{self.entity_type}）",
        ]
        # 属性
        for a in self.attributes.values():
            parts.append(f"  {a.name}: {a.value}")
        # 接口
        for iface in self.interfaces.values():
            parts.append(f"  └─接口[{iface.interface_id}]: {iface.name} — {iface.description}")
        return "\n".join(parts)

    def dump(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "type": self.entity_type,
            "attributes": {k: v.to_dict() for k, v in self.attributes.items()},
            "interfaces": list(self.interfaces.keys()),
            "connected_to": list(self.connected_entity_ids),
        }
