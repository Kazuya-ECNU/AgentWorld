"""
纯拓扑交互图模型

实体间只有有向边，边只携带数量（quantity），无接口类型标记。
节点（Entity）自带丰富自我描述（类型、角色、属性、性格、描述）。

设计原则：边 = 拓扑连接 + 数量。语义由 LLM 从节点描述推断。
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ─── 实体属性 ───

class EntityAttribute(BaseModel):
    """实体的一个属性（数值/状态/描述）"""
    name: str                                    # 属性名: vitality / strength / state
    value: Any = None                            # 当前值
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""                        # 人类可读描述

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "range": f"{self.min_value}~{self.max_value}" if self.min_value is not None or self.max_value is not None else "无限制",
            "description": self.description,
        }


# ─── 纯拓扑边 ───

class InteractionEdge(BaseModel):
    """
    一条有向边：source → target。
    边只有数量（quantity），没有类型标签。
    语义从两端的节点描述中推断：如果 NPC→Item 且 qty>0 → 持有；
    如果 Item→NPC 且 qty>0 → 归属（谁拥有）；NPC→NPC 的边 → 交互。
    """
    edge_id: str                                 # 唯一边 ID
    source_entity_id: str                        # 起点实体
    target_entity_id: str                        # 终点实体
    quantity: int = 0                            # 数量（0=无数量意义的边）
    is_active: bool = True                       # 该边本 tick 是否激活

    def to_prompt_line(self) -> str:
        """格式化一行供 LLM 使用"""
        qty = f" x{self.quantity}" if self.quantity != 0 else ""
        flags = "" if self.is_active else " [非活跃]"
        return f"  {self.source_entity_id} ──▸ {self.target_entity_id}{qty}{flags}"


# ─── 交互结果 ───

class AttributeEffect(BaseModel):
    """对实体属性的单一变更"""
    target_entity_id: str
    attribute_name: str
    operation: str = "set"                       # set / add / sub
    value: float | str | None = None
    description: str = ""


# ─── 交互图 ───

class InteractionGraph(BaseModel):
    """
    整个世界的交互有向图。
    边 = 纯拓扑连接，无接口/类型标记。
    """
    edges: list[InteractionEdge] = Field(default_factory=list)

    def add_edge(self, edge: InteractionEdge):
        # 不重复添加
        for existing in self.edges:
            if (existing.source_entity_id == edge.source_entity_id
                    and existing.target_entity_id == edge.target_entity_id):
                return
        self.edges.append(edge)

    def get_edge(self, src: str, tgt: str) -> InteractionEdge | None:
        for e in self.edges:
            if e.source_entity_id == src and e.target_entity_id == tgt:
                return e
        return None

    def remove_edge(self, src: str, tgt: str) -> bool:
        for i, e in enumerate(self.edges):
            if e.source_entity_id == src and e.target_entity_id == tgt:
                self.edges.pop(i)
                return True
        return False

    def to_prompt_block(self, max_edges: int = 100) -> str:
        """格式化全部边供 LLM 使用"""
        lines = []
        for e in self.edges[:max_edges]:
            lines.append(e.to_prompt_line())
        return "\n".join(lines) if lines else "  （无连接）"

    def graph_summary(self) -> str:
        """简要列出所有边供 prompt 使用"""
        lines = []
        for e in self.edges:
            qty = f" x{e.quantity}" if e.quantity > 0 else ""
            lines.append(f"  {e.source_entity_id}{qty} ──▸ {e.target_entity_id}")
        return "\n".join(lines)
