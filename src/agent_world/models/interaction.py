"""
交互图核心模型 —— 实体属性、接口、有向边
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


# ─── 实体接口（交互端口） ───

class EntityInterface(BaseModel):
    """实体的一个公开交互接口"""
    interface_id: str                            # 全局唯一 ID: "npc_<id>_work"
    entity_id: str                               # 所属实体 ID
    name: str                                    # 接口名: "工作接口" / "训练接口"
    description: str                             # LLM 用描述：该接口的完整语义
    input_schema: dict[str, str] = Field(default_factory=lambda: {
        "source_entity": "发起交互的实体 ID",
        "action_params": "动作参数（JSON dict）",
    })
    output_schema: dict[str, str] = Field(default_factory=lambda: {
        "effect_type": "属性变更类型: set/add/sub",
        "effect_target": "变更哪个属性名",
        "effect_value": "变更数值",
        "effect_description": "交互结果描述",
    })

    def to_prompt_block(self) -> str:
        """格式化接口描述供 LLM 使用"""
        lines = [
            f"  - 接口 [{self.interface_id}]：{self.name}",
            f"    描述：{self.description}",
        ]
        return "\n".join(lines)


# ─── 有向边 ───

class InteractionEdge(BaseModel):
    """
    一条有向交互边：source → target。
    LLM 在每次 tick 推导出要激活的边集合。
    quantity 表示持有关系中的物品数量（0=非库存边）。
    """
    edge_id: str                                 # 唯一边 ID
    source_entity_id: str                        # 起点实体
    source_interface_id: str                     # 起点的接口
    target_entity_id: str                        # 终点实体
    target_interface_id: str                     # 终点的接口
    description: str = ""                        # 边含义描述（LLM 可读）
    is_active: bool = True                       # 该边本 tick 是否激活
    quantity: int = 0                            # 持有数量（0=非库存边）

    # 执行结果（由 engine 或 LLM 填充）
    effects: list[dict] = Field(default_factory=list)  # [{"attr":"vitality","op":"sub","value":15}, ...]
    result_text: str = ""                        # 人类可读结果

    def to_prompt_block(self) -> str:
        flags = "✓" if self.is_active else "✗"
        qty = f" x{self.quantity}" if self.quantity > 0 else ""
        return (f"  edge[{self.edge_id}] {flags} "
                f"{self.source_entity_id}.{self.source_interface_id}{qty} "
                f"──({self.description})──▸ "
                f"{self.target_entity_id}.{self.target_interface_id}")


# ─── 交互结果（执行副效应用） ───

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
    节点 = 实体接口，边 = 交互关系。
    """
    edges: list[InteractionEdge] = Field(default_factory=list)
    active_edges: list[str] = Field(default_factory=list)  # 本 tick 激活的 edge_id

    def add_edge(self, edge: InteractionEdge):
        self.edges.append(edge)

    def get_active(self) -> list[InteractionEdge]:
        return [e for e in self.edges if e.edge_id in self.active_edges]

    def set_active(self, edge_ids: list[str]):
        self.active_edges = edge_ids

    def reset(self):
        """Tick 开始前清空活跃状态"""
        self.active_edges.clear()
        for e in self.edges:
            e.is_active = False
            e.effects.clear()
            e.result_text = ""

    def to_prompt_block(self, max_edges: int = 50) -> str:
        """格式化全部可选边供 LLM 使用"""
        active = [e for e in self.edges if e.is_active]
        parts = []
        for e in active[:max_edges]:
            parts.append(e.to_prompt_block())
            for eff in e.effects[:3]:
                parts.append(f"    → {eff.get('description', '')}")
        return "\n".join(parts) if parts else "  （无可选交互边）"

    def graph_summary(self) -> str:
        """简要列出所有边供 prompt 使用"""
        lines = []
        for e in self.edges:
            qty = f" x{e.quantity}" if e.quantity > 0 else ""
            lines.append(f"  {e.source_entity_id}{qty} ──▸ {e.target_entity_id}  ({e.description})")
        return "\n".join(lines)
