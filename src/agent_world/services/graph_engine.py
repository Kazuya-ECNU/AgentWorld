"""
交互图引擎 —— 管理所有实体的交互图，驱动 LLM 推导交互边。

纯数据容器：
  1. 注册/注销实体
  2. 维护实体间拓扑连接
  3. 管理边（创建 / 设置数量 / 查询）
  4. 解析 LLM 响应
  5. 执行副效应（属性变更 / 库存变更）

LLM 推理不在本引擎中——由 NPC Prompt Builder + graph_npc_engine 逐 NPC 处理。
"""

from __future__ import annotations

import json
import logging
from typing import Any
from ..models.interaction import (
    InteractionGraph, InteractionEdge, EntityInterface,
    EntityAttribute, AttributeEffect,
)
from ..entities.base_entity import Entity

logger = logging.getLogger("graph_engine")


class GraphEngine:
    """
    交互图引擎——纯数据容器。

    职责：
    1. 实体注册 / 注销
    2. 拓扑连接（connect / build_graph）
    3. 边管理（get_or_create_edge / set_edge_quantity / modify_edge_quantity）
    4. 查询（get_inventory_view / get_held_quantity）
    5. LLM 响应解析 + 效果执行
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self.graph = InteractionGraph()

    # ─── 实体注册 ───

    def register_entity(self, entity: Entity):
        self._entities[entity.entity_id] = entity

    def unregister_entity(self, entity_id: str):
        self._entities.pop(entity_id, None)
        self.graph.edges = [e for e in self.graph.edges
                            if e.source_entity_id != entity_id
                            and e.target_entity_id != entity_id]

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def get_npcs(self) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type == "npc"]

    # ─── 拓扑构建 ───

    def connect(self, from_id: str, to_id: str):
        """在两个实体之间建立拓扑连接"""
        src = self._entities.get(from_id)
        dst = self._entities.get(to_id)
        if src and dst:
            src.connect_to(to_id)
            dst.connect_to(from_id)

    def build_graph(self):
        """
        基于实体间的拓扑连接生成交互边。
        每个连接生成两条单向边（A→B 和 B→A），每条边覆盖所有接口组合。
        """
        self.graph.edges.clear()
        edge_idx = 0

        for eid, entity in self._entities.items():
            for connected_id in entity.connected_entity_ids:
                target = self._entities.get(connected_id)
                if not target:
                    continue

                for src_iface in entity.interfaces.values():
                    for dst_iface in target.interfaces.values():
                        edge = InteractionEdge(
                            edge_id=f"e_{edge_idx}",
                            source_entity_id=eid,
                            source_interface_id=src_iface.interface_id,
                            target_entity_id=connected_id,
                            target_interface_id=dst_iface.interface_id,
                            description=f"{entity.name} 通过 {src_iface.name} 与 {target.name} 的 {dst_iface.name} 交互",
                            is_active=True,
                        )
                        self.graph.edges.append(edge)
                        edge_idx += 1

    # ─── 库存边管理 ───

    def get_or_create_edge(self, src_eid: str, src_iface: str,
                            tgt_eid: str, tgt_iface: str) -> InteractionEdge:
        """查找已有的边，没有则创建（不重新生成索引）"""
        for e in self.graph.edges:
            if (e.source_entity_id == src_eid
                and e.source_interface_id == src_iface
                and e.target_entity_id == tgt_eid
                and e.target_interface_id == tgt_iface):
                return e
        edge = InteractionEdge(
            edge_id=f"e_{len(self.graph.edges)}",
            source_entity_id=src_eid,
            source_interface_id=src_iface,
            target_entity_id=tgt_eid,
            target_interface_id=tgt_iface,
            description="库存持有关系",
            is_active=True,
            quantity=0,
        )
        self.graph.edges.append(edge)
        return edge

    def set_edge_quantity(self, src_eid: str, tgt_eid: str, qty: int):
        """设置两个实体间的库存数量（自动在持有→可持有间查找）"""
        for e in self.graph.edges:
            if e.source_entity_id == src_eid and e.target_entity_id == tgt_eid \
               and e.source_interface_id.endswith('_hold') \
               and e.target_interface_id.endswith('_holdable'):
                e.quantity = max(0, qty)
                return
        # 没找到匹配边则创建一条
        edge = self.get_or_create_edge(src_eid, f"{src_eid[:12]}_hold",
                                        tgt_eid, f"{tgt_eid[:12]}_holdable")
        edge.quantity = max(0, qty)

    def modify_edge_quantity(self, src_eid: str, tgt_eid: str, delta: int) -> int:
        """增减库存数量，返回实际变化值"""
        old_qty = 0
        for e in self.graph.edges:
            if e.source_entity_id == src_eid and e.target_entity_id == tgt_eid \
               and e.source_interface_id.endswith('_hold') \
               and e.target_interface_id.endswith('_holdable'):
                old_qty = e.quantity
                new_qty = max(0, e.quantity + delta)
                delta_actual = new_qty - e.quantity
                e.quantity = new_qty
                return delta_actual
        # 不存在则创建
        edge = self.get_or_create_edge(src_eid, f"{src_eid[:12]}_hold",
                                        tgt_eid, f"{tgt_eid[:12]}_holdable")
        old_qty = 0
        edge.quantity = max(0, delta)
        return edge.quantity

    def get_held_quantity(self, entity_id: str, item_entity_id: str) -> int:
        """查询某个实体持有某个物品的数量"""
        for e in self.graph.edges:
            if e.source_entity_id == entity_id and e.target_entity_id == item_entity_id \
               and e.source_interface_id.endswith('_hold'):
                return e.quantity
        return 0

    def get_inventory_view(self, entity_id: str) -> dict[str, int]:
        """返回实体的库存视图 {item_name: qty}"""
        result = {}
        for e in self.graph.edges:
            if e.source_entity_id == entity_id and e.quantity > 0 \
               and e.source_interface_id.endswith('_hold'):
                item_ent = self._entities.get(e.target_entity_id)
                if item_ent:
                    result[item_ent.name] = e.quantity
        return result

    # ─── 筛选 NPC 的边 ───

    def get_npc_outgoing_edges(self, npc_eid: str) -> list[InteractionEdge]:
        """获取某 NPC 为源的所有边（只含活跃边）"""
        return [e for e in self.graph.edges
                if e.source_entity_id == npc_eid and e.is_active]

    def get_edge_readable_descriptions(self, edges: list[InteractionEdge]) -> dict[str, str]:
        """构建边 ID → 可读描述的映射（用于 LLM prompt）"""
        result = {}
        for e in edges:
            src_name = "?"
            dst_name = "?"
            src_ent = self._entities.get(e.source_entity_id)
            dst_ent = self._entities.get(e.target_entity_id)
            if src_ent:
                src_name = src_ent.name
            if dst_ent:
                dst_name = dst_ent.name
            src_iface_name = e.source_interface_id.split('_')[-1] if '_' in e.source_interface_id else e.source_interface_id
            dst_iface_name = e.target_interface_id.split('_')[-1] if '_' in e.target_interface_id else e.target_interface_id
            result[e.edge_id] = f"{src_name}[{src_iface_name}] → {dst_name}[{dst_iface_name}]"
        return result

    # ─── LLM 响应解析 ───

    def parse_llm_response(self, text: str) -> list[dict]:
        """从 LLM 返回文本中提取交互指令列表"""
        text = text.strip()

        # 策略1：提取 ```json ... ``` / ``` ... ```
        if '```' in text:
            blocks = text.split('```')
            for i, block in enumerate(blocks):
                block = block.strip()
                if block.startswith('json'):
                    block = block[4:].strip()
                if block.startswith('[') or block.startswith('{'):
                    text = block
                    break

        # 策略2：提取 JSON 数组 [...] 或 JSON 对象 {...}
        stack = []
        start = -1
        for i, ch in enumerate(text):
            if ch == '[' or ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    if not stack and start >= 0:
                        text = text[start:i+1]
                        break
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    if not stack and start >= 0:
                        text = text[start:i+1]
                        break

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return []

    def _apply_effects(self, effects: list[dict]):
        """执行属性变更列表，不依赖边 ID"""
        for eff in effects:
            target_id = eff.get("target_entity_id", "")
            target = self._entities.get(target_id)
            if not target:
                continue
            attr_name = eff.get("attribute_name", "")
            op = eff.get("operation", "set")
            value = eff.get("value", 0)
            if op in ("add", "sub"):
                delta = float(value) if op == "add" else -float(value)
                target.modify_attr(attr_name, delta)
            else:
                target.set_attr(attr_name, value)

    def _apply_qty_changes(self, qty_changes: list[dict]):
        """执行库存变更列表，不依赖边 ID"""
        for qc in qty_changes:
            src = qc.get("source_entity_id", "")
            tgt = qc.get("target_entity_id", "")
            delta = qc.get("delta", 0)
            if src and tgt and delta != 0:
                self.modify_edge_quantity(src, tgt, delta)

    def ensure_entity_exists(self, entity_id: str, name: str,
                               entity_type: str = "item") -> str:
        """
        确保实体存在，如果不存在则通过实体最小化原则自动创建。
        返回最终的 entity_id。
        """
        # 先按 entity_id 找
        existing = self._entities.get(entity_id)
        if existing:
            return entity_id

        # 再按名字找
        for e in self._entities.values():
            if e.name == name and e.entity_type == entity_type:
                return e.entity_id

        # 通过实体最小化原则推导和注册
        from ..entities.derivation import EntityDerivationEngine
        engine = EntityDerivationEngine(self)
        return engine.derive_and_register(entity_id, name, entity_type)

    def execute_npc_instruction(self, instr: dict) -> str:
        """执行单条 NPC 指令，返回 result_text"""
        edge_id = instr.get("edge_id", "")
        edge = next((e for e in self.graph.edges if e.edge_id == edge_id), None)

        action = instr.get("action", "") or "交互中"
        result_text = instr.get("result_text", action)
        effects = instr.get("effects", [])
        qty_changes = instr.get("edge_qty_changes", [])

        # 1. 自动注册不存在的实体（effects 中的 target + qty_changes 中的 target）
        self._auto_register_missing(effects, qty_changes)

        # 2. 属性变更
        self._apply_effects(effects)

        # 3. 库存变更
        self._apply_qty_changes(qty_changes)

        if edge:
            edge.is_active = True
            edge.result_text = result_text
            edge.effects = effects
            return result_text
        else:
            return result_text or f"[无对应边] {edge_id}"

    def _auto_register_missing(self, effects: list[dict], qty_changes: list[dict]):
        """
        自动创建 LLM 引用但不存在的实体（使用实体最小化原则）。

        代替旧的推断类型逻辑，采用三步推导：
          1. 尝试从已有实体通过制造链推导
          2. 递归分解为基本实体
          3. 无法推导时注册为新基础实体
        """
        from ..entities.derivation import EntityDerivationEngine
        engine = EntityDerivationEngine(self)

        # Process effects
        for eff in effects:
            tid = eff.get("target_entity_id", "")
            if not tid or tid in self._entities:
                continue
            # Infer type from attribute name
            attr = eff.get("attribute_name", "")
            if attr in ("vitality", "hunger", "mood", "strength"):
                etype = "npc"
            elif attr in ("status", "state"):
                etype = "object"
            else:
                etype = "item"
            # Extract a readable name
            name = self._extract_name_from_eid(tid, eff.get("description", ""))
            engine.derive_and_register(tid, name, etype)

        # Process qty_changes
        for qc in qty_changes:
            for key in ("source_entity_id", "target_entity_id"):
                eid = qc.get(key, "")
                if not eid or eid in self._entities:
                    continue
                # Infer type from eid prefix
                if "npc_" in eid:
                    etype = "npc"
                elif "obj_" in eid:
                    etype = "object"
                else:
                    etype = "item"
                name = self._extract_name_from_eid(eid, "")
                engine.derive_and_register(eid, name, etype)

    @staticmethod
    def _extract_name_from_eid(entity_id: str, fallback: str) -> str:
        """Extract a readable entity name from an entity_id."""
        for prefix in ("item_", "obj_", "npc_", "zone_"):
            if entity_id.startswith(prefix):
                remaining = entity_id[len(prefix):]
                if remaining:
                    return remaining[:20]
        # If no prefix matched, maybe it's already a human-readable name
        if fallback:
            return fallback[:20]
        return entity_id[:20]

    def execute_effects(self, instructions: list[dict]) -> list[str]:
        """兼容旧接口：批量执行指令列表，每条调用 execute_npc_instruction"""
        return [self.execute_npc_instruction(instr) for instr in instructions]
