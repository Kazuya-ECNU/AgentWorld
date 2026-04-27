"""
Post Processor —— LLM #3 后处理层

根据 LLM #1 的原始决策 + LLM #2 的执行结果，生成完整的数据更新：
- 属性变化（vitality/hunger/mood），包涵交互双方的 NPC
- 库存变化（trade 自动对称）
- 记忆事件
- 关系变化

流水线位置：
  LLM #1 (决策) → LLM #2 (解析交互目标) → IntentExecutor (拓扑执行)
      → **LLM #3 PostProcessor (数据更新)** → 写入图引擎和 NPC 模型
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("post_processor")


def _build_post_prompt(
    npc_name: str,
    npc_role: str,
    zone_id: str,
    raw_decision: str,
    execution_result: dict,
    current_state: dict,
    nearby_npcs: list[dict] | None = None,
) -> str:
    """
    构建 LLM #3 后处理 prompt。

    Args:
        npc_name: NPC 名字
        npc_role: 角色（farmer/merchant 等）
        zone_id: 当前所在区域
        raw_decision: LLM #1 的自然语言决策
        execution_result: IntentExecutor 的执行结果（zone_changed, interacted_npcs等）
        current_state: 当前状态 {vitality, hunger, mood, inventory}
        nearby_npcs: 同区域其他 NPC [{name, role}]
    """
    inv_str = "、".join(
        f"{k}x{v}" for k, v in current_state.get("inventory", {}).items() if v > 0
    ) if current_state.get("inventory") else "空手"

    # 执行摘要
    exec_lines = []
    if execution_result.get("zone_changed"):
        exec_lines.append(f"  移动：{execution_result['zone_before']} → {execution_result['zone_after']}")
    if execution_result.get("interacted_npcs"):
        exec_lines.append(f"  交互对象：{', '.join(execution_result['interacted_npcs'])}")
    if execution_result.get("interacted_objects"):
        exec_lines.append(f"  使用物体：{', '.join(execution_result['interacted_objects'])}")
    if not exec_lines:
        exec_lines.append("  无拓扑变化（原地休息/闲逛）")
    exec_summary = "\n".join(exec_lines)

    # 同区其他人
    nearby_str = ""
    if nearby_npcs:
        nearby_str = "同区域 NPC：\n" + "\n".join(
            f"  - {zn['name']}（{zn['role']}）" for zn in nearby_npcs
        ) + "\n"

    return f"""你是世界模拟引擎的后处理模块。根据 NPC 的决策和实际执行结果，生成完整的状态更新。

## NPC 基础信息
- 名字：{npc_name}
- 角色：{npc_role}
- 当前位置：{zone_id}

## 当前属性
- 体力：{current_state.get('vitality', 100):.0f}/100
- 饥饿：{current_state.get('hunger', 50):.0f}/100
- 心情：{current_state.get('mood', 50):.0f}/100
- 持有：{inv_str}

## 原始决策（LLM #1 的意图）
"{raw_decision}"

## 实际执行结果
{exec_summary}

{nearby_str}请根据以上信息，推理本次行动的完整后果，输出严格 JSON（只输出 JSON，不要多余文字）：

{{
  "updates": [
    {{
      "npc_name": "{npc_name}",
      "attribute_changes": [
        {{"attr": "vitality", "op": "sub", "value": 10, "description": "前往market消耗体力"}},
        {{"attr": "mood", "op": "add", "value": 5, "description": "成功卖小麦心情不错"}}
      ],
      "inventory_changes": [
        {{"item_name": "小麦", "action": "remove", "quantity": 5, "type": "transfer"}},
        {{"item_name": "金币", "action": "add", "quantity": 10, "type": "transfer"}}
      ],
      "memories": [
        {{"event": "在market卖了5小麦给王老板，换了10金币", "importance": 0.7}}
      ],
      "relationships": {{
        "王老板": 3
      }}
    }}
  ]
}}

规则：
1. updates 数组包含本次行动影响到的**所有 NPC**。如果 NPC 和其他人交互，必须为双方都生成 update。
2. attribute_changes：attr 可选 vitality/hunger/mood，op 可选 add/sub/set。体力消耗要合理（去市场 -5~10，巡逻 -3~5，聊天几乎不消耗）。
3. inventory_changes：每个 inventory_changes 元素包含：
   - item_name：物品名
   - action："add" 获得 / "remove" 消耗
   - quantity：数量
   - type：变更类型
     * "transfer"：物品在 NPC 间转移（trade/give）— 必须双方对称
     * "craft"：配方制造（消耗原料产出成品）— 需 recipe 字段标明配方名
     * "consume"：消耗物品影响自身属性
     * "gather"：从环境采集获得

   例如老张卖5小麦给王老板（transfer）：
   - 老张：{{"item_name": "小麦", "action": "remove", "quantity": 5, "type": "transfer"}}
   - 王老板：{{"item_name": "小麦", "action": "add", "quantity": 5, "type": "transfer"}}
4. memories：用自然语言描述这个 NPC 视角下发生了什么。importance 是 0-1 的加权值（0.3=日常，0.5=普通，0.8=重要事件）。
5. relationships：对象 NPC 的名字→关系值变化（-100~+100）。
6. 如果 NPC 什么都没做（休息/无事），attribute_changes 保留空数组，可以增加少量体力恢复记忆。
"""


def _npc_name_to_eid(name: str, graph_engine) -> str | None:
    """通过 NPC 显示名查找实体 ID"""
    for ent in graph_engine.all_entities():
        if ent.entity_type == "npc" and ent.name == name:
            return ent.entity_id
    return None


# 全局已应用操作集合，用于去重
_applied_ops: set[tuple] = set()


def _apply_updates(
    updates: list[dict],
    npc_name_map: dict[str, object],
    graph_engine,
    interact_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    将 PostProcessor 输出的 updates 应用到图引擎和 NPC 模型。

    所有 PostProcessor 的输出汇总后批量调用此函数。
    去重策略：跟踪每个 (npc_name, category, key, delta) 组合是否已应用过。
    即使 老张 和 王老板 的 PP 都生成了同一笔交易，也不会重复执行。

    推理兜底：如果 NPC A 的 inventory_changes 有物品减少/金币增加，
    但交互对手 B 没有对应的反向变更，则自动补全。
    这解决了 LLM 有时忘记输出交易对手侧的问题。

    Args:
        updates: [{npc_name, attribute_changes, inventory_changes, memories, relationships}]
        npc_name_map: {npc_name: npc_model}
        graph_engine: GraphEngine 实例
        interact_map: {npc_name: [interacted_npc_names]} 用于推理交易对手

    Returns:
        操作日志列表
    """
    logs = []

    # 第一步：收集所有NPC的当前库存用于推理
    before_inv: dict[str, dict[str, int]] = {}
    for name in npc_name_map:
        eid = _npc_name_to_eid(name, graph_engine)
        if eid:
            before_inv[name] = dict(graph_engine.get_inventory_view(eid))
        else:
            before_inv[name] = {}

    # 第二步：应用所有 PP 更新
    for up in updates:
        name = up.get("npc_name", "")
        npc_model = npc_name_map.get(name)
        eid = _npc_name_to_eid(name, graph_engine)
        if not eid:
            logs.append(f"  ⚠️ {name}: 找不到实体")
            continue

        # 属性变化
        for attr_c in up.get("attribute_changes", []):
            attr = attr_c.get("attr", "")
            op = attr_c.get("op", "set")
            val = attr_c.get("value", 0)
            desc = attr_c.get("description", "")

            op_key = (name, "attr", attr, op, val)
            if op_key in _applied_ops:
                continue
            _applied_ops.add(op_key)

            ent = graph_engine.get_entity(eid)
            if not ent:
                continue
            current = ent.get_attr(attr) or 0
            if op == "add":
                new_val = min(100.0, max(0.0, current + val))
            elif op == "sub":
                new_val = min(100.0, max(0.0, current - val))
            else:
                new_val = min(100.0, max(0.0, float(val)))
            ent.set_attr(attr, new_val)
            logs.append(f"  ✅ {name}: {attr} {op} {val} -> {new_val:.0f} ({desc})")

        # 库存变化
        # 库存变化（支持新旧两种格式）
        for inv_c in up.get("inventory_changes", []):
            # 新格式: item_name + action + quantity
            # 旧格式: item + delta
            item = inv_c.get("item_name", "") or inv_c.get("item", "")
            if not item:
                continue

            # 新格式: action + quantity; 旧格式: delta
            new_action = inv_c.get("action", "")
            new_qty = inv_c.get("quantity", 0)
            old_delta = inv_c.get("delta", 0)
            if new_action:
                delta = new_qty if new_action == "add" else -new_qty
            else:
                delta = old_delta

            item_eid = f"item_{item}"
            op_key = (name, "inv", item, delta)
            if op_key in _applied_ops:
                continue
            _applied_ops.add(op_key)

            gei = graph_engine.get_entity(eid)
            action_text = "获得" if delta > 0 else "消耗"
            if gei and item_eid in gei.connected_entity_ids:
                graph_engine.modify_edge_quantity(eid, item_eid, delta)
                logs.append(f"  ✅ {name}: {action_text}{abs(delta)} {item}")
            else:
                logs.append(f"  ⚠️ {name} 无法交易 {item}: 未连接")

        # 记忆
        npc = npc_model
        if npc and hasattr(npc, 'add_memory'):
            for mem in up.get("memories", []):
                event = mem.get("event", "")
                importance = mem.get("importance", 0.5)
                op_key = (name, "mem", event[:60], round(importance, 1))
                if op_key in _applied_ops:
                    continue
                _applied_ops.add(op_key)
                npc.add_memory(event, importance=importance, location="")
                logs.append(f"  🧠 {name}: 记忆「{event[:40]}...」")

        # 关系
        if npc and hasattr(npc, 'update_relationship'):
            for rel_npc, delta in up.get("relationships", {}).items():
                op_key = (name, "rel", rel_npc, delta)
                if op_key in _applied_ops:
                    continue
                _applied_ops.add(op_key)
                npc.update_relationship(rel_npc, delta)
                logs.append(f"  💞 {name}->{rel_npc}: {delta:+d}")

    # 第三步：推理兜底——检查交易对称性
    if interact_map:
        after_inv: dict[str, dict[str, int]] = {}
        for name in npc_name_map:
            eid = _npc_name_to_eid(name, graph_engine)
            if eid:
                after_inv[name] = dict(graph_engine.get_inventory_view(eid))

        for name in npc_name_map:
            partners = interact_map.get(name, [])
            if not partners:
                continue

            # 看这个 NPC 的库存变化
            for item in set(list(before_inv[name].keys()) + list(after_inv[name].keys())):
                delta = after_inv[name].get(item, 0) - before_inv[name].get(item, 0)
                if delta == 0:
                    continue

                # 如果这个 NPC 有变化，看对手有没有对应的反向变化
                for partner in partners:
                    partner_delta = after_inv.get(partner, {}).get(item, 0) - before_inv.get(partner, {}).get(item, 0)

                    if delta > 0:
                        # NPC 获得了 item，对手应该消耗
                        expected_partner_delta = -delta
                        if partner_delta != expected_partner_delta and partner_delta == 0:
                            # 对手没变化，补上！
                            partner_eid = _npc_name_to_eid(partner, graph_engine)
                            if partner_eid:
                                item_eid = f"item_{item}"
                                op_key = (partner, "inv_inferred", item, expected_partner_delta)
                                if op_key not in _applied_ops:
                                    _applied_ops.add(op_key)
                                    graph_engine.modify_edge_quantity(partner_eid, item_eid, expected_partner_delta)
                                    logs.append(f"  🔄 {partner}: 补充{abs(expected_partner_delta)} {item}（交易自动对称）")
                    else:
                        # NPC 消耗了 item，对手应该获得
                        expected_partner_delta = -delta
                        if partner_delta != expected_partner_delta and partner_delta == 0:
                            partner_eid = _npc_name_to_eid(partner, graph_engine)
                            if partner_eid:
                                item_eid = f"item_{item}"
                                op_key = (partner, "inv_inferred", item, expected_partner_delta)
                                if op_key not in _applied_ops:
                                    _applied_ops.add(op_key)
                                    graph_engine.modify_edge_quantity(partner_eid, item_eid, expected_partner_delta)
                                    logs.append(f"  🔄 {partner}: 补充{abs(expected_partner_delta)} {item}（交易自动对称）")

    return logs
    logs = []
    for up in updates:
        name = up.get("npc_name", "")
        npc_model = npc_name_map.get(name)
        eid = _npc_name_to_eid(name, graph_engine)
        if not eid:
            logs.append(f"  ⚠️ {name}: 找不到实体")
            continue

        # 属性变化
        for attr_c in up.get("attribute_changes", []):
            attr = attr_c.get("attr", "")
            op = attr_c.get("op", "set")
            val = attr_c.get("value", 0)
            desc = attr_c.get("description", "")

            op_key = (name, "attr", attr, op, val)
            if op_key in _applied_ops:
                continue
            _applied_ops.add(op_key)

            ent = graph_engine.get_entity(eid)
            if not ent:
                continue
            current = ent.get_attr(attr) or 0
            if op == "add":
                new_val = min(100.0, max(0.0, current + val))
            elif op == "sub":
                new_val = min(100.0, max(0.0, current - val))
            else:
                new_val = min(100.0, max(0.0, float(val)))
            ent.set_attr(attr, new_val)
            logs.append(f"  ✅ {name}: {attr} {op} {val} → {new_val:.0f} ({desc})")

        # 库存变化（支持新旧两种格式）
        for inv_c in up.get("inventory_changes", []):
            item = inv_c.get("item_name", "") or inv_c.get("item", "")
            if not item:
                continue
            new_action = inv_c.get("action", "")
            new_qty = inv_c.get("quantity", 0)
            old_delta = inv_c.get("delta", 0)
            if new_action:
                delta = new_qty if new_action == "add" else -new_qty
            else:
                delta = old_delta
            item_eid = f"item_{item}"
            op_key = (name, "inv", item, delta)
            if op_key in _applied_ops:
                continue
            _applied_ops.add(op_key)
            gei = graph_engine.get_entity(eid)
            action_text = "获得" if delta > 0 else "消耗"
            if gei and item_eid in gei.connected_entity_ids:
                graph_engine.modify_edge_quantity(eid, item_eid, delta)
                logs.append(f"  ✅ {name}: {action_text}{abs(delta)} {item}")
            else:
                logs.append(f"  ⚠️ {name} 无法交易 {item}: 未连接")

        # 记忆
        npc = npc_model
        if npc and hasattr(npc, 'add_memory'):
            for mem in up.get("memories", []):
                event = mem.get("event", "")
                importance = mem.get("importance", 0.5)
                op_key = (name, "mem", event[:60], round(importance, 1))
                if op_key in _applied_ops:
                    continue
                _applied_ops.add(op_key)
                npc.add_memory(event, importance=importance, location=up.get("zone_id", ""))
                logs.append(f"  🧠 {name}: 记忆「{event[:40]}...」")

        # 关系
        if npc and hasattr(npc, 'update_relationship'):
            for rel_npc, delta in up.get("relationships", {}).items():
                op_key = (name, "rel", rel_npc, delta)
                if op_key in _applied_ops:
                    continue
                _applied_ops.add(op_key)
                npc.update_relationship(rel_npc, delta)
                logs.append(f"  💞 {name}→{rel_npc}: {delta:+d}")

    return logs


def reset_applied_ops():
    """重置去重集合（tick 之间调用）"""
    _applied_ops.clear()


class PostProcessor:
    """
    LLM #3: 根据执行结果生成数据更新。

    被 graph_npc_engine 调用，传入每个 NPC 的执行结果。
    """

    def __init__(self, resolver=None):
        self._resolver = resolver  # 复用 InteractionResolver 的 LLM 调用能力

    def process(
        self,
        npc_name: str,
        npc_role: str,
        zone_id: str,
        raw_decision: str,
        execution_result: dict,
        current_state: dict,
        nearby_npcs: list[dict] | None = None,
    ) -> list[dict]:
        """
        处理一个 NPC 的执行结果，生成数据更新。

        Args:
            npc_name: NPC 名字
            npc_role: 角色
            zone_id: 当前区域
            raw_decision: LLM #1 的原始自然语言决策
            execution_result: ExecutionResult.to_dict()
            current_state: {vitality, hunger, mood, inventory}
            nearby_npcs: 同区 NPC [{name, role}]

        Returns:
            [{npc_name, attribute_changes, inventory_changes, memories, relationships}]
            失败时返回空列表。
        """
        prompt = _build_post_prompt(
            npc_name=npc_name,
            npc_role=npc_role,
            zone_id=zone_id,
            raw_decision=raw_decision,
            execution_result=execution_result,
            current_state=current_state,
            nearby_npcs=nearby_npcs,
        )

        if self._resolver:
            raw = self._resolver._call_llm(prompt)
        else:
            logger.warning("[PostP] 无 LLM resolver，使用规则兜底")
            return []

        if not raw or not raw.strip():
            return []

        parsed = self._parse_response(raw)
        if parsed and isinstance(parsed, dict):
            updates = parsed.get("updates", [])
            logger.info(
                f"[PostP] {npc_name}: 生成了 {len(updates)} 个更新"
            )
            return updates

        logger.warning(f"[PostP] {npc_name}: 解析失败，原始响应={raw[:100]}")
        return []

    def _parse_response(self, text: str) -> dict | None:
        """提取 JSON 对象"""
        text = text.strip()
        if '```' in text:
            blocks = text.split('```')
            for block in blocks:
                block = block.strip()
                if block.startswith('json'):
                    block = block[4:].strip()
                if block.startswith('{'):
                    text = block
                    break

        stack = []
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    if not stack and start >= 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
        return None


def apply_updates(
    updates: list[dict],
    npc_name_map: dict[str, object],
    graph_engine,
) -> list[str]:
    """便捷入口"""
    return _apply_updates(updates, npc_name_map, graph_engine)
