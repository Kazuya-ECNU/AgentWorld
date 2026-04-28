"""
Post Processor —— LLM #3 后处理层

根据 LLM #1 的原始决策 + LLM #2 的执行结果，生成完整的数据更新：
- 属性变化（vitality/satiety/mood），包涵交互双方的 NPC
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
        current_state: 当前状态 {vitality, satiety, mood, inventory}
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
    unreachable = execution_result.get("unreachable_targets", [])
    if unreachable:
        exec_lines.append(f"  目标不可达：{', '.join(unreachable)}（不在同一区域）")
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
- 饱腹：{current_state.get('satiety', 50):.0f}/100
- 心情：{current_state.get('mood', 50):.0f}/100
- 持有：{inv_str}

⚠️  属性警戒值（请根据数值自行判断后果）：
  - 体力 < 30：极度疲劳，必须休息
  - 饱腹 < 30：极度饥饿，必须进食，降到 0 会饿死
  - 心情 < 20：心情极差，需要社交或娱乐缓解

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
        {{"item_name": "金币", "action": "add", "quantity": 10, "type": "transfer", "from_npc": "王老板"}}
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
2. attribute_changes：attr 可选 vitality/satiety/mood，op 可选 add/sub/set。体力消耗要合理（去市场 -5~10，巡逻 -3~5，聊天几乎不消耗）。
3. inventory_changes：每个 inventory_changes 元素包含：
   - item_name：物品名
   - action："add" 获得 / "remove" 消耗 / "consume" 消耗不影响库存
   - quantity：数量
   - type：变更类型
     * "transfer"：物品在 NPC 间转移 — **只输出 add 条目 + from_npc**，remove 由系统自动生成
     * "craft"：配方制造（消耗原料产出成品）— 需 recipe 字段标明配方名
     * "consume"：消耗物品影响自身属性
     * "gather"：从环境采集获得

   例如老张卖5小麦给王老板（transfer，单 NPC 视角）：
   - 老张：{{"item_name": "金币", "action": "add", "quantity": 10, "type": "transfer", "from_npc": "王老板"}}
   （老张收到金币，出让小麦给系统自动生成；王老板的条目在 batch 中由另一个 update 包含）

   **注意**：同一 NPC 可以有多条同一物品的边，每笔交易独立列出即可。
   例如老张卖小麦收 80 金币又买菜花 20 金币：
   - (老张, 金币, +80)
   - (老张, 金币, -20)
   这是两笔独立交易，完全合法。关键是要确保（源、目标、数量及方向）定义明确。
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

    # 第一步：自动补全交易对称性
    _auto_symmetry(updates, logger)

    # 第二步：收集所有NPC的当前库存用于推理
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
            if gei:
                if item_eid not in gei.connected_entity_ids:
                    gei.connect_to(item_eid)
                    item_ent = graph_engine.get_entity(item_eid)
                    if item_ent:
                        item_ent.connect_to(eid)
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
            if gei:
                if item_eid not in gei.connected_entity_ids:
                    gei.connect_to(item_eid)
                    item_ent = graph_engine.get_entity(item_eid)
                    if item_ent:
                        item_ent.connect_to(eid)
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
            current_state: {vitality, satiety, mood, inventory}
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

            # 目标不可达 → 决定：追了还是放弃了？
            unreachable = execution_result.get("unreachable_targets", [])
            zone_changed = execution_result.get("zone_changed", False)
            if unreachable:
                for u in updates:
                    if u.get("npc_name") == npc_name:
                        has_inv = bool(u.get("inventory_changes"))
                        u["inventory_changes"] = []
                        if has_inv:
                            target_list = "、".join(unreachable)
                            inv_desc = "、".join(
                                f"{ic.get('item_name','?')}{'-' if ic.get('action') in ('remove','consume') else '+'}{ic.get('quantity',0)}"
                                for ic in u.get("inventory_changes", [])
                            )
                            if zone_changed:
                                # NPC 决定追过去了 → 积极记忆
                                zone_after = execution_result.get("zone_after", "?")
                                mem = f"去{zone_after}找{target_list}准备交易({inv_desc})，还没碰上面"
                                imp = 0.5
                                log_msg = f"追去{zone_after}找{target_list}({inv_desc})"
                            else:
                                # NPC 放弃了 → 失败记忆
                                mem = f"想找{target_list}交易({inv_desc})但不在同区域，决定算了"
                                imp = 0.6
                                log_msg = f"{target_list}不可达({inv_desc})，放弃"
                            existing_mems = u.get("memories", [])
                            existing_mems.append({"event": mem, "importance": imp})
                            u["memories"] = existing_mems
                            logger.info(f"[PostP] {npc_name}: {log_msg}")

            logger.info(
                f"[PostP] {npc_name}: 生成了 {len(updates)} 个更新"
            )
            return updates

        logger.warning(f"[PostP] {npc_name}: 解析失败，原始响应={raw[:100]}")
        return []

    def process_batch(
        self,
        npc_states: list[dict],
        stories: list[str],
        edge_summaries: list[dict] | None = None,
    ) -> list[dict]:
        """
        集中式批处理。一次 LLM 调用，产出全部更新。

        Args:
            npc_states: [{name, role, zone, vitality, satiety, mood, inventory}]
            stories: InteractionLayer 生成的故事描述列表
            edge_summaries: [{source, target, success, chase}] 用于校验（可选）

        Returns:
            [{npc_name, attribute_changes, inventory_changes, memories, relationships}]
        """
        prompt = _build_batch_post_prompt(npc_states, stories)

        if not self._resolver:
            logger.warning("[PostP Batch] 无 LLM resolver")
            return []

        raw = self._resolver._call_llm(prompt)
        if not raw or not raw.strip():
            return []

        parsed = self._parse_response(raw)
        if parsed and isinstance(parsed, dict):
            updates = parsed.get("updates", [])

            # 清洗：过滤 quantity=0 的库存条目（LLM #4 可能输出金币+0、小麦+0 等噪音）
            for u in updates:
                ics = u.get("inventory_changes", [])
                clean_ics = [ic for ic in ics if ic.get("quantity", 0) != 0]
                if len(clean_ics) != len(ics):
                    logger.info(f"[PostP Batch] {u.get('npc_name','')}: 过滤掉 {len(ics)-len(clean_ics)} 条 quantity=0 条目")
                    u["inventory_changes"] = clean_ics

            # 后校验：如果某个 NPC 参与的交互是未完成的，清库存变化
            if edge_summaries:
                for u in updates:
                    npc_name = u.get("npc_name", "")
                    inv_changes = u.get("inventory_changes", [])
                    if not inv_changes:
                        continue
                    for es in edge_summaries:
                        if es.get("source") == npc_name or es.get("target") == npc_name:
                            if not es.get("success"):
                                u["inventory_changes"] = []
                                mem = u.get("memories", [])
                                partner = es.get("target") if es.get("source") == npc_name else es.get("source")
                                # 格式化库存变化为可读文本
                                change_parts = [
                                    f"{ic.get('item_name','?')}{'-' if ic.get('action') in ('remove','consume') else '+'}{ic.get('quantity',0)}"
                                    for ic in inv_changes
                                ]
                                change_desc = "、".join(change_parts) if change_parts else "（未知）"
                                mem.append({
                                    "event": f"想找{partner}交易{change_desc}但{partner}不在同区或已离开，交易未完成",
                                    "importance": 0.5,
                                })
                                u["memories"] = mem
                                logger.info(
                                    f"[PostP Batch] {npc_name}: {partner}交互未完成，清库存"
                                )

            # 自动对称：根据 to_npc/from_npc 补全对手方库存变化
            _auto_symmetry(updates, logger)

            # 二次清洗：auto_symmetry 可能清除了方向错误条目但残留了 quantity=0
            for u in updates:
                ics = u.get("inventory_changes", [])
                u["inventory_changes"] = [ic for ic in ics if ic.get("quantity", 0) != 0]

            logger.info(f"[PostP Batch] 生成了 {len(updates)} 个更新")
            return updates

        logger.warning(f"[PostP Batch] 解析失败")
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


def _auto_symmetry(updates: list[dict], logger=None):
    """
    根据 transfer 项的 to_npc/from_npc 字段自动补全对手方库存变化。
    
    原则：每条 transfer 边带来一条对偶边，不检查是否重复。
   重复注入由 _apply_updates 的 op_key dedup 处理，方向错误由 Validator 检测。
   不修改、不删除伙伴已有的任何条目。
    """
    partner_additions: dict[str, list[dict]] = {}
    for up in updates:
        name = up.get("npc_name", "")
        for inv_c in up.get("inventory_changes", []):
            if inv_c.get("type") != "transfer":
                continue
            to_npc = inv_c.get("to_npc", "")
            from_npc = inv_c.get("from_npc", "")
            if not to_npc and not from_npc:
                continue
            partner = to_npc or from_npc
            item = inv_c.get("item_name", "")
            action = inv_c.get("action", "")
            qty = inv_c.get("quantity", 0)
            partner_action = "add" if action in ("remove", "consume") else "remove"

            # 每条 transfer 边独立生成对偶边，不分伙伴是否已在 updates 中。
            # _apply_updates 通过 op_key=("name", "inv", item, delta) 自动去重，
            # 重复注入的结果一样。方向错误由 Validator 检测 Σ≠0。
            partner_up = next((u for u in updates if u.get("npc_name") == partner), None)
            if partner_up:
                partner_up["inventory_changes"].append({
                    "item_name": item,
                    "action": partner_action,
                    "quantity": qty,
                    "type": "transfer",
                })
                if logger:
                    logger.info(f"[AutoSym] {partner}: 注入{partner_action}{qty} {item}（来自{name}的to_npc/from_npc）")
            else:
                if partner not in partner_additions:
                    partner_additions[partner] = []
                partner_additions[partner].append({
                    "item_name": item,
                    "action": partner_action,
                    "quantity": qty,
                    "type": "transfer",
                })
                if logger:
                    logger.info(f"[AutoSym] {partner}: 自动补全{partner_action}{qty} {item}（来自{name}的to_npc/from_npc）")

    for partner_name, extra_changes in partner_additions.items():
        updates.append({
            "npc_name": partner_name,
            "attribute_changes": [],
            "inventory_changes": extra_changes,
            "memories": [],
            "relationships": {},
        })


# ─── Batch PostProcessor Prompt ───

def _build_batch_post_prompt(
    npc_states: list[dict],
    stories: list[str],
) -> str:
    """
    构建集中式 PostProcessor prompt。

    Args:
        npc_states: [{name, role, zone, vitality, satiety, mood, inventory}]
        stories: LLM #3 生成的故事描述列表，每条对应一次交互
    """
    parts = ["你是一个世界模拟引擎的后处理模块。"]
    parts.append("根据 NPC 的当前状态和故事叙事层的描述，推理出数据层面的变化。")
    parts.append("")
    parts.append("==== 当前所有 NPC 状态 ====")
    parts.append("")
    parts.append("⚠️  属性警戒值（请根据 NPC 当前数值自行判断）：")
    parts.append("  - 体力 < 30：极度疲劳，若不休息会导致行动力下降甚至无法行动")
    parts.append("  - 饱腹 < 30：极度饥饿，若不进食降到 0 会饿死")
    parts.append("  - 心情 < 20：心情极差，若不缓解可能导致抑郁")

    for s in npc_states:
        inv_str = "、".join(
            f"{k}x{v}" for k, v in s.get("inventory", {}).items() if v > 0
        ) or "空手"
        parts.append(
            f"- {s['name']}（{s.get('role','?')}）@{s.get('zone','?')} | "
            f"体力{s.get('vitality',100):.0f}/100 饱腹{s.get('satiety',50):.0f}/100 "
            f"心情{s.get('mood',50):.0f}/100 | 持有：{inv_str}"
        )

    parts.append("")
    parts.append("==== 故事叙事层描述的本轮事件 ====")
    if stories:
        for i, story in enumerate(stories, 1):
            parts.append(f"---") if i > 1 else None
            parts.append(story)
    else:
        parts.append("（无交互事件）")

    parts.append("")
    parts.append("==== 输出格式 ====")
    parts.append("""请输出严格 JSON (只输出 JSON，不要多余文字)：

{
  "updates": [
    {
      "npc_name": "老张",
      "attribute_changes": [
        {"attr": "vitality", "op": "sub", "value": 8, "description": "前往market消耗体力"}
      ],
      "inventory_changes": [
        {"item_name": "金币", "action": "add", "quantity": 10, "type": "transfer", "from_npc": "王老板"}
      ],
      "memories": [
        {"event": "在market卖了5小麦给王老板，换了10金币", "importance": 0.6}
      ],
      "relationships": {
        "王老板": 3
      }
    },
    {
      "npc_name": "王老板",
      "attribute_changes": [
        {"attr": "vitality", "op": "sub", "value": 3, "description": "摊位上接待客人"}
      ],
      "inventory_changes": [
        {"item_name": "小麦", "action": "add", "quantity": 5, "type": "transfer", "from_npc": "老张"}
      ],
      "memories": [
        {"event": "收到了老张送来的5袋新小麦，付了10个金币", "importance": 0.5}
      ]
    }
  ]
}

规则：
1. 为**每一个 NPC** 都生成一条 update，即使什么都没做也要有属性恢复/自然消耗。
2. attribute_changes: attr 可选 vitality/satiety/mood, op 可选 add/sub。
3. inventory_changes type 必须是以下之一：transfer(交易) / craft(制造) / consume(消耗) / gather(采集)。**不允许其他值**（如 earnings、found、collected 都不是合法类型）。
4. **transfer 只输出接收方（add + from_npc）**：
   对于 NPC 之间发生的物品转移，你只需要输出每个 NPC **获得了什么**。
   每条 add 类型的 transfer 条目必须标注 `from_npc` 指明来源 NPC。
   **不需要输出 remove/to_npc 条目**——出让方条目由系统自动补全。

   【黄金规则】每个交易故事必须输出 **N×2 条 add+from_npc 条目**（N=交换物品种类数）：
   - 每条物品的接收方各一条 add
   - 每笔交易的钱和货都不能漏

   例子：老张卖5小麦给王老板换了10金币
   正确输出：
     NPC 老张:
       inventory_changes:
         - {item_name: 金币, action: add, quantity: 10, type: transfer, from_npc: 王老板}
     NPC 王老板:
       inventory_changes:
         - {item_name: 小麦, action: add, quantity: 5, type: transfer, from_npc: 老张}
   系统自动补全：老张小麦-5, 王老板金币-10

   ⚠️ **输出前自查清单**:
   - 故事中有没有物品交换？如果有 → 找出所有流动的物品（金币、货物）
   - 对每种流动的物品：接收方是谁？→ 输出一条 add+from_npc
   - 总共应该有 N×2 条 add 条目（货币+实物各一方向）？实际输出了几条？
   - 如果只输出了金币 add 而没有相应货物 add，交易在系统中会断裂！
4b. **必须标注 from_npc，不要用 to_npc**：
   每个 transfer 类型的 add 条目都必须标注 `from_npc`。
   `to_npc` 字段已弃用（出让方由系统自动生成，不需要你输出）。
   如果 NPC 从环境获取物品（如采集草药），type 应为 gather 且不需要 from_npc。
5. 交易物品数量从故事中推理。如果故事说了"十袋小麦换了20个金币"，
   则老张金币+10 from_npc:王老板，王老板小麦+5 from_npc:老张。
6. 如果故事描述 NPC 未能完成交互（如"没找到人""错过了"），则不要生成库存变化。
7. memories 用第一人称自然语言。
8. 体力消耗要合理（区域移动-5~10，聊天~0，休息+5~20）。
9. **严禁编造物品名称**：inventory_changes 中的 item_name 必须来自 NPC 当前库存中已有的物品，
   或者是当前区域可采集的常见物品。不要自己编造（如"井水""铜板"等不存在的物品）。
   如果你不确定物品名称是否正确，就不要写 inventory_changes。
10. **没有实物交易就不写 inventory_changes**：如果故事只描述了聊天、讨价还价、约定改日交易等未实际发生物品交换的情节，
    给该 NPC 的 inventory_changes 设为空数组 []。不要在没有任何物品交换发生时假装有交易。
    只有当故事明确写了"递给""掏出""支付""接过""收下""数了钱"等实际交货动作时，才生成 inventory_changes。

==== 常见错误案例（必须避免） ====
以下是系统历史运行中发现的真实错误，**不要再犯同样错误**：

❌ 错误0：输出 remove 类型的 transfer 条目
  故事：老张卖5小麦给王老板，收了10金币
  输出：老张 **小麦-5** [transfer]  ← 错了！不应该输出出让方条目
  正确：老张 金币+10 from_npc:王老板 | 王老板 小麦+5 from_npc:老张
  （出让方条目由系统自动补全，不需要你在 LLM 输出中写）

❌ 错误1：漏写 from_npc
  故事：老张卖5小麦给王老板，收10金币
  输出：老张 金币+10 [transfer]  ← 缺 from_npc
  正确：老张 金币+10 [transfer from_npc:王老板]
  老张是从王老板那里获得金币的，必须标注来源。

❌ 错误2：把交易标记成采集
  故事：老张把5小麦卖给王老板
  输出：王老板 小麦+5 [**gather**]  ← 收来的东西不能标gather
  正确：王老板 小麦+5 [transfer from_npc:老张]
  （gather只用于从系统外获得新物品，比如从鱼塘钓鱼、从菜园摘菜）

❌ 错误3：交易只写一方
  故事：赵酒师给田嫂1坛米酒，田嫂给了1干香菇
  输出：赵酒师 米酒+1 from_npc:田嫂 | 田嫂：(无变化)  ← 田嫂收到了吗？
  正确：赵酒师 干香菇+1 from_npc:田嫂 | 田嫂 米酒+1 from_npc:赵酒师
  双方都写各自的 add 条目。

❌ 错误3b：**只写了金币方向，漏写了货物方向（高频错误!）**
  故事：老张卖小麦给王老板，收了8金币
  输出：
     老张：金币+8 from_npc:王老板
     王老板：(无变化)  ← 这是错的！王老板付了金币却没收到小麦
  正确：
     老张：金币+8 from_npc:王老板  ← 老张收到金币
     王老板：小麦+？ from_npc:老张  ← 王老板收到小麦
  每笔交易中：**每个NPC从对方那里获得了什么，就输出一条 add+from_npc**。
  **自查**：故事里"老张卖小麦给王老板" => 两样东西在移动：金币→老张, 小麦→王老板 => 需要2条add

❌ 错误4：编造不存在的 type 字段
  输出：type: "consume1" / type: "earnings" / type: "found" / type: "collected"
  正确：type 只能是 transfer / craft / consume / gather 四个值之一，没有例外

❌ 错误5：编造不存在的物品名称
  输出：老张 "井水"+1  ← 这些物品不在系统定义中
  正确：item_name 必须来自 NPC 库存（如小麦、金币、白菜、米酒）或系统已知的常见资源""")

    return "\n".join(parts)


# ─── 公共入口 ───

def apply_updates(
    updates: list[dict],
    npc_name_map: dict[str, object],
    graph_engine,
) -> list[str]:
    """便捷入口"""
    return _apply_updates(updates, npc_name_map, graph_engine)
