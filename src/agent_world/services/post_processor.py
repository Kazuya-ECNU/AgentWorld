"""
Post Processor —— LLM #4（拓扑数值增量层）

根据 LLM #3 的故事描述 + 当前拓扑，输出数值增量操作。

输入：
  - LLM #3 的故事文本（自然语言）
  - 当前拓扑视图（1-hop 子图）

输出：
  - 边数量增量：[{op: "delta", src, tgt, delta}]
  - 属性变更（可选）：[{op: "attr", target, attr, delta, description}]

设计原则：
  - LLM #4 从不直接写入数据
  - 输出纯结构化指令，由 GraphEngine.apply_edge_operations() 执行
  - 每条 delta 是独立的数值变化，系统保证守恒性
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .graph_engine import GraphEngine

logger = logging.getLogger("post_processor")


class PostProcessor:
    """
    LLM #4: 故事 + 拓扑 → 数值增量。

    根据 LLM #3 的故事描述以及当前拓扑状态，推理出：
    - 物品持有数量的变化（{op, src, tgt, delta}）
    - NPC 属性的变化（可选的 {op, target, attr, delta, description}）
    """

    def __init__(self, resolver=None):
        self._resolver = resolver  # InteractionResolver（复用 LLM 调用能力）

    def resolve_topology_deltas(
        self,
        npc_plans: dict[str, str],
        stories: list[str],
        graph_engine: GraphEngine,
        world_time_str: str | None = None,
        tick_duration_str: str | None = None,
    ) -> tuple[list[dict], dict[str, str]]:
        """
        主入口：故事 + 拓扑 → 数值增量操作 + 节点近况投影。

        Args:
            npc_plans: {NPC 名: LLM #1 的原始决策文本}
            stories: LLM #3 生成的故事描述列表
            graph_engine: 图引擎（用于获取当前拓扑视图）
            world_time_str: 当前世界时间
            tick_duration_str: 本 tick 时长

        Returns:
            (operations, recent_info_map)
            operations: [{op: "delta"|"attr", ...}]
            recent_info_map: {NPC/Zone 名: 近况摘要}
        """
        if not stories or not npc_plans:
            return [], {}

        # 构建 prompt
        prompt = self._build_prompt(
            npc_plans=npc_plans,
            stories=stories,
            graph_engine=graph_engine,
            world_time_str=world_time_str,
            tick_duration_str=tick_duration_str,
        )

        if not self._resolver:
            logger.warning("[LLM #4] 无 LLM resolver")
            return [], {}

        raw = self._resolver._call_llm(prompt)
        if not raw or not raw.strip():
            return [], {}

        ops, recent_info_map = self._parse_output(raw, graph_engine)
        return ops, recent_info_map

    # ─── Prompt 构建 ───

    def _build_prompt(
        self,
        npc_plans: dict[str, str],
        stories: list[str],
        graph_engine: GraphEngine,
        world_time_str: str | None = None,
        tick_duration_str: str | None = None,
    ) -> str:
        """
        构建 LLM #4 prompt。

        输入：每个 NPC 的当前状态 + 故事 + 拓扑子图
        输出：数值增量（delta 操作）
        """
        parts = [
            "你是一个世界模拟引擎的数值变化推理模块（LLM #4）。",
            "",
            "你的任务：根据 NPC 的计划、故事叙事以及当前拓扑状态，",
            "推理出本次 tick 的数值变化。",
            "",
        ]

        if world_time_str:
            parts.append(f"当前时间：{world_time_str}")
        if tick_duration_str:
            parts.append(f"本 tick 时长：{tick_duration_str}")
        if world_time_str or tick_duration_str:
            parts.append("")

        # NPC 状态
        parts.append("==== 当前 NPC 状态 ====")
        for npc_eid, plan in npc_plans.items():
            ent = graph_engine.get_entity(npc_eid)
            if ent:
                eid = ent.entity_id
                inv = graph_engine.get_inventory_view(eid)
                inv_str = "、".join(
                    f"{i['item_name']}x{i['quantity']}" for i in inv
                ) if inv else "空手"

                # 找 zone
                zone_name = "?"
                for conn in ent.connected_entity_ids:
                    e = graph_engine.get_entity(conn)
                    if e and e.entity_type == "zone":
                        zone_name = e.name
                        break

                parts.append(
                    f"- {ent.name}（{ent.role or '?'}）@{zone_name} | "
                    f"体力{ent.attributes.get('vitality', 100):.0f}/100 "
                    f"饱腹{ent.attributes.get('satiety', 50):.0f}/100 "
                    f"心情{ent.attributes.get('mood', 50):.0f}/100 | "
                    f"持有：{inv_str}"
                )
            else:
                parts.append(f"- {npc_eid}（？）")
        parts.append("")

        # NPC 计划
        parts.append("==== 每位 NPC 的本轮计划 ====")
        for npc_eid, plan in npc_plans.items():
            ent = graph_engine.get_entity(npc_eid)
            display_name = ent.name if ent else npc_eid
            parts.append(f"- {display_name}：{plan[:300]}")
        parts.append("")

        # 故事
        parts.append("==== 本轮故事叙事 ====")
        for i, story in enumerate(stories, 1):
            parts.append(f"--- 事件 {i} ---")
            parts.append(story)
        parts.append("")

        # 近况投影指令
        parts.append("==== 节点近况投影 ====")
        parts.append("""请根据本轮故事，为故事中涉及到的节点生成近况摘要。
近况会成为该节点下个 tick 的上下文。

规则：
- 每条 20-60 字，简洁但有信息量
- NPC 用第一人称（"我做了什么"），zone 写"这里发生了什么"
- 没有事件可以不写
- 不是 NPC 的节点也可以写（zone、item 等）
""")
        parts.append("")

        # 拓扑子图
        parts.append("==== 当前拓扑（库存+连接） ====")
        for npc_eid in npc_plans:
            ent = graph_engine.get_entity(npc_eid)
            if ent:
                sub = graph_engine.get_1hop_subgraph_text(npc_eid)
                parts.append(sub)
        parts.append("")

        # 实体 ID 映射（去重）
        parts.append("==== 实体 ID 映射（在 delta/attr 的 src/tgt/target 中可以用 <= 左侧的名称） ====")
        mapped_ids: set[str] = set()
        for npc_eid in npc_plans:
            ent = graph_engine.get_entity(npc_eid)
            if ent and ent.entity_id not in mapped_ids:
                parts.append(f"  {ent.name} → {ent.entity_id}")
                mapped_ids.add(ent.entity_id)
                for conn_eid in ent.connected_entity_ids:
                    ce = graph_engine.get_entity(conn_eid)
                    if ce and ce.entity_type == "item" and conn_eid not in mapped_ids:
                        iname = ce.name or conn_eid
                        parts.append(f"  {iname} → {conn_eid}")
                        mapped_ids.add(conn_eid)
        parts.append("")

        # 输出格式
        parts.append("==== 输出格式 ====")
        parts.append("""根据故事中的物品移动和属性变化，输出数值增量操作列表。

支持的操作类型：
1. delta: 修改边的数量（物品持有量的增减）
   {"op": "delta", "src": "老陈", "tgt": "item_金币", "delta": 10}
   src/tgt 可用角色名或物品名，系统会自动映射到实体 ID

2. set_qty: 设置边的数量（新建持有物品时初始化用）
   {"op": "set_qty", "src": "老陈", "tgt": "item_面包", "qty": 2}

3. attr: 修改 NPC 属性
   {"op": "attr", "target": "老陈", "attr": "vitality", "delta": -5, "description": "走路消耗体力"}
   attr 可选: vitality / satiety / mood

核心规则：
1. **每笔交易都是双边操作**：如果有人获得了物品，就有人失去了物品。
   用独立 delta 表示：A 获得 → (src=A, delta=+N)，B 失去 → (src=B, delta=-N)。
2. **delta 是独立的**：每笔操作 independently 增减。
   NPC 的最终持有量 = 初始值 + Σ(delta)。系统会自动处理跨 Tick 的叠加。
3. **delta 可以叠加**：A 获得卖小麦的10金币 + 支付买菜的5金币 →
   对 A：金币 delta:+10 和 金币 delta:-5 是两条独立操作。
4. **总和守恒**：每个物品的所有 delta 之和应为 0。
   例：老张-10金币 + 王老板+10金币 = 0。
5. **属性变化能量守恒**：体力恢复来自休息（+），消耗于活动（-）。
   交易成功→心情+，失败→心情-。饱腹只能通过吃东西增加（-食品 +饱腹）。
6. **物品名必须准确匹配**：物品名来自 NPC 持有的物品列表（如 金币、小麦、面包...）。
   不要编造物品名！
7. **没有交易就不写 delta**：如果故事只是聊天、讨价还价、约定等没有实际物品移动，
   或者双方都空手（没有物品可交易），对该 NPC 只输出 attr 操作，不写 delta。

输出格式（JSON 对象，两个字段）：
1. operations: 数值增量操作列表（规则同上）
2. recent_info: 节点近况投影
  {"实体名": "20-60字的近况摘要"}

示例：
{
  "operations": [
    {"op": "delta", "src": "老陈", "tgt": "item_金币", "delta": 10},
    {"op": "attr", "target": "刘猎户", "attr": "vitality", "delta": -10}
  ],
  "recent_info": {
    "老陈": "在market和王老板讨价还价，10金币没谈拢",
    "王老板": "想低价收老陈的小麦他嫌价高没成",
    "market": "老陈和王老板在讨价还价"
  }
}

注意：
- recent_info 中每个实体一条，不限于 NPC
- 没有事件可以不写 recent_info
- 只输出 JSON，不要多余文字，不要 markdown 代码块。""")

        return "\n".join(parts)

    # ─── 解析 ───

    def _parse_output(self, raw: str, graph_engine: GraphEngine) -> tuple[list[dict], dict[str, str]]:
        """
        解析 LLM #4 双输出。

        兼容旧格式：纯 JSON 数组 → operations only, recent_info={}
        新格式：JSON 对象 → {operations, recent_info}

        Returns:
            (operations, recent_info_map)
            recent_info_map: {实体 ID: 近况文本}
        """
        raw = raw.strip()
        json_str = self._extract_json(raw)

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"[LLM #4] JSON 解析失败: {raw[:200]}")
            return [], {}

        recent_info_map: dict[str, str] = {}

        if isinstance(parsed, list):
            # 兼容旧格式：纯数组 → operations only
            ops = parsed
        elif isinstance(parsed, dict):
            ops = parsed.get("operations", [])
            raw_ri = parsed.get("recent_info", {})
            if isinstance(raw_ri, dict):
                for name, text in raw_ri.items():
                    if text and isinstance(text, str) and text.strip():
                        ent = graph_engine.find_entity_by_name(name)
                        if ent:
                            recent_info_map[ent.entity_id] = text.strip()
        else:
            return [], {}

        # 验证并修复实体 ID
        valid_ops = []
        for op in ops:
            op_type = op.get("op", "")
            if op_type == "delta":
                src = op.get("src", "")
                tgt = op.get("tgt", "")
                delta = op.get("delta", 0)
                if not src or not tgt or delta == 0:
                    continue
                # 如果 LLM 输出了名字而不是 ID，尝试查找
                real_src = self._resolve_name(src, graph_engine)
                real_tgt = self._resolve_name(tgt, graph_engine)
                if real_src and real_tgt:
                    valid_ops.append({"op": "delta", "src": real_src, "tgt": real_tgt, "delta": delta})

            elif op_type == "set_qty":
                src = op.get("src", "")
                tgt = op.get("tgt", "")
                qty = op.get("qty", 0)
                if not src or not tgt:
                    continue
                real_src = self._resolve_name(src, graph_engine)
                real_tgt = self._resolve_name(tgt, graph_engine)
                if real_src and real_tgt:
                    valid_ops.append({"op": "set_qty", "src": real_src, "tgt": real_tgt, "qty": qty})

            elif op_type == "attr":
                target = op.get("target", "")
                attr = op.get("attr", "")
                delta = op.get("delta", 0)
                desc = op.get("description", "")
                if not target or not attr or delta == 0:
                    continue
                real_target = self._resolve_name(target, graph_engine)
                if real_target:
                    valid_ops.append({"op": "attr", "target": real_target, "attr": attr, "delta": delta, "description": desc})

        logger.info(f"[LLM #4] 解析到 {len(valid_ops)}/{len(ops)} 有效操作, {len(recent_info_map)} 条近况")
        return valid_ops, recent_info_map

    def _resolve_name(self, name_or_id: str, graph_engine: GraphEngine) -> str | None:
        """将 NPC 名/物品名 解析为 entity_id"""
        # 已经是 entity_id 格式（如 npc_2ca464dd / item_2282968f）
        if name_or_id.startswith("npc_") or name_or_id.startswith("item_"):
            if graph_engine.get_entity(name_or_id):
                return name_or_id
            # 可能是 LLM 编的 item_皮毛（带 prefix 的名字），剥离 prefix 再查
            _, _, bare_name = name_or_id.partition("_")
            if bare_name:
                ent = graph_engine.find_entity_by_name(bare_name)
                if ent:
                    return ent.entity_id
                # 再尝试 fuzzy match（皮毛 → 皮毛, 皮毛→, 皮毛）
                for eid, e in graph_engine._entities.items():
                    if e.name and bare_name in e.name:
                        return eid
        # 尝试按名称查找
        ent = graph_engine.find_entity_by_name(name_or_id)
        if ent:
            return ent.entity_id
        # fuzzy match 兜底
        for eid, e in graph_engine._entities.items():
            if e.name and name_or_id in e.name:
                return eid
        return None

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        if '```' in text:
            blocks = text.split('```')
            for block in blocks:
                block = block.strip()
                if block.startswith('json'):
                    block = block[4:].strip()
                if block.startswith('[') or block.startswith('{'):
                    return block
        # 根据文本首字符决定提取类型
        stripped = text.lstrip()
        if stripped.startswith('{'):
            # JSON 对象 — 括号匹配（避免 {} 内含的 [] 被错误匹配）
            stack = []
            start = -1
            for i, ch in enumerate(stripped):
                if ch == '{':
                    if not stack:
                        start = i
                    stack.append(ch)
                elif ch == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack and start >= 0:
                            return stripped[start:i + 1]
        elif stripped.startswith('['):
            # JSON 数组 — 括号匹配
            stack = []
            start = -1
            for i, ch in enumerate(stripped):
                if ch == '[':
                    if not stack:
                        start = i
                    stack.append(ch)
                elif ch == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()
                        if not stack and start >= 0:
                            return stripped[start:i + 1]
        return text
