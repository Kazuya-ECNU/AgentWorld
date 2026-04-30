"""
Intent Resolver + Executor（LLM #2）

拓扑结构变更层：将 LLM #1（NPC 计划）→ 拓扑结构变更。

输入：每个 NPC 的自然语言计划 + 当前拓扑子图
输出：[{op, src, tgt, qty}] — 仅限结构变更（connect/disconnect/set_qty）

设计原则：
  - 只改变拓扑结构（连接/断开节点之间的边）
  - 不改变数据值（数值修正是 LLM #4 的职责）
  - 每个 NPC 独立推理

支持的 op:
  - connect:    建立边（NPC→Zone 移动、NPC→NPC 建交等）
  - disconnect: 移除边（离开区域、断开连接等）
  - set_qty:    设置边的数量（首次交互时设置初始状态）
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any

from .graph_engine import GraphEngine

logger = logging.getLogger("intent_executor")


class IntentResolver:
    """
    LLM #2：将 NPC 计划 → 拓扑结构变更。

    在旧的架构中 IntentResolver 输出交互目标（action/location/target），
    现在输出结构化拓扑变更操作。
    """

    def __init__(self, graph_engine: GraphEngine, resolver=None):
        self._graph = graph_engine
        self._resolver = resolver  # InteractionResolver（用于调用 LLM）

    # ─── 主入口 ───

    def resolve_all_intents(
        self, npc_plans: dict[str, str]
    ) -> list[EdgeOperation]:
        """
        为所有有计划的 NPC 解析拓扑结构变更。

        Args:
            npc_plans: {npc_eid: 自然语言计划}

        Returns:
            list[EdgeOperation]: [{op, src, tgt, qty}, ...]
        """
        if not npc_plans:
            return []

        # 为每个 NPC 构建 prompt
        items = []
        for npc_eid, plan in npc_plans.items():
            prompt = self._build_prompt(npc_eid, plan)
            items.append((npc_eid, prompt))

        if not items:
            return []

        # 调用 LLM（批量推理）
        prompt_text = self._build_combined_prompt(items)
        raw = self._call_llm(prompt_text) if self._resolver else ""
        if not raw:
            return self._fallback(items)

        # 解析输出的操作列表
        return self._parse_ops(raw)

    # ─── Prompt 构建 ───

    def _build_prompt(self, npc_eid: str, plan: str) -> str:
        """
        为单个 NPC 构建 prompt。

        输入：NPC 的当前拓扑子图 + 自然语言计划
        输出：该 NPC 应该执行的结构变更操作
        """
        ent = self._graph.get_entity(npc_eid)
        if not ent:
            return ""

        # 拓扑视图
        subgraph = self._graph.get_1hop_subgraph_text(npc_eid)

        # NPC 自身描述
        self_desc = ent.to_prompt_block()

        # 库存
        inventory = self._graph.get_inventory_view(npc_eid)
        inv_str = "、".join(
            f"{i['item_name']}x{i['quantity']}" for i in inventory
        ) if inventory else "（空）"

        # Zone 信息
        zone_name = "?"
        for conn in ent.connected_entity_ids:
            e = self._graph.get_entity(conn)
            if e and e.entity_type == "zone":
                zone_name = e.name
                break

        # 同区域的其他 NPC（纯拓扑数据）
        same_zone_npcs = []
        for conn in ent.connected_entity_ids:
            e = self._graph.get_entity(conn)
            if e and e.entity_type == "zone" and e.name == zone_name:
                for other in self._graph.all_entities():
                    if other.entity_type == "npc" and other.entity_id != npc_eid \
                       and other.is_connected_to(conn):
                        same_zone_npcs.append(other.name)
                break

        # 在子图中追加同区域 NPC 信息
        if same_zone_npcs:
            subgraph += f"\n### 同区域中的其他实体\n"
            subgraph += f"  {', '.join(same_zone_npcs)} 也连接着 {zone_name}\n"

        return (
            f"你是 NPC {ent.name}，位于 {zone_name}。\n"
            f"持有：{inv_str}\n"
            f"自我描述：\n{self_desc}\n\n"
            f"当前拓扑视图：\n{subgraph}\n\n"
            f"你的计划：{plan}\n\n"
            f"=== 指令 ===\n"
            f"输出你需要执行的拓扑操作来执行此计划。\n"
            f"你在 {zone_name}，只能操作你所在区域的实体。\n\n"
            f"格式（JSON 数组）：\n"
            f'[{{"op":"connect","src":"{npc_eid}","tgt":"zone_南集市","qty":0}}]\n\n'
            f"可用的 op：\n"
            f'  "connect"    — 建立连接（NPC→Zone 或 NPC→NPC）\n'
            f'  "disconnect" — 断开连接（NPC→Zone 或 NPC→NPC）\n\n'
            f"重要规则：\n"
            f"1. 你只负责拓扑结构，不改数值。物品持有变更由 LLM #4 处理。\n"
            f"2. 区域连接用 qty=-1，NPC↔NPC 连接用 qty=0。\n"
            f"3. 输出纯 JSON，不要多余文字，不要 markdown。"
        )

    def _build_combined_prompt(self, items: list[tuple[str, str]]) -> str:
        """合并多个 NPC 的 prompt"""
        parts = [
            "你是一个世界模拟引擎的拓扑结构变更模块（LLM #2）。",
            "你的任务：根据每个 NPC 的自然语言计划，输出拓扑结构变更操作。",
            "",
            f"共 {len(items)} 个 NPC。",
            "",
        ]



        for i, (eid, prompt) in enumerate(items):
            parts.append(f"==== NPC {i+1}: {eid} ====")
            parts.append(prompt)
            parts.append("")

        parts.append("==== 最终输出格式 ====")
        parts.append("输出一个 JSON 对象，key 为 NPC entity_id，value 为操作数组：")
        parts.append("""{
  "npc_abc123": [
    {"op": "connect", "src": "npc_abc123", "tgt": "zone_南集市", "qty": 0},
    {"op": "disconnect", "src": "npc_abc123", "tgt": "zone_酒馆"}
  ],
  "npc_def456": [...]
}""")
        parts.append("不要多余文字，不要 markdown 代码块。")

        return "\n".join(parts)

    # ─── 解析 ───

    def _parse_ops(self, raw: str) -> list[EdgeOperation]:
        """
        解析 LLM 输出的 JSON。
        返回扁平的操作列表。
        """
        raw = raw.strip()
        # 提取 JSON
        json_str = _extract_json(raw)

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"[LLM #2] JSON 解析失败，原始输出: {raw[:200]}")
            return []

        # 扁平化
        ops: list[EdgeOperation] = []
        if isinstance(parsed, dict):
            for npc_eid, npc_ops in parsed.items():
                if isinstance(npc_ops, list):
                    for op in npc_ops:
                        if isinstance(op, dict):
                            op["src"] = op.get("src", npc_eid)
                            ops.append(op)

        if isinstance(parsed, list):
            ops = [op for op in parsed if isinstance(op, dict)]

        # 验证
        valid_ops = []
        for op in ops:
            if op.get("op") in ("connect", "disconnect", "set_qty"):
                if op.get("src") and op.get("tgt"):
                    valid_ops.append(op)

        logger.info(f"[LLM #2] 解析到 {len(valid_ops)}/{len(ops)} 有效操作")
        return valid_ops

    # ─── 降级 ───

    def _fallback(self, items: list[tuple[str, str]]) -> list[EdgeOperation]:
        """LLM 不可用时，根据计划关键词做简单推理"""
        ops = []
        for eid, plan in items:
            ent = self._graph.get_entity(eid)
            if not ent:
                continue
            # 检测移动意图
            for conn in ent.connected_entity_ids:
                e = self._graph.get_entity(conn)
                if e and e.entity_type == "zone":
                    current_zone = e.name
                    break
            else:
                current_zone = ""
            # 关键词检测
            zone_moves = re.findall(r"去(\w+)", plan)
            for z in zone_moves:
                target_zone = f"zone_{z}"
                if current_zone and target_zone != f"zone_{current_zone}":
                    ops.append({"op": "disconnect", "src": eid, "tgt": f"zone_{current_zone}"})
                    ops.append({"op": "connect", "src": eid, "tgt": target_zone})
        return ops

    # ─── LLM 调用 ───

    def _call_llm(self, prompt: str) -> str:
        if not self._resolver:
            return ""
        return self._resolver._call_llm(prompt)


# ─── Op 类型 ───

class EdgeOperation:
    """拓扑操作类型（运行时类型检查用）"""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    SET_QTY = "set_qty"
    DELTA = "delta"


# ─── 辅助 ───

def _extract_json(text: str) -> str:
    """从文本中提取 JSON"""
    # 尝试提取 ```json ... ```
    if '```' in text:
        blocks = text.split('```')
        for block in blocks:
            block = block.strip()
            if block.startswith('json'):
                block = block[4:].strip()
            if block.startswith('{') or block.startswith('['):
                return block

    # 尝试提取 JSON 对象
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
                    return text[start:i + 1]

    # 尝试提取 JSON 数组
    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '[':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
                if not stack and start >= 0:
                    return text[start:i + 1]

    return text
