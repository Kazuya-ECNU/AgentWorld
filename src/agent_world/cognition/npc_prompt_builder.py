"""
NPC Prompt Builder —— 为每个 NPC 独立构建推理上下文。

每个 NPC 的 prompt 包含：
  - 自身属性（体力/饥饿/心情/位置/角色）
  - 最近经历（记忆）
  - 性格标签
  - 库存
  - 图连接关系（从图中筛选出以该 NPC 为源的边）
  
输出：单个 NPC 的完整 prompt 字符串，供 LLM 独立推理该 NPC 下一步行动。
"""

from __future__ import annotations

from typing import Any
from ..models.interaction import InteractionEdge
from ..entities.base_entity import Entity


def build_one_npc_prompt(
    npc_entity: Entity,
    npc_name: str,
    npc_role: str,
    memories: list[dict],
    personality_tags: list[str],
    inventory: dict[str, int],
    zone_npcs: list[dict] | None = None,
) -> str:
    """
    为一个 NPC 构建完整的推理 prompt。

    LLM #1 只看到：自身信息、标签、记忆、自己有啥、在哪、同区域有谁。
    不暴露图结构细节（边/物体/配方），那都是 LLM #2 解析层判断的。

    Args:
        npc_entity: 图引擎中的 NPC Entity
        npc_name: NPC 显示名称（如"老张"）
        npc_role: 角色（如 farmer）
        memories: 最近记忆列表，每项含 event/timestamp 等
        personality_tags: 性格标签列表（如 ["勤劳", "务实"]）
        inventory: 库存 {物品名: 数量}
        zone_npcs: 同区域的其他 NPC（[{"name": "王老板", "role": "merchant"}]）

    Returns:
        格式化的 prompt 字符串
    """
    # ── 属性部分 ──
    vitality = npc_entity.get_attr("vitality") or 0
    hunger = npc_entity.get_attr("hunger") or 0
    mood = npc_entity.get_attr("mood") or 0
    zone_id = npc_entity.get_attr("zone_id") or "?"

    parts = [
        f"## NPC: {npc_name}",
        f"角色: {npc_role}  |  位置: {zone_id}",
        f"体力: {vitality:.0f}/100  |  饥饿: {hunger:.0f}/100  |  心情: {mood:.0f}/100",
        "",
    ]

    # ── 性格标签 ──
    if personality_tags:
        tags_str = "、".join(personality_tags)
        parts.append(f"### 性格标签\n{tags_str}\n")

    # ── 最近经历 ──
    if memories:
        parts.append("### 最近经历")
        for m in memories[:8]:  # 最多 8 条
            ts = m.get("timestamp", "")
            if hasattr(ts, 'strftime'):
                ts = ts.strftime("%H:%M")
            event = m.get("event", "")
            loc = m.get("location", "")
            if loc:
                parts.append(f"  [{ts}] 在{loc} {event}")
            else:
                parts.append(f"  [{ts}] {event}")
        parts.append("")

    # ── 库存 ──
    if inventory:
        inv_str = "、".join(f"{name}x{qty}" for name, qty in inventory.items() if qty > 0)
        parts.append(f"### 当前持有\n{inv_str}\n")
    else:
        parts.append("### 当前持有\n空手\n")

    # ── 同区域的其他 NPC ──
    if zone_npcs:
        other_str = "、".join(f"{zn['name']}（{zn['role']}）" for zn in zone_npcs)
        parts.append(f"### 当前区域还有\n{other_str}\n")

    # ── 决策指令 ──
    parts.append("### 决策")
    parts.append(f"请为 {npc_name} 决定下一步行动。")
    parts.append("用自然语言描述：你想做什么？在哪里做？有什么影响（体力/库存变化）？为什么？")
    parts.append("不用输出 JSON，不用关心格式，直接说你想干嘛。")
    parts.append("")
    parts.append(f"格式示例：我叫{npc_name}，我是{npc_role}。我目前在{zone_id}，持有小麦x21，体力7。")
    parts.append("我决定去market卖掉5小麦换金币，体力会消耗一些。")
    parts.append("用第一人称说人话就行。")

    return "\n".join(parts)


def build_one_fallback_prompt(
    npc_entity: Entity,
    npc_name: str,
    npc_role: str,
    zone_id: str,
) -> dict:
    """
    兜底模式的 NPC 决策（不用 LLM）。
    返回和 LLM 同样结构的 dict 供 execute_effects 使用。
    """
    neid = npc_entity.entity_id
    vitality = npc_entity.get_attr("vitality") or 50

    # 如果体力太低，休息
    if vitality < 20:
        return {
            "edge_id": "",
            "action": "休息",
            "effects": [
                {
                    "target_entity_id": neid,
                    "attribute_name": "vitality",
                    "operation": "add",
                    "value": 5,
                    "description": f"{npc_name}休息中恢复体力",
                },
                {
                    "target_entity_id": neid,
                    "attribute_name": "hunger",
                    "operation": "add",
                    "value": 2,
                    "description": f"{npc_name}休息时轻微饥饿",
                },
            ],
            "edge_qty_changes": [],
            "result_text": f"{npc_name}在{zone_id}休息恢复体力",
        }

    # 默认：在区域中闲逛，消耗一点体力
    return {
        "edge_id": "",
        "action": "闲逛",
        "effects": [
            {
                "target_entity_id": neid,
                "attribute_name": "vitality",
                "operation": "sub",
                "value": 3,
                "description": f"{npc_name}闲逛消耗体力",
            },
            {
                "target_entity_id": neid,
                "attribute_name": "hunger",
                "operation": "add",
                "value": 2,
                "description": "饥饿增加",
            },
        ],
        "edge_qty_changes": [],
        "result_text": f"{npc_name}在{zone_id}闲逛",
    }
