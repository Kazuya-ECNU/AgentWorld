"""
Intent Executor —— LLM #2 意图解析层 + 拓扑执行层

只做两件事：
  1. IntentResolver (LLM #2)：自然语言 → interact_with（目标类型/id）
  2. IntentExecutor：执行拓扑变更（区域移动、NPC/物体连接）
  
属性/库存/记忆更新已解耦到 PostProcessor (LLM #3) 处理。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger("intent_executor")


# ─── 常量 ───

_ZONE_NAMES = [
    "village_square", "farm", "market", "tavern",
    "barracks", "library", "temple", "forest",
]

_ITEM_ENTITY_MAP = {
    "小麦": "item_小麦", "金币": "item_金币", "铁锭": "item_铁锭",
    "药水": "item_药水", "蔬菜": "item_蔬菜", "面包": "item_面包",
    "皮毛": "item_皮毛", "草药": "item_草药", "纸张": "item_纸张",
    "货物": "item_货物", "武器": "item_武器", "书籍": "item_书籍",
    "酒": "item_酒", "佛经": "item_佛经", "面粉": "item_面粉",
    "工具": "item_工具", "衣物": "item_衣物", "家具": "item_家具",
    "木材": "item_木材",
}


# ─── 执行结果（传递给 LLM #3） ───

@dataclass
class ExecutionResult:
    """IntentExecutor 执行拓扑变更后的结果"""
    npc_eid: str
    npc_name: str
    npc_role: str
    zone_before: str
    zone_after: str
    zone_changed: bool = False
    interacted_npcs: list[str] = field(default_factory=list)
    interacted_objects: list[str] = field(default_factory=list)
    unreachable_targets: list[str] = field(default_factory=list)
    narrative: str = ""
    raw_intent: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── LLM #2 prompt ───

def _build_intent_prompt(
    npc_name: str,
    npc_role: str,
    current_zone: str,
    inventory: dict[str, int],
    raw_text: str,
    nearby_npcs: list[dict] | None = None,
) -> str:
    """构建 LLM #2 意图解析 prompt（只输出 interact_with）"""
    inv_str = "、".join(f"{k}x{v}" for k, v in inventory.items() if v > 0) if inventory else "空手"

    npc_list_str = ""
    if nearby_npcs:
        npc_list_str = "\n当前区域其他 NPC：" + "、".join(
            f"{zn['name']}（{zn['role']}）" for zn in nearby_npcs
        )

    return f"""你是世界模拟引擎的意图解析模块。将 NPC 的自然语言决策翻译为结构化的交互指令。

当前 NPC 状态：
- {npc_name}（{npc_role}）当前在 {current_zone} 区域
- 持有：{inv_str}
{npc_list_str}

NPC 的自然语言决策：
"{raw_text}"

可用区域：{', '.join(_ZONE_NAMES)}

请分析这段决策，输出严格 JSON（只输出 JSON，不要多余文字）：

{{
  "interact_with": [
    {{"type": "zone", "id": "market"}},
    {{"type": "npc", "id": "王老板"}}
  ],
  "narrative": "老张去market找王老板卖掉5小麦"
}}

规则：
1. interact_with：NPC 需要交互的对象——type 可选 zone（区域）、object（物体）、npc（其他角色）
2. **重要——交易必须指定 NPC**：如果 NPC 说要卖东西、买东西、交易、聊生意、找人合作，
   必须同时输出 type:zone（目标区）和 type:npc（交易对象）。
   只输出 zone 而不写 npc 会导致交易无法发生、失物不消失。
   例子：{{"type": "zone", "id": "market"}}, {{"type": "npc", "id": "王老板"}}
3. "前往X区" → type:zone, id:X，但前提是 NPC 当前不在该区
4. 仅休息/闲逛/无事 → interact_with 可为空数组
5. narrative：用人类可读的一句话总结发生了什么
"""


# ─── IntentResolver (LLM #2) ───

class IntentResolver:
    """
    LLM #2: 把 LLM #1 的自然语言决策解析为交互目标。
    只输出 interact_with，不涉及属性/库存变化。
    """

    def __init__(self, resolver=None):
        self._resolver = resolver

    def resolve_intent(
        self,
        npc_name: str,
        npc_role: str,
        current_zone: str,
        inventory: dict[str, int],
        raw_text: str,
        nearby_npcs: list[dict] | None = None,
    ) -> dict:
        """
        解析一条 NPC 的自然语言决策。

        Returns:
            { "interact_with": [...], "narrative": "..." }
            解析失败时返回兜底结构。
        """
        if not raw_text or not raw_text.strip():
            return {"interact_with": [], "narrative": ""}

        prompt = _build_intent_prompt(
            npc_name=npc_name,
            npc_role=npc_role,
            current_zone=current_zone,
            inventory=inventory,
            raw_text=raw_text,
            nearby_npcs=nearby_npcs,
        )

        if self._resolver:
            raw = self._resolver._call_llm(prompt)
        else:
            raw = self._rule_based_resolve(raw_text, current_zone)

        if not raw:
            return {"interact_with": [], "narrative": raw_text}

        intent = self._parse_response(raw)
        if intent and isinstance(intent, dict):
            logger.info(
                f"[Intent] {npc_name}: "
                f"interact={json.dumps(intent.get('interact_with',[]), ensure_ascii=False)}"
            )
            return intent

        return {"interact_with": [], "narrative": raw_text}

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

    def _rule_based_resolve(self, raw_text: str, current_zone: str) -> str:
        """兜底：无 LLM 时用简单规则"""
        text = raw_text.lower()
        target_zone = current_zone
        for z in _ZONE_NAMES:
            if z in text:
                target_zone = z
                break

        interact = []
        if target_zone and target_zone != current_zone:
            interact.append({"type": "zone", "id": target_zone})

        return json.dumps({
            "interact_with": interact,
            "narrative": raw_text,
        }, ensure_ascii=False)


# ─── IntentExecutor（只做拓扑） ───

class IntentExecutor:
    """
    执行交互意图——只做拓扑变更（区域移动、物体连接、NPC 连接）。
    属性/库存/记忆更新由 PostProcessor 处理。
    """

    def __init__(self, graph_engine):
        self._ge = graph_engine

    def execute(self, npc_eid: str, npc_name: str, npc_role: str, intent: dict, raw_intent: str = "") -> ExecutionResult:
        """
        执行意图（仅拓扑）。

        Args:
            npc_eid: NPC 实体 ID
            npc_name: NPC 显示名（如"老张"）
            npc_role: 角色（如 farmer）
            intent: { interact_with: [{type, id}], narrative: str }

        Returns:
            ExecutionResult（传递给 PostProcessor）
        """
        narrative = intent.get("narrative", "")
        npc_ent = self._ge.get_entity(npc_eid)
        if not npc_ent:
            return ExecutionResult(
                npc_eid=npc_eid, npc_name=npc_name, npc_role=npc_role,
                zone_before="?", zone_after="?", narrative=narrative, raw_intent=raw_intent,
            )

        zone_before = npc_ent.get_attr("zone_id") or "?"
        interacted_npcs = []
        interacted_objects = []
        unreachable_targets = []
        zone_chase_candidates = {}  # zone_name → [target_names]

        for target in intent.get("interact_with", []):
            ttype = target.get("type", "")
            tid = target.get("id", "")
            if ttype == "zone":
                self._handle_zone_interaction(npc_eid, tid)
            elif ttype == "object":
                if self._handle_object_interaction(npc_eid, tid):
                    interacted_objects.append(tid)
                else:
                    unreachable_targets.append(tid)
            elif ttype == "npc":
                success, tzone = self._handle_npc_interaction(npc_eid, tid)
                if success:
                    interacted_npcs.append(tid)
                elif tzone:
                    # 目标在不同区 → 记录以决定是否追过去
                    if tzone not in zone_chase_candidates:
                        zone_chase_candidates[tzone] = []
                    zone_chase_candidates[tzone].append(tid)
                    unreachable_targets.append(tid)
                else:
                    unreachable_targets.append(tid)

        # ─── 决策：要不要追去目标 NPC 的区域？ ───
        zone_after = npc_ent.get_attr("zone_id") or zone_before
        zone_changed = zone_before != zone_after

        if zone_chase_candidates and not zone_changed:
            chase_zone = self._decide_chase(
                npc_name, npc_role, npc_ent,
                zone_before, zone_chase_candidates,
            )
            if chase_zone:
                target_names = "、".join(zone_chase_candidates[chase_zone])
                logger.info(f"[Chase] {npc_name}: 决定追去 {chase_zone} 找{target_names}")
                self._handle_zone_interaction(npc_eid, chase_zone)
                zone_after = npc_ent.get_attr("zone_id") or zone_before
                zone_changed = zone_before != zone_after

        return ExecutionResult(
            npc_eid=npc_eid,
            npc_name=npc_name,
            npc_role=npc_role,
            zone_before=zone_before,
            zone_after=zone_after,
            zone_changed=zone_changed,
            interacted_npcs=interacted_npcs,
            interacted_objects=interacted_objects,
            unreachable_targets=unreachable_targets,
            narrative=narrative,
            raw_intent=raw_intent,
        )

    # ─── 追人决策 ───

    _PROACTIVE_TAGS = {"勤劳", "急性子", "热心", "积极", "勇敢", "好动", "主动"}
    _LAZY_TAGS = {"懒散", "怕麻烦", "随遇而安", "谨慎", "孤僻", "慢吞吞"}

    def _decide_chase(
        self,
        npc_name: str,
        npc_role: str,
        npc_ent: object,
        current_zone: str,
        zone_candidates: dict[str, list[str]],
    ) -> str | None:
        """
        根据性格+随机决定是否追去目标 NPC 的区域。

        返回要去的 zone_name，或 None 表示放弃。
        """
        import random

        # 1. 从 persona_tags 提取性格倾向
        tags = (npc_ent.get_attr("persona_tags") or "").lower() if hasattr(npc_ent, 'get_attr') else ""
        # graph_npc_engine 传 personality 的方式不一致，兜底
        is_proactive = any(t in tags for t in ["勤劳", "热心", "急性子", "积极", "勇敢"])
        is_lazy = any(t in tags for t in ["懒散", "怕麻烦", "谨慎"])

        role_proactive = npc_role in ("merchant", "trader", "messenger", "hunter")
        role_lazy = npc_role in ("scholar", "priest", "noble")

        # 2. 基础概率
        base_chance = 0.5  # 50% 保底
        if is_proactive or role_proactive:
            base_chance = 0.85
        if is_lazy or role_lazy:
            base_chance = 0.25

        # 3. 遍历区域候选项，对每个独立决策
        for zone_name, target_names in zone_candidates.items():
            if zone_name == current_zone:
                continue
            roll = random.random()
            if roll < base_chance:
                return zone_name
            else:
                logger.info(f"[Chase] {npc_name}: 放弃追随到 {zone_name}（roll={roll:.2f} < {base_chance}）")

        return None

    # ─── 拓扑处理 ───

    def _handle_zone_interaction(self, npc_eid: str, zone_name: str):
        """处理区域交互：更新 zone_id + 拓扑连接"""
        zone_eid = f"zone_{zone_name}"
        npc_ent = self._ge.get_entity(npc_eid)
        zone_ent = self._ge.get_entity(zone_eid)

        if not npc_ent or not zone_ent:
            return

        old_zone = npc_ent.get_attr("zone_id")
        if old_zone == zone_name:
            return

        npc_ent.set_attr("zone_id", zone_name)
        logger.info(f"[Zone] {npc_ent.name}: {old_zone} → {zone_name}")

        if zone_eid not in npc_ent.connected_entity_ids:
            self._ge.connect(npc_eid, zone_eid)

        for ent in self._ge.all_entities():
            if ent.entity_type == "object" and ent.get_attr("zone_id") == zone_name:
                oeid = ent.entity_id
                if oeid not in npc_ent.connected_entity_ids:
                    self._ge.connect(npc_eid, oeid)

    def _handle_object_interaction(self, npc_eid: str, object_type: str) -> bool:
        """处理物体交互"""
        npc_ent = self._ge.get_entity(npc_eid)
        if not npc_ent:
            return False

        current_zone = npc_ent.get_attr("zone_id")
        for ent in self._ge.all_entities():
            if ent.entity_type != "object":
                continue
            obj_type = ent.get_attr("object_type") or ent.name
            if object_type in obj_type or object_type == obj_type:
                if current_zone and ent.get_attr("zone_id") != current_zone:
                    continue
                oeid = ent.entity_id
                if oeid not in npc_ent.connected_entity_ids:
                    self._ge.connect(npc_eid, oeid)
                    logger.info(f"[Object] {npc_ent.name} ↔ {ent.name}")
                return True
        return False

    def _handle_npc_interaction(self, npc_eid: str, target_npc_name: str) -> tuple[bool, str | None]:
        """处理 NPC 间交互

        Returns:
            (True, None)         — 成功建立连接（双方同区）
            (False, None)        — 目标 NPC 不存在
            (False, target_zone) — 目标在不同区，返回 ta 所在的区域名
        """
        src_ent = self._ge.get_entity(npc_eid)
        if not src_ent:
            return False, None

        src_zone = src_ent.get_attr("zone_id")

        for ent in self._ge.all_entities():
            if ent.entity_type == "npc" and ent.name == target_npc_name:
                target_zone = ent.get_attr("zone_id")
                # 仅当双方在同一区域时才建立交互
                if src_zone and target_zone and src_zone != target_zone:
                    logger.info(f"[NPC] {src_ent.name}↔{ent.name}: 不同区({src_zone}≠{target_zone})")
                    return False, target_zone
                teid = ent.entity_id
                if teid not in src_ent.connected_entity_ids:
                    self._ge.connect(npc_eid, teid)
                    logger.info(f"[NPC] {src_ent.name} ↔ {ent.name}")
                return True, None
        return False, None
