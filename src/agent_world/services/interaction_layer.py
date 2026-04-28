"""
Interaction Layer —— 边级交互结果处理器（LLM 驱动）

功能：
  1. 收集所有 NPC 的 ExecutionResult，按边去重
  2. 对每条唯一边调用 LLM 生成自由格式的自然语言故事描述
  3. 输出给 PostProcessor 做集中式批处理

避免分布式更新不一致：
  - 老张↔王老板的 trade 只作为一条边出现一次
  - PostProcessor 拿到所有边的故事描述后，一次产出全部更新
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("interaction_layer")

# 故事标记正则：匹配 【源↔目标】
_STORY_HEAD_RE = re.compile(r"【([^】]+?)↔([^】]+?)】")


# ─── 边级结果 ───

@dataclass
class EdgeResult:
    """一条交互边的结果（供 PostProcessor 消费）"""
    source: str
    target: str
    edge_type: str          # "npc_npc" | "npc_object" | "npc_zone"
    zone: str
    description: str        # LLM 生成的自然语言故事描述
    success: bool
    chase: bool = False
    items_involved: dict[str, int] = field(default_factory=dict)


# ─── 交互层 ───

class InteractionLayer:
    """
    边级交互处理层。
    LLM 驱动：给定 NPC 执行结果，自由生成每条边的故事。
    """

    def __init__(self, resolver=None):
        self._resolver = resolver

    def process(self, exec_results: list[dict]) -> list[EdgeResult]:
        """
        输入：list[ExecutionResult.to_dict()]
        输出：list[EdgeResult]（去重后的边级结果，description 为 LLM 生成的故事）
        """
        if not exec_results:
            return []

        edges = self._build_edges(exec_results)
        unique = self._deduplicate(edges)

        if self._resolver and unique:
            # LLM 驱动：生成故事描述
            story_text = self._ask_llm_for_story(unique, exec_results)
            self._assign_stories(unique, story_text, exec_results)
        else:
            # 降级：模板描述（走老逻辑）
            self._fallback_describe(unique, exec_results)

        return unique

    # ───── 公共方法 ─────

    def _build_edges(self, exec_results: list[dict]) -> list[EdgeResult]:
        """从所有执行结果中提取边"""
        results = []

        for er in exec_results:
            src = er["npc_name"]
            zone_after = er.get("zone_after", "?")
            zone_before = er.get("zone_before", "?")
            zone_changed = er.get("zone_changed", False)

            # 区域移动边
            if zone_changed:
                results.append(EdgeResult(
                    source=src,
                    target=zone_after,
                    edge_type="npc_zone",
                    zone=zone_after,
                    description="",
                    success=True,
                ))

            # NPC 交互边
            for target_name in er.get("interacted_npcs", []):
                results.append(EdgeResult(
                    source=src,
                    target=target_name,
                    edge_type="npc_npc",
                    zone=zone_after,
                    description="",
                    success=True,
                ))

            # 物体交互边
            for obj_name in er.get("interacted_objects", []):
                results.append(EdgeResult(
                    source=src,
                    target=obj_name,
                    edge_type="npc_object",
                    zone=zone_after,
                    description="",
                    success=True,
                ))

            # 不可达目标
            for target_name in er.get("unreachable_targets", []):
                chase = zone_changed
                results.append(EdgeResult(
                    source=src,
                    target=target_name,
                    edge_type="npc_npc",
                    zone=zone_after,
                    description="",
                    success=False,
                    chase=chase,
                ))

        return results

    def _deduplicate(self, edges: list[EdgeResult]) -> list[EdgeResult]:
        """
        去重：老张↔王老板 只保留一条。
        双向边（A↔B + B↔A）合并为 A↔B，success 取或。
        """
        sig_map: dict[str, EdgeResult] = {}

        def sig(a: str, b: str) -> str:
            return f"{min(a,b)}↔{max(a,b)}"

        for e in edges:
            if e.edge_type != "npc_npc":
                sig_map[f"{e.edge_type}:{e.source}→{e.target}"] = e
                continue
            key = sig(e.source, e.target)
            existing = sig_map.get(key)
            if not existing:
                sig_map[key] = e
            elif existing.source != e.source:
                # 双向边合并
                existing.success = existing.success or e.success
                existing.chase = existing.chase or e.chase
                for k, v in e.items_involved.items():
                    existing.items_involved[k] = existing.items_involved.get(k, 0) + v

        return list(sig_map.values())

    # ───── LLM 驱动 ─────

    def _ask_llm_for_story(self, edges: list[EdgeResult], exec_results: list[dict]) -> str:
        """调用 LLM 为所有边生成故事描述"""
        prompt = self._build_story_prompt(edges, exec_results)
        raw = self._resolver._call_llm(prompt)
        return raw or ""

    def _build_story_prompt(self, edges: list[EdgeResult], exec_results: list[dict]) -> str:
        """构建故事生成 prompt"""
        # NPC 状态索引
        npc_map: dict[str, dict] = {}
        for er in exec_results:
            name = er["npc_name"]
            npc_map[name] = {
                "role": er.get("npc_role", "?"),
                "zone": er.get("zone_after", er.get("zone_before", "?")),
                "raw_intent": er.get("raw_intent", ""),
            }

        # 构建 prompt 正文
        parts = [
            "你是世界模拟引擎的故事叙事层。",
            "你的任务：为本轮 NPC 之间的交互行为写出生动的故事描述。",
            "完全自由发挥，不要输出任何 JSON 或结构化格式，只写故事。",
            "",
            "==== 当前世界 ====",
        ]

        for name, info in npc_map.items():
            parts.append(f"- {name}（{info['role']}）@{info['zone']}")

        parts.append("")
        parts.append("==== 每个 NPC 本轮的想法和行动 ====")
        for er in exec_results:
            name = er["npc_name"]
            intent = er.get("raw_intent", "（无）")
            zone_changed = er.get("zone_changed", False)
            zone_before = er.get("zone_before", "?")
            zone_after = er.get("zone_after", "?")
            interacted = er.get("interacted_npcs", [])
            unreachable = er.get("unreachable_targets", [])

            lines = [f"\n### {name}"]
            lines.append(f"原本打算：{intent}")
            if zone_changed:
                lines.append(f"移动：{zone_before} → {zone_after}")
            if interacted:
                lines.append(f"找到：{'、'.join(interacted)}")
            if unreachable:
                chase = zone_changed
                if chase:
                    lines.append(f"追去{zone_after}找{'、'.join(unreachable)}")
                else:
                    lines.append(f"想找{'、'.join(unreachable)}但不在同区未动身")
            parts.append("\n".join(lines))

        parts.append("")
        parts.append("==== 需要故事描述的交互边 ====")
        for i, e in enumerate(edges, 1):
            status = "成功" if e.success else "未完成"
            if e.chase:
                status += "（追人）"
            parts.append(f"{i}. {e.source} ↔ {e.target}  [{e.edge_type}] 状态：{status}  区域：{e.zone}")

        parts.append("")
        parts.append(
            "请为以上每条交互边，写一段生动的故事描述。"
            "可以描写 NPC 的动作、对话、心理活动、天气、环境等，完全自由发挥。"
            "每条边上换行用 --- 分隔即可。"
            ""
            "**重要——每条边独立描述**：同 tick 内多条边代表同时发生的平行事件，不是时序先后。"
            "例如王老板同时跟老张、田嫂、赵酒师三人都做了交易是完全合理的。"
            "可以自由发挥，写成最终成交、继续讨价还价、或者约定下次都行。"
            "关键是每条边独立考虑，不要因为一条边在等就把另一条边也压着不写。"
            ""
            "### 格式硬性要求——每条故事必须在一行开头加 【源↔目标】 标记！！！"
            ""
            "例如："
            "【王老板↔田嫂】王老板在市场摊位上整理货物，田嫂扛着一袋蔬菜走进市场，两人开始讨价还价。"
            "【老张↔王老板】老张从酒馆来到市场找到王老板，商量着把存粮卖掉换些金币。"
            ""
            "每条故事的第一行必须以【X↔Y】开头，X 和 Y 就是上面列表中的"
            "交互边两端角色名，**必须与列表中的完全一致**。"
        )

        return "\n".join(parts)

    @staticmethod
    def _match_paragraph_to_label(
        para: str, edge_by_label: dict[str, EdgeResult]
    ) -> str | None:
        """检查一段文本是否以【X↔Y】开头，返回匹配的 label"""
        for label in edge_by_label:
            if para.startswith(f"【{label}】"):
                return label
        return None

    def _assign_stories(self, edges: list[EdgeResult], story_text: str, exec_results: list[dict]):
        """将 LLM 返回的故事文本按边分配"""
        if not story_text.strip():
            self._fallback_describe(edges, exec_results)
            return

        # 按 --- 分隔符拆分段落
        paragraphs = [p.strip() for p in story_text.split("---") if p.strip()]

        # 按标记匹配：故事必须以【源↔目标】开头
        edge_by_label: dict[str, EdgeResult] = {}
        for e in edges:
            label = f"{e.source}↔{e.target}"
            edge_by_label[label] = e

        assigned = set()
        for para in paragraphs:
            matched = self._match_paragraph_to_label(para, edge_by_label)
            if matched:
                # 如果段落内有多个故事标记，说明 --- 拆分不充分
                # 只取第一个标记对应的故事内容，剩下的让正则兜底处理
                next_match = _STORY_HEAD_RE.search(para, 1)  # 从第二个字符开始找
                if next_match:
                    # 段落包含多个故事 → 截取第一段，不标记已分配
                    edge_by_label[matched].description = para[:next_match.start()].strip()
                    assigned.add(matched)
                    logger.info(f"[IL] ⚠️  多故事段落，截取第一个: {matched}")
                else:
                    edge_by_label[matched].description = para
                    assigned.add(matched)
            else:
                logger.info(f"[IL] 无法匹配标记的故事段落（前60字）：{para[:60]}...")

        # 如果 --- 拆分覆盖不佳（<50%），从原始文本按 【】 标记直接提取
        if len(assigned) < len(edges) * 0.5:
            logger.info(
                f"[IL] --- 拆分覆盖不足 ({len(assigned)}/{len(edges)})，"
                "尝试从原始文本按标记提取"
            )
            for m in _STORY_HEAD_RE.finditer(story_text):
                src, tgt = m.group(1), m.group(2)
                label = f"{src}↔{tgt}"
                if label in edge_by_label and label not in assigned:
                    # 从标记开始到下一个标记或结尾
                    story_end = _STORY_HEAD_RE.search(story_text, m.end())
                    body = (
                        story_text[m.start():story_end.start()].strip()
                        if story_end
                        else story_text[m.start():].strip()
                    )
                    edge_by_label[label].description = body
                    assigned.add(label)
                    logger.info(f"[IL] ✅ 正则提取匹配: {label}")

        # 未匹配到的边用模板
        for label, e in edge_by_label.items():
            if label not in assigned:
                e.description = f"{e.source}与{e.target}发生了交互"
                logger.info(f"[IL] {label}: 无匹配故事，使用模板")

        logger.info(f"[IL] LLM 生成 {len(paragraphs)} 段故事，匹配到 {len(assigned)}/{len(edges)} 条边")

    # ───── 降级：模板描述 ─────

    def _fallback_describe(self, edges: list[EdgeResult], exec_results: list[dict]):
        """LLM 不可用时用回模板"""
        raw_intents = {er["npc_name"]: er.get("raw_intent", "") for er in exec_results}

        for e in edges:
            if e.edge_type == "npc_zone":
                e.description = f"{e.source}前往{e.target}"
            elif e.edge_type == "npc_npc":
                if e.success:
                    e.description = (
                        f"{e.source}和{e.target}在{e.zone}碰头，"
                        f"双方可以在此进行交易或社交。"
                    )
                elif e.chase:
                    e.description = (
                        f"{e.source}前往{e.zone}寻找{e.target}，"
                        f"但未能立刻见面"
                    )
                else:
                    e.description = (
                        f"{e.source}想找{e.target}但{e.target}不在同一个区域，"
                        f"也没有动身去找"
                    )
            elif e.edge_type == "npc_object":
                e.description = f"{e.source}在{e.zone}使用{e.target}"
