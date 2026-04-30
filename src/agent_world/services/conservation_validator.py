"""
Conservation Validator — 拓扑增量守恒校验

校验 LLM #4 输出的 delta 操作是否满足能量守恒：
  - 每个物品的 Σ(delta) ≈ 0（物品不凭空产生/消失）
  - 每个 NPC 的持有量 ≥ 0（不能透支）

和 GraphEngine 无关，只校验操作列表本身的守恒性。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .graph_engine import GraphEngine

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    PASS = "pass"
    SOFT_WARN = "soft_warn"
    HARD_FAIL = "hard_fail"


@dataclass
class ValidationOutcome:
    result: ValidationResult
    message: str = ""
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.result in (ValidationResult.PASS, ValidationResult.SOFT_WARN)


class ConservationValidator:
    """
    校验 delta 操作列表是否守恒。

    原理：
      - 收集所有 item 的 delta 操作
      - 按物品分组 Σ(delta) ≈ 0
      - 检查每个 NPC 的最终持有量 ≥ 0
    """

    def __init__(self, graph_engine: GraphEngine | None = None):
        self._graph = graph_engine

    # ═══════════════════════════════════════════
    # 公共 API
    # ═══════════════════════════════════════════

    def validate_deltas(self, ops: list[dict], epsilon: float = 0.001) -> ValidationOutcome:
        """
        校验一组 delta 操作。

        Args:
            ops: [{op: "delta", src, tgt, delta}, ...]
            epsilon: 浮点误差容限

        Returns:
            ValidationOutcome
        """
        if not ops:
            return ValidationOutcome(ValidationResult.PASS, "无操作，自动通过")

        # 只提取 delta 类型的操作
        deltas = [op for op in ops if op.get("op") == "delta"]
        if not deltas:
            return ValidationOutcome(ValidationResult.PASS, "无 delta 操作，自动通过")

        details: list[str] = []

        # 1. 按 target (item_eid) 分组 Σ(delta) = 0
        item_sums: dict[str, int] = {}
        for d in deltas:
            tgt = d.get("tgt", "")
            delta = d.get("delta", 0)
            item_sums[tgt] = item_sums.get(tgt, 0) + delta

        imbalances = {
            item: delta for item, delta in item_sums.items()
            if abs(delta) > epsilon
        }

        if imbalances:
            for item, delta in imbalances.items():
                details.append(f"  {item}: Σ = {delta:+d}（不守恒！）")
            fail_msg = "物品不守恒: " + "; ".join(
                f"{item}={delta:+d}" for item, delta in imbalances.items()
            )
            return ValidationOutcome(ValidationResult.HARD_FAIL, fail_msg, details)

        for item, delta in item_sums.items():
            if abs(delta) <= epsilon:
                details.append(f"  {item}: Σ = 0 ✅")

        # 2. 检查每个 NPC 的最终持有量 ≥ 0（如果有图引擎的话）
        if self._graph:
            for d in deltas:
                src = d.get("src", "")
                tgt = d.get("tgt", "")
                delta = d.get("delta", 0)
                if not src or not tgt:
                    continue
                ent = self._graph.get_entity(src)
                if not ent or ent.entity_type != "npc":
                    continue
                current = self._graph.get_held_quantity(src, tgt)
                if current < 0:
                    details.append(f"  {src} 持有 {tgt} = {current} < 0 ❌")
                    return ValidationOutcome(
                        ValidationResult.HARD_FAIL,
                        f"{src} 持有 {tgt} 为负 ({current})",
                        details,
                    )

        logger.debug(f"[Conservation] " + "\n".join(details))
        return ValidationOutcome(ValidationResult.PASS, "守恒校验通过 ✅", details)
