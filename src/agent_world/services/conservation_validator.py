"""
Energy Conservation Validator — Post-LLM Gate

Validates that PostProcessor (LLM #3) output satisfies the energy conservation
principle **by interaction type**, not by numeric sum.

The validator checks:
  - transfer:   Σ(delta) per item_name = 0  (same item moves between NPCs)
  - craft:      matches known recipe ratios   (inputs → outputs)
  - consume:    always passes                  (item → attribute)
  - gather:     always passes                  (environment boundary input)
  - unknown:    HARD VIOLATION                 (rollback + alarm)

Key design: the PostProcessor MUST declare the 'type' for each inventory
change. The validator does NOT infer or guess.

See DESIGN_PHILOSOPHY.md ("Energy Conservation Principle") for the full theory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Result Types ───

class ValidationResult(Enum):
    PASS = "pass"
    SOFT_WARN = "soft_warn"   # 通过了但值得记录（如未知配方，可能是LLM发现新配方）
    HARD_FAIL = "hard_fail"   # 不守恒，需要回滚


@dataclass
class ValidationOutcome:
    """Validator 的返回结果。"""

    result: ValidationResult
    message: str = ""
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.result in (ValidationResult.PASS, ValidationResult.SOFT_WARN)


# ─── Validator ───

class ConservationValidator:
    """校验 PostProcessor 更新是否满足能量守恒（基于类型，而非数值）。"""

    def __init__(self, recipe_registry=None):
        self._registry = recipe_registry

    # ═══════════════════════════════════════════
    # 公共 API
    # ═══════════════════════════════════════════

    def validate(self, updates: list[dict], epsilon: float = 0.001) -> ValidationOutcome:
        """校验一组 PostProcessor 输出。

        校验流程：
          1. 按 type 分组 inventory_changes
          2. 每组调用对应的校验逻辑
          3. 合并结果：有一个 HARD_FAIL 整体返回 HARD_FAIL

        Args:
            updates: PostProcessor 输出的 updates[] 数组
            epsilon: transfer 类型检查的浮点误差

        Returns:
            ValidationOutcome
        """
        if not updates:
            return ValidationOutcome(ValidationResult.PASS, "无更新，自动通过")

        # 按 type 分组
        groups: dict[str, list[dict]] = {
            "transfer": [],
            "craft": [],
            "consume": [],
            "gather": [],
            "unknown": [],
        }

        for update in updates:
            for ic in update.get("inventory_changes", []):
                t = ic.get("type", "unknown")
                if t not in groups:
                    t = "unknown"
                groups[t].append(ic)

        # 依次校验每种类型
        all_details: list[str] = []
        worst = ValidationResult.PASS

        # 1. 未知类型 → 硬失败
        if groups["unknown"]:
            items = [
                f"{ic['item_name']} {ic['action']}{ic.get('quantity', '?')}"
                for ic in groups["unknown"]
            ]
            worst = ValidationResult.HARD_FAIL
            all_details.append(f"未知变更类型: {', '.join(items)}")

        # 2. 校验 transfer
        if groups["transfer"]:
            outcome = self._validate_transfer(groups["transfer"], epsilon)
            all_details.extend(outcome.details)
            if outcome.result.value != "pass":
                worst = max(worst, outcome.result, key=lambda r: ["pass", "soft_warn", "hard_fail"].index(r.value))

        # 3. 校验 craft
        if groups["craft"]:
            outcome = self._validate_craft(groups["craft"])
            all_details.extend(outcome.details)
            if outcome.result.value != "pass":
                worst = max(worst, outcome.result, key=lambda r: ["pass", "soft_warn", "hard_fail"].index(r.value))

        # 4. consume 和 gather 总是通过
        if groups["consume"]:
            items = [ic["item_name"] for ic in groups["consume"]]
            all_details.append(f"消耗: {', '.join(items)} ✅")

        if groups["gather"]:
            items = [ic["item_name"] for ic in groups["gather"]]
            all_details.append(f"采集: {', '.join(items)} ✅")

        # 构建消息
        if worst == ValidationResult.PASS:
            msg = "守恒校验通过"
            logger.debug(f"[Conservation] {msg}\n" + "\n".join(all_details))
        elif worst == ValidationResult.SOFT_WARN:
            msg = f"守恒校验通过（有警告）"
            logger.info(f"[Conservation] {msg}\n" + "\n".join(all_details))
        else:
            msg = f"⚠ 守恒校验失败 — 已回滚该次 inventory 变更"
            logger.warning(f"[Conservation] {msg}\n" + "\n".join(all_details))

        return ValidationOutcome(
            result=worst,
            message=msg,
            details=all_details,
        )

    # ═══════════════════════════════════════════
    # 类型校验
    # ═══════════════════════════════════════════

    def _validate_transfer(
        self, changes: list[dict], epsilon: float
    ) -> ValidationOutcome:
        """校验 transfer 类型：同一物品跨 NPC 转移必须 Σ(delta) = 0。

        把同一 tick 内所有 NPC 的 transfer 变化按 item_name 汇总：
          老张: 小麦 -5, 王老板: 小麦 +5  →  Σ = 0 ✅
          老张: 小麦 -5, 王老板: 金币 +4  →  Σ ≠ 0 ❌
        """
        items: dict[str, float] = {}
        for ic in changes:
            item = ic.get("item_name", "")
            qty = float(ic.get("quantity", 0))
            if ic.get("action") == "remove":
                qty = -qty
            items[item] = items.get(item, 0.0) + qty

        details = [f"  {item}: Σ = {delta:+.0f}" for item, delta in items.items()]
        unbalanced = {item: delta for item, delta in items.items() if abs(delta) > epsilon}

        if not unbalanced:
            return ValidationOutcome(ValidationResult.PASS, "transfer 守恒", details)

        fail_msg = "transfer 不守恒: " + ", ".join(
            f"{item} 差额={delta:+.0f}" for item, delta in unbalanced.items()
        )
        return ValidationOutcome(ValidationResult.HARD_FAIL, fail_msg, details)

    def _validate_craft(self, changes: list[dict]) -> ValidationOutcome:
        """校验 craft 类型：input/output 比例必须匹配已知配方。

        收集所有 craft 变更的 inputs（remove）和 outputs（add），
        然后在 RecipeRegistry 中查找匹配的配方。

        如果没有注册表或找不到配方 → SOFT_WARN（可能是 LLM 新发现的配方）
        如果找到配方但比例不匹配 → HARD_FAIL（LLM 算错了）
        """
        inputs: dict[str, int] = {}
        outputs: dict[str, int] = {}
        recipe_name = ""
        for ic in changes:
            rn = ic.get("recipe", "")
            if rn:
                recipe_name = rn
            item = ic.get("item_name", "")
            qty = int(ic.get("quantity", 0))
            if ic.get("action") == "remove":
                inputs[item] = inputs.get(item, 0) + qty
            else:
                outputs[item] = outputs.get(item, 0) + qty

        inp_str = " + ".join(f"{k}x{v}" for k, v in inputs.items())
        out_str = " + ".join(f"{k}x{v}" for k, v in outputs.items())
        detail_line = f"  {recipe_name}: {inp_str} → {out_str}"

        # 没有配方名 → soft warn
        if not recipe_name:
            return ValidationOutcome(
                ValidationResult.SOFT_WARN,
                f"craft 无配方名: {inp_str} → {out_str}（可能为新发现配方）",
                [detail_line],
            )

        # 尝试查询配方
        recipe = self._lookup_recipe(recipe_name)
        if recipe is None:
            return ValidationOutcome(
                ValidationResult.SOFT_WARN,
                f"craft [{recipe_name}] 未在注册表中找到（可能为新发现配方）",
                [detail_line],
            )

        # 校验比例
        if recipe["inputs"] == inputs and recipe["outputs"] == outputs:
            return ValidationOutcome(
                ValidationResult.PASS,
                f"craft [{recipe_name}] 配方匹配 ✅",
                [detail_line],
            )

        return ValidationOutcome(
            ValidationResult.HARD_FAIL,
            f"craft [{recipe_name}] 比例不匹配: 预期 "
            f"{recipe['inputs']} → {recipe['outputs']}, 实际 {inputs} → {outputs}",
            [detail_line],
        )

    # ═══════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════

    def _lookup_recipe(self, name: str) -> dict | None:
        """按名称从 RecipeRegistry 或 DerivationRegistry 查询配方。"""
        if self._registry:
            try:
                recipe = self._registry.get_by_name(name)
                if recipe:
                    return {
                        "inputs": dict(getattr(recipe, "inputs", {})),
                        "outputs": dict(getattr(recipe, "outputs", {})),
                    }
            except Exception:
                pass

        # 尝试 DerivationRegistry
        try:
            from ..entities.derivation import DerivationRegistry

            for chains in DerivationRegistry._chains.values():
                for c in chains:
                    if c.name == name:
                        return {
                            "inputs": dict(c.inputs),
                            "outputs": dict(c.outputs),
                        }
        except Exception:
            pass

        return None

    def check_inventory_changes(
        self, npc_name: str, inventory_changes: list[dict], epsilon: float = 0.001
    ) -> ValidationOutcome:
        """单独校验一个 NPC 的 inventory_changes（调试用）。"""
        return self.validate(
            [{"npc_name": npc_name, "inventory_changes": inventory_changes}],
            epsilon,
        )
