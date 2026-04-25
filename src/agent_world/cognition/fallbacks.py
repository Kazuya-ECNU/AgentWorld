"""
Fallback Engine - 规则引擎兜底

当 LLM 不可用或调用失败时，使用规则引擎作为兜底方案。
规则引擎基于 NPC 的 role、状态、位置等信息，以硬编码条件分支的方式决定目标。

这保证了即使 LLM 不可用，NPC 仍然有合理的行为选择。
"""

from typing import Optional
from .reasoner import GoalOutput, GoalType


# === 规则定义 ===

class GoalRule:
    """单条规则"""
    def __init__(self, name: str, condition_fn, goal_type: str, priority: int = 0):
        self.name = name
        self.condition_fn = condition_fn  # (npc_state) -> bool
        self.goal_type = goal_type
        self.priority = priority


# 规则集合（Role -> [GoalRules]）
ROLE_RULES: dict[str, list[GoalRule]] = {
    "merchant": [
        GoalRule("天黑回家", lambda s: s.get("is_night", False), GoalType.REST, priority=15),
        GoalRule("能量低需要休息", lambda s: s.get("energy", 100) < 30, GoalType.REST, priority=10),
        GoalRule("商人去摆摊", lambda s: True, GoalType.TRADE, priority=5),
        GoalRule("商人探索新市场", lambda s: s.get("energy", 100) > 60, GoalType.EXPLORE, priority=3),
    ],
    "farmer": [
        GoalRule("天黑不干活", lambda s: s.get("is_night", False), GoalType.REST, priority=15),
        GoalRule("能量低需要休息", lambda s: s.get("energy", 100) < 30, GoalType.REST, priority=10),
        GoalRule("农民去农场干活", lambda s: True, GoalType.FARM, priority=7),
        GoalRule("农民休息", lambda s: s.get("energy", 100) > 80 and s.get("last_action") == "farm", GoalType.REST, priority=4),
    ],
    "miner": [
        GoalRule("夜间不挖矿", lambda s: s.get("is_night", False), GoalType.REST, priority=15),
        GoalRule("能量低需要休息", lambda s: s.get("energy", 100) < 35, GoalType.REST, priority=10),
        GoalRule("矿工去挖矿", lambda s: True, GoalType.MINE, priority=6),
    ],
    "guard": [
        GoalRule("守卫夜间巡逻", lambda s: s.get("is_night", False), GoalType.WORK, priority=14),
        GoalRule("守卫白天巡逻", lambda s: True, GoalType.WORK, priority=6),
        GoalRule("守卫休息", lambda s: s.get("energy", 100) < 25, GoalType.REST, priority=10),
    ],
    "scholar": [
        GoalRule("夜间休息", lambda s: s.get("is_night", False), GoalType.REST, priority=15),
        GoalRule("学者研究", lambda s: True, GoalType.WORK, priority=6),
        GoalRule("学者社交", lambda s: s.get("energy", 100) > 50, GoalType.SOCIALIZE, priority=4),
        GoalRule("学者休息", lambda s: s.get("energy", 100) < 30, GoalType.REST, priority=10),
    ],
    "healer": [
        GoalRule("夜间休息", lambda s: s.get("is_night", False), GoalType.REST, priority=15),
        GoalRule("治疗师治疗", lambda s: True, GoalType.WORK, priority=6),
        GoalRule("治疗师休息", lambda s: s.get("energy", 100) < 30, GoalType.REST, priority=10),
    ],
    "wanderer": [
        GoalRule("流浪者探索", lambda s: not s.get("is_night", False), GoalType.EXPLORE, priority=7),
        GoalRule("流浪者夜间休息", lambda s: s.get("is_night", False), GoalType.REST, priority=14),
        GoalRule("流浪者社交", lambda s: s.get("energy", 100) > 50, GoalType.SOCIALIZE, priority=4),
        GoalRule("流浪者休息", lambda s: s.get("energy", 100) < 35, GoalType.REST, priority=10),
    ],
}

ROLE_GOAL_PLANS: dict[str, list[str]] = {
    GoalType.TRADE: ["移动到 market", "摆摊交易"],
    GoalType.FARM: ["移动到 farm", "在农场耕作"],
    GoalType.MINE: ["移动到 mine", "挖掘矿石"],
    GoalType.REST: ["移动到 tavern", "在酒馆休息"],
    GoalType.SOCIALIZE: ["移动到 tavern", "在酒馆聊天"],
    GoalType.EXPLORE: ["探索相邻区域"],
    GoalType.WORK: ["执行工作"],
}

# 默认规则（未匹配到 Role 时使用）
DEFAULT_RULES = [
    GoalRule("通用休息", lambda s: s.get("energy", 100) < 30, GoalType.REST, priority=10),
    GoalRule("通用工作", lambda s: True, GoalType.WORK, priority=5),
    GoalRule("通用探索", lambda s: s.get("energy", 100) > 50, GoalType.EXPLORE, priority=3),
]


# === Fallback Engine ===

class FallbackEngine:
    """
    规则引擎兜底。
    
    当 LLM 不可用时，按以下顺序决定 Goal：
    1. 按 priority 从高到低遍历规则
    2. 第一个满足 condition 的规则生效
    3. 返回对应的 GoalOutput
    """

    def resolve(self, role: str, state: dict) -> GoalOutput:
        """
        根据 role 和 state 解析出 Goal。
        
        Args:
            role: NPC 的角色 (merchant, farmer, miner...)
            state: NPC 当前状态 dict，包含 energy, inventory, position, last_action 等
        """
        rules = ROLE_RULES.get(role, DEFAULT_RULES)

        # 按 priority 从高到低排序
        sorted_rules = sorted(rules, key=lambda r: -r.priority)

        for rule in sorted_rules:
            try:
                if rule.condition_fn(state):
                    return GoalOutput(
                        goal=rule.goal_type,
                        reason=f"[规则兜底] {rule.name}",
                        urgency=min(1.0, rule.priority / 10.0),
                        plan=ROLE_GOAL_PLANS.get(rule.goal_type, []),
                    )
            except Exception:
                continue

        # 兜底：idle
        return GoalOutput(
            goal=GoalType.IDLE,
            reason="[规则兜底] 无匹配规则，默认空闲",
            urgency=0.1,
        )


# === 全局实例 ===
RULE_GOALS = [r.goal_type for r in DEFAULT_RULES]
fallback_engine = FallbackEngine()