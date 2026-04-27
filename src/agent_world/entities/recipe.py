"""
Recipe 系统 —— 实体间物品转换配方

每个 Recipe = NPC 通过与某物体的交互完成物品转换。
转换不是魔法——NPC 必须持有到目标物体的边，交互后扣减输入、增加输出。

架构：
  配方（知识） → NPC → [物体边] → 交互执行 → 扣原料 + 增产物 + 耗体力
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger("recipe_registry")


@dataclass
class Recipe:
    """
    一个配方 = NPC 通过与特定物体交互完成物品转换。

    Attributes:
        name: 配方名称（如"烘焙面包"）
        inputs: {物品名: 消耗数量}
        outputs: {物品名: 产出数量}
        required_object_type: 需要交互的世界物体类型（如"磨具"、"熔炉"、"酿酒桶"）
        zone_id: 限制区域（None 表示不限）
        vitality_cost: 执行消耗的体力
        description: 人类可读描述
        source: "builtin" 内置 / "llm" LLM 发现
    """
    name: str
    inputs: dict[str, int]
    outputs: dict[str, int]
    required_object_type: str | None = None   # 需要与何种物体交互
    required_object_interface: str | None = None  # 使用物体的哪个接口
    zone_id: str | None = None
    vitality_cost: int = 10
    description: str = ""
    source: str = "builtin"

    def __str__(self) -> str:
        inp = " + ".join(f"{k}x{v}" for k, v in self.inputs.items())
        out = " + ".join(f"{k}x{v}" for k, v in self.outputs.items())
        obj = f" 需[{self.required_object_type}]" if self.required_object_type else ""
        loc = f" @ {self.zone_id}" if self.zone_id else ""
        return f"{self.name}: {inp} → {out}{obj}{loc} (体耗{self.vitality_cost})"

    def to_prompt_line(self) -> str:
        """给 LLM 看到的一行描述"""
        inp = " + ".join(f"{k}x{v}" for k, v in self.inputs.items())
        out = " + ".join(f"{k}x{v}" for k, v in self.outputs.items())
        parts = [f"  - {self.name}: {inp} → {out}"]
        if self.required_object_type:
            parts[-1] += f"  (需与 [{self.required_object_type}] 交互)"
        parts[-1] += f"  体力-{self.vitality_cost}"
        return parts[-1]


class RecipeRegistry:
    """
    中央配方注册表。
    - 预置内置配方（含物体依赖）
    - 支持 LLM 动态发现新配方
    - 按区域 / 物体类型 / 名称查询
    """

    _recipes: ClassVar[dict[str, Recipe]] = {}
    _zone_index: ClassVar[dict[str, list[str]]] = {}
    _object_type_index: ClassVar[dict[str, list[str]]] = {}

    # ─── 内置配方 ───

    _DEFAULT_RECIPES: ClassVar[list[Recipe]] = [
        Recipe(
            name="烘焙面包",
            inputs={"小麦": 2},
            outputs={"面包": 1},
            required_object_type="烤炉",
            required_object_interface="烤制",
            zone_id="farm",
            vitality_cost=5,
            description="用磨具将小麦磨粉后在烤炉烘焙",
        ),
        Recipe(
            name="碾磨面粉",
            inputs={"小麦": 3},
            outputs={"面粉": 2},
            required_object_type="磨具",
            required_object_interface="研磨",
            zone_id="farm",
            vitality_cost=8,
            description="用磨具将小麦磨成面粉",
        ),
        Recipe(
            name="锻造工具",
            inputs={"铁锭": 2},
            outputs={"工具": 1},
            required_object_type="铁砧",
            required_object_interface="锻造",
            zone_id="market",
            vitality_cost=8,
            description="在铁砧上锻造工具",
        ),
        Recipe(
            name="酿酒",
            inputs={"小麦": 3, "蔬菜": 1},
            outputs={"酒": 2},
            required_object_type="酿酒桶",
            required_object_interface="酿造",
            zone_id="tavern",
            vitality_cost=10,
            description="用酿酒桶酿造酒",
        ),
        Recipe(
            name="制作药水",
            inputs={"草药": 3},
            outputs={"药水": 1},
            required_object_type="药臼",
            required_object_interface="研磨",
            zone_id="temple",
            vitality_cost=5,
            description="用药臼研磨草药制成药水",
        ),
        Recipe(
            name="加工皮毛",
            inputs={"皮毛": 2},
            outputs={"衣物": 1},
            required_object_type="缝纫台",
            required_object_interface="缝制",
            zone_id="forest",
            vitality_cost=8,
            description="在缝纫台上将皮毛加工成衣物",
        ),
        Recipe(
            name="建造家具",
            inputs={"木材": 3},
            outputs={"家具": 1},
            required_object_type="工作台",
            required_object_interface="建造",
            zone_id="market",
            vitality_cost=15,
            description="在工作台上建造家具",
        ),
    ]

    @classmethod
    def init_defaults(cls):
        cls._recipes.clear()
        cls._zone_index.clear()
        cls._object_type_index.clear()
        for r in cls._DEFAULT_RECIPES:
            cls._register_builtin(r)
        # Sync process chains to DerivationRegistry
        cls._sync_to_derivation_registry()

    @classmethod
    def _sync_to_derivation_registry(cls):
        """Register all recipes as process chains in DerivationRegistry."""
        try:
            from .derivation import DerivationRegistry, ProcessChain
            DerivationRegistry.init_defaults()
            # DerivationRegistry.init_defaults already calls _sync_from_recipe_registry
            logger.info(
                f"[RecipeRegistry] 已同步 {len(cls._recipes)} 个配方到 DerivationRegistry"
            )
        except ImportError:
            logger.debug("[RecipeRegistry] DerivationRegistry 不可用，跳过同步")
        except Exception as e:
            logger.warning(f"[RecipeRegistry] 同步 DerivationRegistry 失败: {e}")

    @classmethod
    def _register_builtin(cls, recipe: Recipe):
        cls._recipes[recipe.name] = recipe
        if recipe.zone_id:
            cls._zone_index.setdefault(recipe.zone_id, []).append(recipe.name)
        if recipe.required_object_type:
            cls._object_type_index.setdefault(recipe.required_object_type, []).append(recipe.name)

    @classmethod
    def register_llm_recipe(cls, recipe: Recipe) -> bool:
        """LLM 发现并注册新配方（不覆盖内置配方）"""
        if recipe.name in cls._recipes:
            return False
        recipe.source = "llm"
        cls._recipes[recipe.name] = recipe
        if recipe.zone_id:
            cls._zone_index.setdefault(recipe.zone_id, []).append(recipe.name)
        if recipe.required_object_type:
            cls._object_type_index.setdefault(recipe.required_object_type, []).append(recipe.name)
        # Also register as process chain in DerivationRegistry
        try:
            from .derivation import DerivationRegistry, ProcessChain
            chain = ProcessChain(
                name=recipe.name,
                inputs=dict(recipe.inputs),
                outputs=dict(recipe.outputs),
                action=recipe.name,
                zone_id=recipe.zone_id,
                object_type=recipe.required_object_type,
            )
            DerivationRegistry.register_chain(chain)
            logger.info(f"[RecipeRegistry] LLM配方已同步到 DerivationRegistry: {recipe.name}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[RecipeRegistry] 同步 LLM 配方失败: {e}")
        return True

    # ─── 查询 ───

    @classmethod
    def get_by_zone(cls, zone_id: str) -> list[Recipe]:
        return [cls._recipes[n] for n in cls._zone_index.get(zone_id, []) if n in cls._recipes]

    @classmethod
    def get_by_object_type(cls, obj_type: str) -> list[Recipe]:
        return [cls._recipes[n] for n in cls._object_type_index.get(obj_type, []) if n in cls._recipes]

    @classmethod
    def get_by_name(cls, name: str) -> Recipe | None:
        return cls._recipes.get(name)

    @classmethod
    def get_all(cls) -> list[Recipe]:
        return list(cls._recipes.values())

    @classmethod
    def get_available_for_npc(cls, zone_id: str, inventory: dict[str, int],
                              connected_object_types: list[str]) -> list[Recipe]:
        """
        获取 NPC 在当前位置能用的所有配方。

        Args:
            zone_id: NPC 所在区域
            inventory: {物品名: 数量}——判断原料是否够
            connected_object_types: NPC 连接的物体类型列表

        Returns:
            可用的配方列表
        """
        results = []
        seen = set()
        for r in cls._recipes.values():
            if r.name in seen:
                continue
            seen.add(r.name)

            # 区域检查
            if r.zone_id and r.zone_id != zone_id:
                continue

            # 物体检查：如果有物体依赖，NPC 必须连接到该类型物体
            if r.required_object_type:
                if r.required_object_type not in connected_object_types:
                    continue

            # 库存检查
            if cls._can_afford(r, inventory):
                results.append(r)

        return results

    @classmethod
    def _can_afford(cls, recipe: Recipe, inventory: dict[str, int]) -> bool:
        return all(inventory.get(item, 0) >= qty for item, qty in recipe.inputs.items())

    @classmethod
    def format_for_prompt(cls, available: list[Recipe]) -> str:
        if not available:
            return ""
        lines = ["### 可用配方（通过与物体交互制造物品）"]
        lines.append("你可以使用库存中的材料，通过与特定物体交互来制造新物品：")
        for r in available:
            lines.append(r.to_prompt_line())
        lines.append("")
        lines.append("选择配方后，系统会自动检查物体连接，扣减原料，产出物品。")
        lines.append("也可以自创配方——在 action 中写新配方名并引用物品，系统会自动注册。")
        lines.append("")
        return "\n".join(lines)
