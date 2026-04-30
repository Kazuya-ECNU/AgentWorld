"""
RISC-like Entity Minimalization Axiom — Derivation Engine

When LLM deduces a non-existent entity:
  1. Try to derive from existing entities via process chain
  2. If not, decompose into naturally existing basic entities (recursive)
  3. If not decomposable, register as new basic entity

See DESIGN_PHILOSOPHY.md for conceptual documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ProcessChain:
    """
    A process chain that transforms input entities into output entities.

    This is a broader concept than Recipe: it represents any known transformation,
    including built-in recipes, LLM-discovered recipes, or hand-authored chains.
    """

    name: str                            # Chain name (e.g. "烘焙面包", "冶炼铁锭")
    inputs: dict[str, int]               # {entity_name: quantity}
    outputs: dict[str, int]              # {entity_name: quantity}
    action: str                          # Transformation action description
    zone_id: str | None = None           # Required zone (None = anywhere)
    object_type: str | None = None        # Required interaction object type

    def __str__(self) -> str:
        inp = " + ".join(f"{k}x{v}" for k, v in self.inputs.items())
        out = " + ".join(f"{k}x{v}" for k, v in self.outputs.items())
        obj = f" 需[{self.object_type}]" if self.object_type else ""
        loc = f" @ {self.zone_id}" if self.zone_id else ""
        return f"{self.name}: {inp} → {out}{obj}{loc}"

    def __repr__(self) -> str:
        return str(self)


class DerivationRegistry:
    """
    Central registry of:
      - Basic entities (irreducible primitives)
      - Process chains (transformations that produce entities)

    This is the knowledge base that the 3-step axiom queries during derivation.

    Usage:
      DerivationRegistry.init_defaults()     # seed + sync from RecipeRegistry
      DerivationRegistry.derive(name, set)   # 3-step algorithm
    """

    _chains: ClassVar[dict[str, list[ProcessChain]]] = {}
    """entity_name → list of chains that produce it"""

    _basic_entities: ClassVar[set[str]] = set()
    """names of irreducible basic entities"""

    _DEFAULT_BASIC_ENTITIES: ClassVar[set[str]] = {
        "水",       # water
        "石头",     # stone
        "木材",     # wood
        "铁矿石",   # iron ore
        "砂",       # sand
        "陶土",     # clay
        "种子",     # seeds
        "野果",     # wild fruit
    }

    # ─── Initialization ───

    @classmethod
    def init_defaults(cls):
        """Seed with basic entities and sync process chains from RecipeRegistry."""
        cls._basic_entities = set(cls._DEFAULT_BASIC_ENTITIES)
        cls._chains.clear()
        cls._sync_from_recipe_registry()
        logger.info(
            f"[DerivationRegistry] 初始化: {len(cls._basic_entities)} 基础实体, "
            f"{sum(len(v) for v in cls._chains.values())} 条制造链"
        )

    @classmethod
    def _sync_from_recipe_registry(cls):
        """Import all recipes from RecipeRegistry as process chains."""
        try:
            from .recipe import RecipeRegistry

            for recipe in RecipeRegistry.get_all():
                chain = ProcessChain(
                    name=recipe.name,
                    inputs=dict(recipe.inputs),
                    outputs=dict(recipe.outputs),
                    action=recipe.name,
                    zone_id=recipe.zone_id,
                    object_type=recipe.required_object_type,
                )
                for output_name in recipe.outputs:
                    cls._chains.setdefault(output_name, []).append(chain)
        except ImportError:
            logger.warning("[DerivationRegistry] RecipeRegistry 不可用，跳过同步")

    @classmethod
    def refresh_from_recipe_registry(cls):
        """Re-sync chains from RecipeRegistry (call after new recipes are registered)."""
        cls._chains.clear()
        cls._sync_from_recipe_registry()
        chain_count = sum(len(v) for v in cls._chains.values())
        logger.info(f"[DerivationRegistry] 重新同步: {chain_count} 条制造链")

    # ─── Registration ───

    @classmethod
    def register_chain(cls, chain: ProcessChain):
        """Register a new process chain."""
        for output_name in chain.outputs:
            cls._chains.setdefault(output_name, []).append(chain)
        logger.debug(f"[DerivationRegistry] 注册制造链: {chain}")

    @classmethod
    def add_basic_entity(cls, name: str):
        """Add a new basic entity (one that cannot be decomposed further)."""
        cls._basic_entities.add(name)
        logger.debug(f"[DerivationRegistry] 注册基础实体: {name}")

    # ─── Queries ───

    @classmethod
    def is_basic(cls, entity_name: str) -> bool:
        """Is this entity irreducible (a basic primitive)?"""
        return entity_name in cls._basic_entities

    @classmethod
    def get_chains_producing(cls, entity_name: str) -> list[ProcessChain]:
        """All known chains that can produce this entity."""
        return cls._chains.get(entity_name, [])

    @classmethod
    def get_all_basic_entities(cls) -> set[str]:
        """Return all registered basic entity names."""
        return set(cls._basic_entities)

    @classmethod
    def get_all_chain_outputs(cls) -> set[str]:
        """Return all entity names that can be produced by at least one chain."""
        return set(cls._chains.keys())

    # ─── 3-Step Derivation ───

    @classmethod
    def derive(
        cls,
        target_name: str,
        existing_set: set[str],
        depth: int = 0,
    ) -> tuple[set[str], list[ProcessChain]]:
        """
        The 3-step entity minimalization axiom.

        Args:
            target_name: The entity name to derive (e.g. "面包", "面粉")
            existing_set: Set of entity names that already exist in the world
            depth: Internal recursion depth guard

        Returns:
            (new_basic_entities_needed, new_chains_needed)
            - new_basic_entities_needed: set of irreducible basic entities
              that need to be registered (empty if all inputs exist)
            - new_chains_needed: list of process chains that need to be
              registered to make the derivation work

        Algorithm:
          1. Already exists → (∅, ∅)
          2. Is basic & not existing → ({name}, ∅)
          3. Has chain & all inputs exist → (∅, [chain])
          4. Has chain & some inputs missing → recurse → union of results
          5. No chain → ({name}, ∅) — register as new basic
        """
        # Depth guard
        if depth > 10:
            logger.warning(
                f"[Derivation] 递归深度超限 ({target_name}, depth={depth})，"
                f"作为基础实体注册"
            )
            return {target_name}, []

        marker = "─" * (depth + 1)
        logger.debug(f"{marker} derive: {target_name} (depth={depth})")

        # Step 0: Already exists → nothing needed
        if target_name in existing_set:
            return set(), []

        # Step 3: Is it an irreducible basic entity?
        if cls.is_basic(target_name):
            logger.info(f"[Derivation] {target_name} 是基础实体，需要注册")
            return {target_name}, []

        # Step 1: Try to find a process chain
        chains = cls.get_chains_producing(target_name)
        if chains:
            chain = chains[0]
            logger.info(f"[Derivation] 找到 {target_name} 的制造链: {chain.name}")

            needed_basics: set[str] = set()
            all_inputs_exist = True

            for input_name in chain.inputs:
                sub_basics, sub_chains = cls.derive(
                    input_name, existing_set, depth + 1
                )
                needed_basics.update(sub_basics)
                if sub_basics:
                    all_inputs_exist = False

            if all_inputs_exist:
                # All chain inputs already exist → only need this chain
                logger.info(
                    f"[Derivation] {target_name} 的输入已全部存在，"
                    f"仅需注册制造链"
                )
                return set(), [chain]

            # Some inputs need basic entity registration
            logger.info(
                f"[Derivation] {target_name} 需要 {len(needed_basics)} "
                f"个基础实体支持"
            )
            return needed_basics, [chain]

        # Step 2 & 3: No chain exists → register as new basic entity
        logger.info(f"[Derivation] 找不到 {target_name} 的制造链，注册为基础实体")
        return {target_name}, []


class EntityDerivationEngine:
    """
    Integration bridge between DerivationRegistry and GraphEngine.

    Wraps the 3-step algorithm into a single derive_and_register() call
    that handles entity creation, chain registration, and logging.
    """

    def __init__(self, graph_engine):
        from ..services.graph_engine import GraphEngine

        self._graph: GraphEngine = graph_engine

    # ─── Public API ───

    def derive_and_register(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "item",
    ) -> str:
        """
        Full 3-step derivation: check existence → derive → register.

        This is the main entry point called by GraphEngine._auto_register_missing().

        Args:
            entity_id: Desired entity ID (e.g. "item_面包")
            name: Human-readable name (e.g. "面包")
            entity_type: "item", "npc", or "object"

        Returns:
            The actual entity_id that was registered (may differ from input
            if an entity with the same name already exists).
        """
        from ..entities.base_entity import Entity

        # Step 0: Already exists by entity_id
        existing = self._graph.get_entity(entity_id)
        if existing:
            return entity_id

        # Step 0: Already exists by name
        for ent in self._graph.all_entities():
            if ent.name == name and ent.entity_type == entity_type:
                return ent.entity_id

        # For non-item entities (npc/object), register directly without derivation
        if entity_type != "item":
            ent = Entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
            )
            self._graph.register_entity(ent)
            logger.info(
                f"[Derivation] 注册非物品实体: {entity_id} ({name}, {entity_type})"
            )
            return entity_id

        # Build the set of existing item-level entity names
        existing_names = {
            e.name for e in self._graph.all_entities() if e.entity_type == "item"
        }

        # Run 3-step derivation
        needed_basics, needed_chains = DerivationRegistry.derive(
            name, existing_names
        )

        # Register any indirectly-needed basic entities
        for basic_name in needed_basics:
            if basic_name != name:
                basic_eid = f"item_{basic_name}"
                if not self._graph.get_entity(basic_eid):
                    ent = Entity(
                        entity_id=basic_eid,
                        name=basic_name,
                        entity_type="item",
                    )
                    self._graph.register_entity(ent)
                    DerivationRegistry.add_basic_entity(basic_name)
                    logger.info(
                        f"[Derivation] 注册间接基础实体: {basic_eid} ({basic_name})"
                    )

        # Register the target entity itself
        target_ent = Entity(
            entity_id=entity_id,
            name=name,
            entity_type="item",
        )
        self._graph.register_entity(target_ent)

        # Register target as basic entity if that's how it was resolved
        if name in needed_basics:
            DerivationRegistry.add_basic_entity(name)

        # Log derivation summary
        if needed_basics and name in needed_basics and not needed_chains:
            logger.info(
                f"[Derivation] ③ {name}({entity_id}) 注册为新的基础实体 "
                f"(无制造链)"
            )
        elif needed_chains:
            chain_names = [c.name for c in needed_chains]
            logger.info(
                f"[Derivation] ① {name}({entity_id}) 通过制造链推导: "
                f"{' → '.join(chain_names)}"
            )
            if needed_basics:
                logger.info(
                    f"[Derivation] ② 同时注册 {len(needed_basics)} 个支持基础实体: "
                    f"{needed_basics}"
                )
        else:
            logger.info(f"[Derivation] {name}({entity_id}) 已存在，无需操作")

        return entity_id
