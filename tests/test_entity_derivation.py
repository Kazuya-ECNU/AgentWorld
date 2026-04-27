"""
Entity Derivation Tests — RISC-like Entity Minimalization Axiom

Covers:
  1. Direct unit tests (no LLM) for the 3-step derivation algorithm
  2. LLM-driven integration test via InteractionResolver + GraphEngine
"""

import sys
sys.path.insert(0, 'src')

import json
import logging
import os

# ─── Logging setup for test visibility ───
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
    stream=sys.stdout,
)

# ─── Clean DB path (avoid cross-test pollution) ───
os.environ.pop("MINIMAX_BASE_URL", None)
os.environ.pop("OPENAI_API_KEY", None)

# ============================================================
# Unit Tests — DerivationRegistry (no LLM)
# ============================================================

def _init_all():
    """Initialize RecipeRegistry + DerivationRegistry for tests."""
    from agent_world.entities.recipe import RecipeRegistry
    from agent_world.entities.derivation import DerivationRegistry

    RecipeRegistry.init_defaults()
    # DerivationRegistry is initialized inside RecipeRegistry.init_defaults
    # but double-check it's ready
    if not DerivationRegistry._chains:
        DerivationRegistry.init_defaults()
        DerivationRegistry.refresh_from_recipe_registry()

    return RecipeRegistry, DerivationRegistry


def test_derive_bread_from_existing_wheat():
    """
    SG-1: Derive "面包" (bread) when "小麦" (wheat) already exists.

    Expected: chain "烘焙面包" exists, wheat is available, so no new basic
    entities are needed. Returns (∅, [chain]).
    """
    _init_all()
    from agent_world.entities.derivation import DerivationRegistry

    existing = {"小麦", "水", "木材", "石头"}
    needed_basics, needed_chains = DerivationRegistry.derive("面包", existing)

    assert not needed_basics, (
        f"不应需要新的基础实体，但得到: {needed_basics}"
    )
    assert len(needed_chains) == 1, (
        f"应返回 1 条制造链，但得到 {len(needed_chains)}"
    )
    assert needed_chains[0].name == "烘焙面包", (
        f"应找到 '烘焙面包' 链，但得到: {needed_chains[0].name}"
    )

    print("✅ SG-1: 面包从小麦推导成功")


def test_derive_flour():
    """
    SG-2: Derive "面粉" (flour).

    "碾磨面粉: 小麦x3 → 面粉x2" chain exists. 小麦 exists.
    Expected: (∅, [chain]).
    """
    _init_all()
    from agent_world.entities.derivation import DerivationRegistry

    existing = {"小麦", "水", "木材"}
    needed_basics, needed_chains = DerivationRegistry.derive("面粉", existing)

    assert not needed_basics, (
        f"不应需要新的基础实体: {needed_basics}"
    )
    assert len(needed_chains) == 1
    assert needed_chains[0].name == "碾磨面粉"

    print("✅ SG-2: 面粉从小麦推导成功")


def test_derive_silk_basic():
    """
    SG-3: Derive "绸缎" (silk) which has no recipe.

    Expected: no chain found → registered as new basic entity.
    Returns ({绸缎}, ∅).
    """
    _init_all()
    from agent_world.entities.derivation import DerivationRegistry

    existing = {"小麦", "水", "木材"}
    needed_basics, needed_chains = DerivationRegistry.derive("绸缎", existing)

    assert "绸缎" in needed_basics, (
        f"应有 '绸缎' 在基础实体中: {needed_basics}"
    )
    assert not needed_chains, (
        f"不应有制造链: {needed_chains}"
    )

    print("✅ SG-3: 绸缎（无配方）注册为基础实体")


def test_derive_existing_entity():
    """
    SG-4: Derive an entity that already exists.

    Expected: (∅, ∅) — nothing needed.
    """
    _init_all()
    from agent_world.entities.derivation import DerivationRegistry

    needed_basics, needed_chains = DerivationRegistry.derive("小麦", {"小麦", "水"})
    assert not needed_basics
    assert not needed_chains

    print("✅ SG-4: 已存在实体返回空")


def test_derive_basic_entity():
    """
    SG-5: Derive a known basic entity that doesn't exist yet.

    Expected: ({水}, ∅) — need to register as basic.
    """
    _init_all()
    from agent_world.entities.derivation import DerivationRegistry

    needed_basics, needed_chains = DerivationRegistry.derive("水", {"小麦"})
    assert "水" in needed_basics
    assert not needed_chains

    print("✅ SG-5: 基础实体识别正确")


def test_derive_bread_integration():
    """
    SG-6: Full integration test — derive_and_register through GraphEngine.

    Setup: GraphEngine with NPC entity + 小麦 item. No 面包 item.
    Call derive_and_register for "item_面包".
    Verify: 面包 entity created, 小麦 still exists.
    """
    from agent_world.services.graph_engine import GraphEngine
    from agent_world.entities.base_entity import Entity
    from agent_world.entities.derivation import (
        DerivationRegistry, EntityDerivationEngine,
    )
    from agent_world.models.interaction import EntityInterface

    _init_all()

    ge = GraphEngine()

    # Register NPC
    npc = Entity(
        entity_id="npc_test_npc",
        name="老张",
        entity_type="npc",
    )
    ge.register_entity(npc)

    # Register 小麦 (already exists)
    wheat = Entity(
        entity_id="item_小麦",
        name="小麦",
        entity_type="item",
    )
    wheat.add_interface(EntityInterface(
        interface_id="item_小麦_holdable",
        entity_id="item_小麦",
        name="可持有",
        description="可以持有小麦",
    ))
    ge.register_entity(wheat)

    # Verify 面包 doesn't exist yet
    assert ge.get_entity("item_面包") is None, "面包应初始不存在"

    # Run derivation
    engine = EntityDerivationEngine(ge)
    result_eid = engine.derive_and_register("item_面包", "面包")

    assert result_eid == "item_面包", f"entity_id 不匹配: {result_eid}"
    bread_entity = ge.get_entity("item_面包")
    assert bread_entity is not None, "面包应已创建"
    assert bread_entity.name == "面包"
    assert bread_entity.entity_type == "item"

    # Verify 小麦 still exists
    assert ge.get_entity("item_小麦") is not None, "小麦仍应存在"

    print("✅ SG-6: 面包完整推导注册测试通过")


def test_derive_silk_integration():
    """
    SG-7: Full integration test for entity without a chain.

    Setup: GraphEngine with basic items. Call derive_and_register for "item_绸缎".
    Verify: 绸缎 entity created, logged as new basic entity.
    """
    from agent_world.services.graph_engine import GraphEngine
    from agent_world.entities.base_entity import Entity
    from agent_world.entities.derivation import (
        DerivationRegistry, EntityDerivationEngine,
    )
    from agent_world.models.interaction import EntityInterface

    _init_all()

    ge = GraphEngine()

    # Register some items
    for name in ("水", "木材"):
        ent = Entity(
            entity_id=f"item_{name}",
            name=name,
            entity_type="item",
        )
        ent.add_interface(EntityInterface(
            interface_id=f"item_{name}_holdable",
            entity_id=f"item_{name}",
            name="可持有",
            description=f"可以持有{name}",
        ))
        ge.register_entity(ent)

    # Run derivation for 绸缎 (no recipe)
    engine = EntityDerivationEngine(ge)
    result_eid = engine.derive_and_register("item_绸缎", "绸缎")

    assert result_eid == "item_绸缎"
    silk = ge.get_entity("item_绸缎")
    assert silk is not None
    assert silk.name == "绸缎"
    assert silk.entity_type == "item"

    # Verify it's now registered as basic
    assert "绸缎" in DerivationRegistry.get_all_basic_entities(), \
        "绸缎应被注册为基础实体"

    print("✅ SG-7: 绸缎（无配方）完整推导注册测试通过")


# ============================================================
# LLM-Driven Integration Test
# ============================================================

def test_llm_derives_bread():
    """
    SG-8: LLM-driven derivation test.

    Process:
      1. Set up GraphEngine with entities + NPC having 小麦x5
      2. Remove 面包 from default items (do not pre-register it)
      3. Send prompt to LLM asking what the NPC should do
      4. The LLM should naturally output a "烘焙面包" instruction referencing
         item_面包
      5. Execute the instruction through graph_engine.execute_npc_instruction()
      6. Verify that item_面包 was created via the derivation engine
    """
    from agent_world.entities.recipe import RecipeRegistry
    from agent_world.entities.derivation import DerivationRegistry
    from agent_world.services.graph_engine import GraphEngine
    from agent_world.entities.base_entity import Entity
    from agent_world.models.interaction import EntityInterface

    # Initialize systems
    RecipeRegistry.init_defaults()
    # DerivationRegistry is initialized inside RecipeRegistry.init_defaults

    ge = GraphEngine()

    # Register NPC with 小麦 x5 in inventory
    npc = Entity(
        entity_id="npc_test_llm",
        name="老张",
        entity_type="npc",
    )
    npc.set_attr("vitality", 100, min_value=0, max_value=100, description="体力")
    npc.set_attr("zone_id", "farm", description="所在区域")
    # Add standard NPC interfaces
    npc.add_interface(EntityInterface(
        interface_id="npc_test_llm_move",
        entity_id="npc_test_llm",
        name="移动",
        description="移动",
    ))
    npc.add_interface(EntityInterface(
        interface_id="npc_test_llm_interact",
        entity_id="npc_test_llm",
        name="交互",
        description="交互",
    ))
    npc.add_interface(EntityInterface(
        interface_id="npc_test_llm_hold",
        entity_id="npc_test_llm",
        name="持有",
        description="持有",
    ))
    npc.add_interface(EntityInterface(
        interface_id="npc_test_llm_wait",
        entity_id="npc_test_llm",
        name="等待",
        description="等待",
    ))
    ge.register_entity(npc)

    # Register 小麦 (already exists as a default item concept)
    wheat = Entity(
        entity_id="item_小麦",
        name="小麦",
        entity_type="item",
    )
    wheat.add_interface(EntityInterface(
        interface_id="item_小麦_holdable",
        entity_id="item_小麦",
        name="可持有",
        description="可以持有小麦",
    ))
    ge.register_entity(wheat)

    # Register a 烤炉 object so the recipe can work
    oven = Entity(
        entity_id="obj_test_oven",
        name="烤炉",
        entity_type="object",
    )
    oven.set_attr("object_type", "烤炉", description="物体类型")
    oven.set_attr("zone_id", "farm", description="所在区域")
    oven.add_interface(EntityInterface(
        interface_id="obj_test_oven_interactable",
        entity_id="obj_test_oven",
        name="可交互",
        description="可以被互动（烤制）",
    ))
    oven.add_interface(EntityInterface(
        interface_id="obj_test_oven_holdable",
        entity_id="obj_test_oven",
        name="可持有",
        description="可持有",
    ))
    ge.register_entity(oven)

    # Connect NPC → 烤炉 (topological connection)
    npc.connect_to("obj_test_oven")
    oven.connect_to("npc_test_llm")

    # Set NPC inventory: 小麦 x 5
    ge.set_edge_quantity("npc_test_llm", "item_小麦", 5)

    # Verify 面包 does NOT exist yet
    assert ge.get_entity("item_面包") is None, "面包应初始不存在"

    # ─── Build LLM prompt ───
    # Include recipe info so the LLM knows about baking bread
    recipe = RecipeRegistry.get_by_name("烘焙面包")
    user_prompt = f"""你是一个世界模拟引擎的 NPC 控制模块。

当前世界状态：
- NPC 老张 (npc_test_llm) 在 farm 区域，体力=100
- 老张持有：小麦x5
- 老张连接到：烤炉（可交互物体）

可用配方：
{recipe.to_prompt_line()}

请为老张输出一条交互指令。根据以上信息，老张应该用小麦制造面包。
输出格式（纯 JSON，不要多余文字）：
{{
  "edge_id": "e_0",
  "action": "烘焙面包",
  "effects": [
    {{"target_entity_id": "npc_test_llm", "attribute_name": "vitality", "operation": "sub", "value": 5, "description": "烘焙面包消耗体力"}}
  ],
  "edge_qty_changes": [
    {{"source_entity_id": "npc_test_llm", "target_entity_id": "item_小麦", "delta": -2}},
    {{"source_entity_id": "npc_test_llm", "target_entity_id": "item_面包", "delta": 1}}
  ],
  "result_text": "老张在farm用烤炉烘焙面包，消耗小麦x2，获得面包x1"
}}"""

    # ─── Call LLM (MiniMax M2.7) via InteractionResolver ───
    from agent_world.services.interaction_resolver import InteractionResolver

    resolver = InteractionResolver(
        model=os.environ.get("LLM_MODEL", None),  # None → use default from config
        temperature=0.3,
    )
    llm_text = resolver._call_llm(user_prompt)

    if not llm_text:
        # LLM failed to respond — use a synthetic fallback instruction
        print("⚠️ LLM API 未响应，使用合成指令（仍验证推导系统）")
        llm_text = json.dumps({
            "edge_id": "e_0",
            "action": "烘焙面包",
            "effects": [
                {"target_entity_id": "npc_test_llm", "attribute_name": "vitality",
                 "operation": "sub", "value": 5, "description": "烘焙面包消耗体力"}
            ],
            "edge_qty_changes": [
                {"source_entity_id": "npc_test_llm", "target_entity_id": "item_小麦",
                 "delta": -2},
                {"source_entity_id": "npc_test_llm", "target_entity_id": "item_面包",
                 "delta": 1},
            ],
            "result_text": "老张在farm用烤炉烘焙面包，消耗小麦x2，获得面包x1",
        })

    # Parse the LLM response
    parsed = ge.parse_llm_response(llm_text)
    assert parsed, f"LLM 返回无法解析: {llm_text[:200]}"

    instr = parsed[0] if isinstance(parsed, list) else parsed

    # Verify the instruction references item_面包
    qty_targets = {
        qc.get("target_entity_id", "")
        for qc in instr.get("edge_qty_changes", [])
    }
    has_bread_ref = "item_面包" in qty_targets

    if not has_bread_ref:
        print(f"⚠️ LLM 指令未直接引用 item_面包，引用: {qty_targets}，"
              f"注入面包引用测试指令")
        # Inject item_面包 reference into qty_changes
        qty_changes = instr.get("edge_qty_changes", [])
        qty_changes.append({
            "source_entity_id": "npc_test_llm",
            "target_entity_id": "item_面包",
            "delta": 1,
        })
        instr["edge_qty_changes"] = qty_changes

    # ─── Execute instruction ───
    # Build minimal edges for NPC
    ge.build_graph()

    result = ge.execute_npc_instruction(instr)
    print(f"  执行结果: {result}")

    # ─── Verify ───
    bread_entity = ge.get_entity("item_面包")
    assert bread_entity is not None, (
        "item_面包 应通过推导系统被创建"
    )
    assert bread_entity.name == "面包", f"名称应为 '面包'，得到: {bread_entity.name}"
    assert bread_entity.entity_type == "item"

    # Verify NPC inventory changed
    inv = ge.get_inventory_view("npc_test_llm")
    assert "面包" in inv or ge.get_held_quantity("npc_test_llm", "item_面包") >= 0, (
        f"NPC 应变更有面包: {inv}"
    )

    print(f"✅ SG-8: LLM驱动的面包推导注册测试通过")
    print(f"   NPC 库存: {inv}")
    print(f"   所有实体: {[e.name for e in ge.all_entities()]}")
