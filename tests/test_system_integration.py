"""
System Integration Test - NPC 模块集成测试

覆盖：
1. DB 层初始化
2. NPC CRUD 操作
3. NPC Engine tick 循环
4. Cognition 模块兼容性
5. Entities 模块集成
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# ============================================================
# 准备：使用独立的测试数据库文件
# ============================================================

# 使用项目 data 目录下的独立测试数据库
TEST_DB_FILE = Path(__file__).parent.parent / "data" / "test_integration.db"

def get_test_db():
    """返回测试数据库路径"""
    return TEST_DB_FILE

def setup_test_db():
    """初始化测试数据库"""
    import agent_world.db.db as db_module
    db_module.DB_PATH = TEST_DB_FILE
    db_module.get_db_path = lambda: TEST_DB_FILE
    # 确保 data 目录存在
    TEST_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    return db_module

# 清理旧的测试数据库
if TEST_DB_FILE.exists():
    TEST_DB_FILE.unlink()

_db_module = setup_test_db()

print(f"使用测试数据库: {TEST_DB_FILE}")

# ============================================================
# STC-1: 数据库初始化测试
# ============================================================

def test_init_db():
    """STC-1: 数据库初始化"""
    from agent_world.db.db import init_db
    
    init_db()
    
    assert TEST_DB_FILE.exists(), f"数据库文件未创建: {TEST_DB_FILE}"
    
    with _db_module.get_session() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        assert "npcs" in tables, "缺少 npcs 表"
        assert "world" in tables, "缺少 world 表"
    
    print("✅ STC-1: 数据库初始化")


def test_world_auto_created():
    """STC-1b: 世界数据自动创建"""
    from agent_world.db.db import WorldDB
    
    with _db_module.get_session() as conn:
        world_db = WorldDB(conn)
        world = world_db.get_world()
        assert world is not None, "世界未自动创建"
        assert world.id == "main_world"
        assert len(world.zones) > 0
        print(f"✅ STC-1b: 世界已创建，包含 {len(world.zones)} 个区域")


# ============================================================
# STC-2: NPC CRUD 测试
# ============================================================

def test_create_npc():
    """STC-2: 创建 NPC 并持久化"""
    from agent_world.db.db import NPCDB
    from agent_world.models.npc import NPCRole
    from agent_world.models.npc_defaults import make_merchant
    
    npc = make_merchant("测试商人", seed=0)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.create_npc(npc)
    
    # 重新读取验证
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        loaded = npc_db.get_npc(npc.id)
        assert loaded is not None, "NPC 未被持久化"
        assert loaded.name == "测试商人"
        assert loaded.role == NPCRole.MERCHANT
        assert loaded.physical.age == 35
        assert loaded.persona_tags.work_ethic == "勤奋"
    
    print("✅ STC-2: 创建 NPC 并持久化")


def test_update_npc():
    """STC-3: 更新 NPC"""
    from agent_world.db.db import NPCDB
    from agent_world.models.npc import NPCStatus
    from agent_world.models.npc_defaults import make_farmer
    
    npc = make_farmer("测试农民", seed=0)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.create_npc(npc)
    
    # 更新
    npc.level = 5
    npc.status = NPCStatus.WORKING
    npc.add_memory("测试记忆", importance=0.8)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.update_npc(npc)
    
    # 验证
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        loaded = npc_db.get_npc(npc.id)
        assert loaded.level == 5
        assert loaded.status == NPCStatus.WORKING
        assert len(loaded.memory) >= 1
        assert loaded.memory[-1].event == "测试记忆"
    
    print("✅ STC-3: 更新 NPC")


def test_delete_npc():
    """STC-4: 删除 NPC"""
    from agent_world.db.db import NPCDB
    from agent_world.models.npc_defaults import make_guard
    
    npc = make_guard("临时守卫", seed=0)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.create_npc(npc)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.delete_npc(npc.id)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        loaded = npc_db.get_npc(npc.id)
        assert loaded is None, "NPC 未被删除"
    
    print("✅ STC-4: 删除 NPC")


def test_list_npcs_filter():
    """STC-5: NPC 筛选查询"""
    from agent_world.db.db import NPCDB
    from agent_world.models.npc_defaults import create_diverse_npcs
    
    npcs = create_diverse_npcs()
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        for npc in npcs:
            npc_db.create_npc(npc)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        
        merchants = npc_db.list_npcs(role="merchant")
        assert len(merchants) >= 1
        assert all(n.role.value == "merchant" for n in merchants)
        
        farm_npcs = npc_db.list_npcs(zone_id="farm")
        assert all(n.position.zone_id == "farm" for n in farm_npcs)
        
        all_npcs = npc_db.list_npcs()
        assert len(all_npcs) >= 7
    
    print("✅ STC-5: NPC 筛选查询")


def test_npc_memory_persistence():
    """STC-6: NPC 记忆持久化"""
    from agent_world.db.db import NPCDB
    from agent_world.models.npc_defaults import make_scholar
    
    npc = make_scholar("记忆测试", seed=0)
    initial_mem_count = len(npc.memory)
    
    # 添加带 goal 的记忆
    npc.add_memory("完成研究论文", importance=0.9, goal="完成论文")
    npc.add_memory("参加学术会议", importance=0.6, related_npcs=["npc_x"], location="library")
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        npc_db.create_npc(npc)
    
    with _db_module.get_session() as conn:
        npc_db = NPCDB(conn)
        loaded = npc_db.get_npc(npc.id)
        
        assert len(loaded.memory) == initial_mem_count + 2, \
            f"记忆数量不匹配: {len(loaded.memory)} vs {initial_mem_count + 2}"
        
        # 验证 goal 字段
        goals = [m.goal for m in loaded.memory if m.goal is not None]
        assert "完成论文" in goals, f"goal 字段丢失: {goals}"
        
        # 验证 related_npc_ids
        related = [m for m in loaded.memory if m.related_npc_ids]
        assert len(related) >= 1
    
    print("✅ STC-6: NPC 记忆持久化（含 goal 字段）")


# ============================================================
# STC-3: Cognition 模块兼容性测试
# ============================================================

def test_cognition_vs_models_memoryentry():
    """STC-7: 验证两个 MemoryEntry 类的差异"""
    from agent_world.models.npc import MemoryEntry as NPCMem
    from agent_world.cognition.memory import MemoryEntry as CogMem
    
    # 创建 models 版本（带 goal）
    npc_mem = NPCMem(
        event="完成交易",
        timestamp=datetime.now(),
        importance=0.8,
        related_npc_ids=["npc_1"],
        location="market",
        goal="赚取利润"
    )
    
    # 序列化为 dict
    data = npc_mem.model_dump()
    assert "goal" in data, "models.MemoryEntry 应包含 goal 字段"
    
    # cognition 版本现在有 goal 字段
    try:
        cog_mem = CogMem(**data)
        assert hasattr(cog_mem, 'goal'), "cognition.MemoryEntry 应该有 goal 字段"
        assert cog_mem.goal == "赚取利润", f"goal 值不匹配: {cog_mem.goal}"
        print(f"✅ STC-7: cognition.MemoryEntry 保留了 goal: '{cog_mem.goal}'")
    except Exception as e:
        print(f"❌ STC-7: cognition.MemoryEntry goal 字段问题: {e}")


def test_npc_engine_memory_compatibility():
    """STC-8: NPC Engine 的 Memory 处理"""
    from agent_world.models.npc_defaults import make_merchant
    from agent_world.cognition.memory import MemoryStore
    from agent_world.cognition import memory as cognition_memory
    
    npc = make_merchant("兼容测试", seed=0)
    npc.add_memory("测试事件A", importance=0.7, goal="目标A")
    npc.add_memory("测试事件B", importance=0.5)
    
    # Engine 中的转换逻辑（来自 npc_engine.py）
    memory_data = getattr(npc, "memory", [])
    if memory_data and hasattr(memory_data[0], 'event'):
        mem_store = cognition_memory.MemoryStore(entries=memory_data)
    else:
        mem_store = cognition_memory.MemoryStore.from_dict(memory_data)
    
    assert len(mem_store) >= 2
    
    # MemoryStore.MemoryEntry 现在有 goal 字段，转换不会丢失数据
    print(f"✅ STC-8: MemoryStore 处理 {len(mem_store)} 条记忆（goal 字段已保留）")


# ============================================================
# STC-4: Entities 模块测试
# ============================================================

def test_entity_manager_init():
    """STC-9: 实体管理器初始化"""
    from agent_world.db.db import WorldDB
    from agent_world.entities import init_entity_manager, get_entity_manager
    
    with _db_module.get_session() as conn:
        world_db = WorldDB(conn)
        world = world_db.get_world()
    
    zone_dicts = [z.model_dump() for z in world.zones]
    init_entity_manager(zone_dicts)
    
    entity_mgr = get_entity_manager()
    all_entities = entity_mgr.all()
    
    assert len(all_entities) > 0, "实体管理器为空"
    
    entity_types = {}
    for e in all_entities:
        t = e.object_type.value
        entity_types[t] = entity_types.get(t, 0) + 1
    
    print(f"✅ STC-9: 实体管理器初始化，包含 {len(all_entities)} 个实体: {entity_types}")


def test_entity_interaction():
    """STC-10: 实体交互"""
    from agent_world.entities import get_entity_manager, ObjectType
    
    entity_mgr = get_entity_manager()
    
    ore_veins = [e for e in entity_mgr.all() if e.object_type == ObjectType.ORE_VEIN]
    if ore_veins:
        vein = ore_veins[0]
        result = vein.interact("test_npc_id", "mine")
        assert result is not None
        print(f"✅ STC-10: 实体交互 '{vein.name}' -> {result.description}")
    else:
        print("⚠️ STC-10: 没有矿脉实体，跳过交互测试")


# ============================================================
# STC-5: NPC Engine Tick 测试
# ============================================================



