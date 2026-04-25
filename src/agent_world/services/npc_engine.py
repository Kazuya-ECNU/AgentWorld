"""
NPC Engine v2 - Goal-Driven Behavior Engine

Phase 3.5c: 重构 NPC Engine，接 cognition (GoalReasoner) + entities (WorldObjectManager)

行为循环：
  1. 评估当前 Goal 是否完成/失败
  2. 若无 Goal 或已完成 → 调用 GoalReasoner 推理新 Goal
  3. 执行当前 Plan 的下一步
  4. 若需要移动 → 移动到目标 Zone
  5. 若需要交互物体 → 调用 entity.interact()
  6. 记录记忆，更新状态
  7. 广播结果
"""

import asyncio
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_world.db import get_session, NPCDB, WorldDB
from agent_world.models.npc import NPC, NPCRole, NPCStatus
from agent_world.models.world import World, Zone, DEFAULT_ZONES
from agent_world.cognition import (
    PersonaTags,
    MemoryStore,
    MemoryEntry,
    MemoryManager,
    ContextBuilder,
    GoalReasoner,
    GoalOutput,
    FallbackEngine,
    memory as cognition_memory,
)
from agent_world.entities import (
    ObjectType,
    Stall,
    FarmPlot,
    OreVein,
    BarCounter,
)


# === Goal → Action 映射 ===

GOAL_TO_OBJECTS = {
    "trade": ObjectType.STALL,
    "farm": ObjectType.FARM_PLOT,
    "mine": ObjectType.ORE_VEIN,
    "rest": ObjectType.BAR_COUNTER,  # 酒馆吧台可以休息
    "socialize": ObjectType.BAR_COUNTER,  # 酒馆可以社交
}

GOAL_TO_STATUS = {
    "trade": NPCStatus.WORKING,
    "farm": NPCStatus.WORKING,
    "mine": NPCStatus.WORKING,
    "rest": NPCStatus.RESTING,
    "socialize": NPCStatus.SOCIALIZING,
    "idle": NPCStatus.IDLE,
    "explore": NPCStatus.TRAVELING,
    "work": NPCStatus.WORKING,
}


# === NPC State 扩展 ===

class NPCState:
    """
    NPC 的运行时状态（内存中，非持久化）。
    
    包含 Goal 进度、Plan 索引等运行时数据。
    """
    def __init__(self):
        self.current_goal: GoalOutput | None = None
        self.current_plan: list[str] = []
        self.plan_index: int = 0
        self.goal_cooldown: int = 0  # tick 冷却，避免频繁重新推理
        self.last_goal_reason: str = ""


# === NPC Engine 主类 ===

class NPCEngine:
    """
    Goal-Driven NPC 行为引擎。
    
    核心循环：
      tick() → 每个 NPC 执行一步
        1. 检查 goal 状态
        2. 必要时调用 reasoner 生成新 goal
        3. 执行 plan 当前步骤
        4. 更新 NPC 状态 + 记忆
    """

    def __init__(
        self,
        llm_available: bool = True,
        tick_interval: float = 5.0,
    ):
        self.tick_interval = tick_interval
        self.running = False
        self._listeners: list = []
        self._npc_states: dict[str, NPCState] = {}  # npc_id → NPCState

        # 认知模块
        self.llm_available = llm_available
        self.reasoner = GoalReasoner() if llm_available else None
        self.fallback = FallbackEngine()

        # 实体模块（使用共享管理器）
        from agent_world.entities import get_entity_manager
        self.entity_manager = get_entity_manager()

        # 记忆管理器
        self.memory_manager = MemoryManager()

    def add_listener(self, callback):
        """添加 tick 回调（用于 WebSocket 广播）"""
        self._listeners.append(callback)

    # === Goal 推理 ===

    def _infer_goal(self, npc: NPC, world: World) -> GoalOutput:
        """
        为 NPC 推理新的 Goal。
        
        优先使用 LLM reasoner，失败则用 fallback 规则引擎。
        """
        # 构建 NPC 状态字典（给 fallback 用）
        state_dict = {
            "energy": npc.vitality if hasattr(npc, 'vitality') else getattr(npc, "energy", 80),
            "inventory": list(npc.inventory),
            "position": npc.position.zone_id,
            "role": npc.role.value,
            "time_of_day": world.get_time_of_day() if hasattr(world, 'get_time_of_day') else "day",
            "is_night": world.is_night() if hasattr(world, 'is_night') else False,
        }

        # 尝试 LLM 推理
        if self.reasoner:
            try:
                # 构建上下文
                from agent_world.cognition import PersonaTags, MemoryStore, ContextBuilder

                persona_data = getattr(npc, "persona_tags", None)
                persona = PersonaTags.from_dict(persona_data) if persona_data else PersonaTags()

                memory_data = getattr(npc, "memory", [])
                if memory_data and hasattr(memory_data[0], 'event'):
                    # Already MemoryEntry objects (from models/npc.py)
                    mem_store = cognition_memory.MemoryStore(entries=memory_data)
                else:
                    # Raw dict list
                    mem_store = cognition_memory.MemoryStore.from_dict(memory_data) if memory_data else cognition_memory.MemoryStore()

                ctx_builder = ContextBuilder(
                    name=npc.name,
                    role=npc.role.value,
                    persona=persona,
                    memory=mem_store,
                    recent_context_n=5,
                )

                # 获取世界认知（NPC 熟悉的区域）
                known_zones = [z.id for z in world.zones]
                known_npcs = []  # 后续从关系系统中获取

                prompt = ctx_builder.format_for_llm(
                    known_zones=known_zones,
                    known_npcs=known_npcs,
                    energy=state_dict["energy"],
                    inventory=state_dict["inventory"],
                    position=state_dict["position"],
                )

                goal = self.reasoner.reason_for_npc(prompt)
                if goal:
                    return goal
            except Exception as e:
                print(f"[NPCEngine] LLM 推理失败: {e}")

        # Fallback 规则引擎
        return self.fallback.resolve(npc.role.value, state_dict)

    # === Plan 执行 ===

    def _execute_plan_step(
        self,
        npc: NPC,
        plan_step: str,
        world: World,
        current_goal: GoalOutput | None = None,
    ) -> tuple[str, str]:
        """
        执行单个 Plan 步骤。
        
        Returns:
            (description, memory_event)
        """
        step = plan_step.strip().lower()

        # === 移动相关 ===
        if any(kw in step for kw in ["移动到", "前往", "走到", "出发去"]):
            # 提取目标 zone
            zone_name = step.split("到")[-1].strip() if "到" in step else ""
            target_zone = self._find_zone_by_name(world, zone_name)
            if target_zone:
                npc.position.zone_id = target_zone.id
                npc.position.x = (target_zone.bounds["min_x"] + target_zone.bounds["max_x"]) / 2
                npc.position.y = (target_zone.bounds["min_y"] + target_zone.bounds["max_y"]) / 2
                return f"移动到了 {target_zone.name}", f"移动到了 {target_zone.name}"

        # === 交易 ===
        if "交易" in step or "摆摊" in step:
            obj_type = GOAL_TO_OBJECTS.get(current_goal.goal if current_goal else "")
            if obj_type:
                obj = self.entity_manager.find_nearest_available(
                    from_zone_id=npc.position.zone_id,
                    object_type=obj_type,
                    zone_connections={z.id: z.connected_zones for z in world.zones},
                )
                if obj:
                    result = obj.interact(npc.id, "trade")
                    if result.success:
                        npc.inventory.extend(result.loot)
                        return result.description, f"在 {obj.name} 交易"

        # === 挖矿 ===
        if "挖" in step or "采矿" in step:
            obj = self.entity_manager.find_nearest_available(
                from_zone_id=npc.position.zone_id,
                object_type=ObjectType.ORE_VEIN,
                zone_connections={z.id: z.connected_zones for z in world.zones},
            )
            if obj:
                result = obj.interact(npc.id, "mine")
                if result.success:
                    npc.inventory.extend(result.loot)
                    return result.description, f"在 {obj.name} 挖掘获得 {result.loot}"
                elif result.state_change and "depleted" in result.state_change.value:
                    return f"{obj.name} 已枯竭", f"发现 {obj.name} 已枯竭"

        # === 农业 ===
        if "种" in step or "农" in step or "农场" in step:
            farm = self.entity_manager.find_nearest_available(
                from_zone_id=npc.position.zone_id,
                object_type=ObjectType.FARM_PLOT,
                zone_connections={z.id: z.connected_zones for z in world.zones},
            )
            if farm:
                action = "plant" if hasattr(farm, "growth_state") and farm.growth_state == farm.FALLOW else "harvest"
                result = farm.interact(npc.id, action)
                if result.success:
                    npc.inventory.extend(result.loot)
                    return result.description, f"在 {farm.name} 进行了 {action}"

        # === 休息 ===
        if "休息" in step or "睡觉" in step:
            tavern = self.entity_manager.find_nearest_available(
                from_zone_id=npc.position.zone_id,
                object_type=ObjectType.BAR_COUNTER,
                zone_connections={z.id: z.connected_zones for z in world.zones},
            )
            if tavern:
                result = tavern.interact(npc.id, "drink")
                if result.success:
                    return result.description, "在酒馆休息"
            return "休息了一会儿", "休息了一下"

        # === 社交 ===
        if "社交" in step or "聊天" in step:
            return "和其他人聊了聊天", "和其他 NPC 进行了社交"

        # === 探索 ===
        if "探索" in step or "闲逛" in step:
            connected = self._get_connected_zone(world, npc.position.zone_id)
            if connected:
                npc.position.zone_id = connected.id
                npc.position.x = (connected.bounds["min_x"] + connected.bounds["max_x"]) / 2
                npc.position.y = (connected.bounds["min_y"] + connected.bounds["max_y"]) / 2
                return f"探索了 {connected.name}", f"探索了 {connected.name}"

        # === 通用工作 ===
        if "工作" in step or "忙碌" in step:
            return "正在工作中", None

        return step, None

    # === Zone 辅助 ===

    def _find_zone_by_name(self, world: World, name: str) -> Zone | None:
        for zone in world.zones:
            if name in zone.name or name in zone.id:
                return zone
        return None

    def _get_connected_zone(self, world: World, current_zone_id: str) -> Zone | None:
        for z in world.zones:
            if z.id == current_zone_id and z.connected_zones:
                target_id = random.choice(z.connected_zones)
                for tz in world.zones:
                    if tz.id == target_id:
                        return tz
        return None

    # === 主 Tick 循环 ===

    async def tick(self):
        """执行一次 tick"""
        print(f"[TICK] Starting tick at {datetime.now().isoformat()}")
        with get_session() as conn:
            npc_db = NPCDB(conn)
            world_db = WorldDB(conn)

            world = world_db.get_world()
            if not world:
                return []

            npcs = npc_db.get_all_npcs()

            # 如果实体管理器为空，用 world zones 初始化
            if not self.entity_manager.all():
                from agent_world.entities import init_entity_manager
                zone_dicts = [z.model_dump() for z in world.zones]
                init_entity_manager(zone_dicts)
            
            # 全局实体 tick（农田生长等）
            self.entity_manager.tick()

            results = []

            for npc in npcs:
                npc_state = self._npc_states.get(npc.id, NPCState())
                self._npc_states[npc.id] = npc_state

                # 冷却递减
                if npc_state.goal_cooldown > 0:
                    npc_state.goal_cooldown -= 1

                description = ""
                memory_event = None

                # === 决策：是否需要推理新 Goal ===
                # plan 执行完毕（exhausted）且 cooldown 已过，才重新推理
                plan_exhausted = npc_state.plan_index >= len(npc_state.current_plan)
                if plan_exhausted and npc_state.goal_cooldown <= 0:
                    # 需要新 goal 且可以推理
                    old_goal = npc_state.current_goal
                    npc_state.current_goal = self._infer_goal(npc, world)
                    npc_state.last_goal_reason = npc_state.current_goal.reason
                    npc_state.current_plan = npc_state.current_goal.plan or [npc_state.current_goal.goal]
                    npc_state.plan_index = 0
                    npc_state.goal_cooldown = 3
                    memory_event = None  # 不记录 goal 推理记忆，只记录执行结果
                elif plan_exhausted:
                    # plan 执行完毕但 cooldown 未过，保持 idle
                    description = "四处观望"
                    npc_state.current_goal = None
                    memory_event = None
                
                # === 执行当前 Plan 步骤 ===
                _goal = npc_state.current_goal
                _idx = npc_state.plan_index
                _plan = npc_state.current_plan
                _exec_cond = _goal is not None and _idx < len(_plan)
                
                if _exec_cond:
                    step = _plan[_idx]
                    description, mem_ev = self._execute_plan_step(npc, step, world, current_goal=_goal)
                    memory_event = f"执行步骤{_idx}:{step} 结果:{description}"
                    npc_state.plan_index += 1
                    
                    # 如果是交互步骤（step index >= 1），说明已到达目标区域
                    # 交互成功后设置较长冷却，避免反复移动
                    if _idx >= 1:
                        npc_state.goal_cooldown = 15  # 15 ticks 冷却，期间留在原地重复交互
                        npc_state.current_goal = None  # 留在原地，但不复位 plan_index
                        memory_event = None  # 不记录每次交互，减少噪音
                else:
                    # plan 执行完毕或无 goal，idle
                    description = "四处观望"
                    npc_state.current_goal = None
                    memory_event = None

                # 更新 NPC 状态
                if npc_state.current_goal:
                    goal_type = npc_state.current_goal.goal
                    npc.status = GOAL_TO_STATUS.get(goal_type, NPCStatus.IDLE)
                else:
                    npc.status = NPCStatus.IDLE

                # 记录记忆
                if memory_event:
                    importance = npc_state.current_goal.urgency if npc_state.current_goal else 0.3
                    npc.add_memory(memory_event, importance=importance)

                npc.updated_at = datetime.now()
                npc_db.update_npc(npc)

                results.append({
                    "id": npc.id,
                    "name": npc.name,
                    "status": npc.status.value,
                    "position": npc.position.model_dump(),
                    "action": description,
                    "goal": npc_state.current_goal.goal if npc_state.current_goal else "idle",
                    "goal_reason": npc_state.last_goal_reason,
                    "vitality": npc.vitality if hasattr(npc, 'vitality') else 100.0,
                    "inventory": list(npc.inventory),
                    "memory_count": len(npc.memory),
                })

                # 限制 inventory
                if len(npc.inventory) > 20:
                    npc.inventory = npc.inventory[-20:]
                    npc_db.update_npc(npc)

            # 推进世界时间
            world.world_time.tick(1)
            world.active_npcs = len(npcs)
            world_db.save_world(world)

            return results

    async def run(self):
        """启动引擎主循环"""
        self.running = True
        print(f"[NPC Engine v2] 启动，每 {self.tick_interval}s tick 一次")
        print(f"  LLM 推理: {'启用' if self.llm_available else '禁用（使用规则引擎）'}")
        while self.running:
            try:
                results = await self.tick()
                for listener in self._listeners:
                    try:
                        await listener(results)
                    except Exception as e:
                        print(f"[NPC Engine] 广播错误: {e}")
                await asyncio.sleep(self.tick_interval)
            except Exception as e:
                print(f"[NPC Engine] Tick 错误: {e}")
                await asyncio.sleep(self.tick_interval)

    def stop(self):
        self.running = False
        print("[NPC Engine v2] 已停止")
