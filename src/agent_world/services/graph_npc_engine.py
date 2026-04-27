"""
图引擎 NPC 服务 —— 替代旧的 NPCEngine。
每 tick：从现实世界加载实体 → 构建交互图 → LLM/兜底推导 → 执行副效应 → 同步回模型。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable

from ..db import NPCDB, get_session
from ..entities.manager import get_entity_manager, init_entity_manager
from ..models.npc import NPC, Position
from ..models.world import World, Zone

from ..cognition.npc_prompt_builder import build_one_npc_prompt

from .graph_engine import GraphEngine
from .graph_adapter import build_world_graph
from .interaction_resolver import InteractionResolver
from .intent_executor import IntentResolver, IntentExecutor
from .post_processor import PostProcessor, apply_updates, reset_applied_ops
from .conservation_validator import ConservationValidator, ValidationResult

logger = logging.getLogger("graph_npc_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)


class GraphNPCEngine:
    """
    基于交互图的 NPC 引擎。
    每 tick：加载世界状态 → 构建图 → 推导交互 → 执行 → 写回。
    """

    def __init__(self, llm_available: bool = False, llm_callback=None,
                 llm_model: str | None = None, llm_temperature: float = 0.7):
        self.llm_available = llm_available
        self.llm_callback = llm_callback
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature
        self._resolver: InteractionResolver | None = None
        self._listeners: list[Callable] = []
        self._running = False
        self.tick_count = 0
        self.graph_engine = GraphEngine()
        self._world_initialized = False
        self._ownership_map: dict[str, str] = {}  # obj_eid -> owner_neid

    def add_listener(self, listener: Callable):
        self._listeners.append(listener)

    async def _notify(self, tick_results: list[dict]):
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(tick_results)
                else:
                    listener(tick_results)
            except Exception:
                pass

    def _ensure_world_initialized(self):
        """确保实体管理器已初始化 + 数据库中有 NPC（仅首次调用创建）"""
        if self._world_initialized:
            return
        zones = [
            {"id": "village_square", "zone_type": "village_square"},
            {"id": "farm", "zone_type": "farm"},
            {"id": "market", "zone_type": "market"},
            {"id": "tavern", "zone_type": "tavern"},
            {"id": "barracks", "zone_type": "barracks"},
            {"id": "library", "zone_type": "library"},
            {"id": "temple", "zone_type": "temple"},
            {"id": "forest", "zone_type": "forest"},
        ]
        init_entity_manager(zones)

        # 如果数据库为空，创建默认 NPC
        with get_session() as conn:
            npc_db = NPCDB(conn)
            existing = npc_db.get_all_npcs()
            if not existing:
                from ..models.npc_defaults import create_diverse_npcs
                default_npcs = create_diverse_npcs()
                for npc in default_npcs:
                    npc_db.create_npc(npc)
                logger.info(f"初始化 {len(default_npcs)} 个默认 NPC 到数据库")

        self._world_initialized = True

    def _init_resolver(self):
        """延迟初始化 InteractionResolver"""
        if self._resolver is not None:
            return
        try:
            self._resolver = InteractionResolver(
                model=self._llm_model,
                temperature=self._llm_temperature,
            )
            logger.info("InteractionResolver 初始化成功")
        except ValueError as e:
            logger.warning(f"InteractionResolver 初始化失败: {e}，退回到兜底模式")
            self.llm_available = False
        except Exception as e:
            logger.error(f"InteractionResolver 异常: {e}，退回到兜底模式")
            self.llm_available = False

    # ─── Tick 核心 ───

    async def tick(self) -> list[dict]:
        """执行一次交互图 tick"""
        self.tick_count += 1
        self._ensure_world_initialized()

        # 初始化 Recipe 注册表（首次 tick）
        from ..entities.recipe import RecipeRegistry
        if self.tick_count == 1:
            RecipeRegistry.init_defaults()
            logger.info(f"[Recipe] 已加载 {len(RecipeRegistry.get_all())} 个内置配方")

        with get_session() as conn:
            npc_db = NPCDB(conn)
            db_npcs = npc_db.get_all_npcs()

        if not db_npcs:
            return []

        # 1) 获取世界对象
        mgr = get_entity_manager()
        all_objects = mgr.all()

        # 2) 构建 Zone 列表
        zones = self._get_zones()

        # 3) 构建交互图
        entities = build_world_graph(db_npcs, all_objects, zones, mgr)
        self.graph_engine = GraphEngine()
        for ent in entities:
            self.graph_engine.register_entity(ent)

        self._init_ownership(db_npcs, all_objects)
        self._connect_topology(db_npcs, all_objects, zones)
        self.graph_engine.build_graph()

        # 4) 设置所有权边数量（有主物体 qty=1）
        self._sync_ownership_edges()

        # 5) 设置初始库存
        self._sync_inventory_from_npcs(db_npcs)

        # 5) 推导交互 → 执行
        if self.llm_available:
            self._init_resolver()
        results = await self._derive_and_execute(db_npcs)

        # 6) 写入内存 & 被动衰减
        self._apply_memories_and_decay(db_npcs, results)

        # 7) 同步回 NPC 模型
        self._sync_back_to_npcs(db_npcs)

        # 8) 写回数据库
        with get_session() as conn:
            npc_db = NPCDB(conn)
            for npc in db_npcs:
                npc_db.update_npc(npc)

        return results

    def _get_zones(self) -> list[Zone]:
        zone_defs = [
            ("village_square", ["farm", "market", "tavern", "barracks", "library", "temple"]),
            ("farm", ["village_square"]),
            ("market", ["village_square"]),
            ("tavern", ["village_square"]),
            ("barracks", ["village_square"]),
            ("library", ["village_square"]),
            ("temple", ["village_square"]),
            ("forest", ["village_square"]),
        ]
        return [Zone(id=zid, name=zid, zone_type=zid,
                     bounds={"min_x": 0, "min_y": 0, "max_x": 100, "max_y": 100},
                     connected_zones=conns) for zid, conns in zone_defs]

    # ─── 所有权管理 ───

    def _get_controlled_object_eids(self, objects, npcs) -> dict[str, str]:
        """
        返回 {obj_eid: owner_neid} —— 有主物体的所有者映射。
        目前来源：_ownership_map（由 _init_ownership 或 LLM 效果设置）。
        """
        result: dict[str, str] = {}
        for oeid, owner_neid in self._ownership_map.items():
            # 确保物体还存在
            for obj in objects:
                if f"obj_{obj.id[:8]}" == oeid:
                    result[oeid] = owner_neid
                    break
        return result

    def _init_ownership(self, npcs, objects):
        """
        初始化默认所有权（首次 tick 执行一次）。
        按 NPC 名称匹配谁控制什么区域的物体。
        不写死角色/身份——只是初始配置。
        """
        if self._ownership_map:
            return  # 已初始化

        # 初始配置：[npc_name, zone_id, object_type, object_index]
        # object_index: 该类型区物体按创建顺序的第几个（0-based）
        # object_index=-1 表示该区域所有该类型物体
        initial_owners: list[tuple[str, str, int]] = [
            ("老陈", "tavern", 0),      # 酒吧吧台（该区域第1个物体）
            ("铁匠王", "market", 0),    # 第1个市场摊位
            ("张大娘", "market", 1),    # 第2个市场摊位
            ("翠花", "forest", 2),      # 第3个森林物体（药草园）
        ]

        for owner_name, controlled_zone, obj_index in initial_owners:
            owner_neid = None
            for npc in npcs:
                if npc.name == owner_name:
                    owner_neid = f"npc_{npc.id[:8]}"
                    break
            if not owner_neid:
                continue

            zone_objs = [obj for obj in objects if obj.zone_id == controlled_zone]
            if obj_index >= len(zone_objs):
                continue
            obj = zone_objs[obj_index]
            oeid = f"obj_{obj.id[:8]}"
            self._ownership_map[oeid] = owner_neid
            logger.info(f"所有权初始化: {owner_name}({owner_neid[:12]}) 拥有 {obj.name}({oeid[:12]})")

    def _set_owner(self, obj_eid: str, owner_neid: str):
        """设置/变更一个物体的所有者（运行时由 LLM 效果触发）"""
        self._ownership_map[obj_eid] = owner_neid

    def _clear_owner(self, obj_eid: str):
        """清除物体所有者（变为无主）"""
        self._ownership_map.pop(obj_eid, None)

    def _connect_topology(self, npcs, objects, zones):
        """
        构建交互图拓扑。通用规则：
          - 区域之间互连（连通性）
          - 每个 NPC 连接到所有区域（移动用）
          - 每个 NPC 连接到所有物品种类（库存用）
          - NPC 之间：同区域的全连接（社交/买卖）
          - NPC → 物体：
            - 无主的物体 → 该区域的所有 NPC 都可交互
            - 有主的物体 → 只有所有者连接（通过持有边 qty>0 表达所有权）
        """
        # 1. 区域互连
        for z in zones:
            zeid = f"zone_{z.id}"
            for conn in getattr(z, 'connected_zones', []):
                self.graph_engine.connect(zeid, f"zone_{conn}")
            for obj in objects:
                if obj.zone_id == z.id:
                    self.graph_engine.connect(zeid, f"obj_{obj.id[:8]}")

        # 2. NPC ↔ 区域 / 物品
        npc_eid_by_id: dict[str, str] = {}
        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            npc_eid_by_id[npc.id] = neid
            for z in zones:
                self.graph_engine.connect(neid, f"zone_{z.id}")
            for item in ["item_小麦", "item_金币", "item_铁锭", "item_药水",
                          "item_蔬菜", "item_面包", "item_皮毛", "item_草药",
                          "item_纸张", "item_货物", "item_武器", "item_书籍",
                          "item_酒", "item_佛经"]:
                self.graph_engine.connect(neid, item)

        # 3. NPC ↔ NPC（同区域）
        zone_npc_map: dict[str, list[str]] = {}
        for npc in npcs:
            zone_npc_map.setdefault(npc.position.zone_id, []).append(npc.id)
        for zone_id, nids in zone_npc_map.items():
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    a = f"npc_{nids[i][:8]}"
                    b = f"npc_{nids[j][:8]}"
                    self.graph_engine.connect(a, b)
                    self.graph_engine.connect(b, a)

        # 4. 检测有主物体
        owner_of = self._get_controlled_object_eids(objects, npcs)

        # 5. NPC → 物体（区分有主/无主）
        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            for obj in objects:
                oeid = f"obj_{obj.id[:8]}"
                if obj.zone_id != npc.position.zone_id:
                    continue

                is_controlled = oeid in owner_of
                if is_controlled and owner_of[oeid] != neid:
                    # 有主且不是所有者：不连物体，由 NPC-to-NPC 解决
                    continue
                # 无主 / 所有者：连接
                self.graph_engine.connect(neid, oeid)

    def _sync_inventory_from_npcs(self, npcs):
        item_map = {"小麦": "item_小麦", "金币": "item_金币",
                     "铁锭": "item_铁锭", "药水": "item_药水",
                     "蔬菜": "item_蔬菜", "面包": "item_面包",
                     "皮毛": "item_皮毛", "草药": "item_草药",
                     "纸张": "item_纸张", "货物": "item_货物",
                     "武器": "item_武器", "书籍": "item_书籍",
                     "酒": "item_酒", "佛经": "item_佛经"}
        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            for inv_item in getattr(npc, 'inventory', []) or []:
                name = inv_item.name if hasattr(inv_item, 'name') else str(inv_item)[:8]
                if name in item_map:
                    qty = inv_item.quantity if hasattr(inv_item, 'quantity') else 1
                    self.graph_engine.modify_edge_quantity(neid, item_map[name], qty)

    def _sync_ownership_edges(self):
        """
        同步所有权边的数量。
        有主物体：owner 持有 qty=1；
        无主物体：保留 qty=0（所有权的默认值）。
        """
        for oeid, owner_neid in self._ownership_map.items():
            self.graph_engine.set_edge_quantity(owner_neid, oeid, 1)
            logger.info(f"所有权边: {owner_neid} ──[持有]──▸ {oeid} (qty=1)")

    # ─── 兜底决策 + 执行 ───

    ZONE_ACTIONS = {
        "farm": ("播下了种子", 10, [("item_小麦", 2)]),
        "market": ("摆摊交易", 10, [("item_金币", 2)]),
        "barracks": ("训练完毕", 15, [("item_铁锭", 1)]),
        "tavern": ("吃了顿饭休息片刻", 0, []),
        "library": ("阅读了相关书籍", 5, []),
        "temple": ("祈祷后感到心神安宁", 8, [("item_药水", 1)]),
        "forest": ("探索森林收集材料", 12, [("item_草药", 1), ("item_皮毛", 1)]),
        "village_square": ("在广场闲逛", 5, []),
    }

    async def _derive_and_execute(self, npcs) -> list[dict]:
        if self.llm_available and self._resolver:
            try:
                return await self._execute_llm_individual(npcs)
            except asyncio.TimeoutError:
                logger.warning("LLM 超时")
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
        return self._fallback_derive_and_execute(npcs)

    async def _execute_llm_individual(self, npcs) -> list[dict]:
        """
        每个 NPC 独立构建 prompt → LLM 独立推理 → 逐条执行。
        图只提供实体间关系信息，不参与推理逻辑。
        """
        # 1. 构建每个 NPC 的独立 prompt
        npc_prompts: list[tuple[str, str]] = []  # [(neid, prompt_str)]
        npc_info: dict[str, dict] = {}  # neid -> {name, model, entity, zone}

        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue

            # 库存
            inv = self.graph_engine.get_inventory_view(neid)

            # 最近经历（记忆）
            memories = []
            for m in (getattr(npc, 'memory', []) or []):
                event = m.event if hasattr(m, 'event') else str(m)[:50]
                ts = getattr(m, 'timestamp', None)
                loc = getattr(m, 'location', "")
                # 取最近 10 条
                memories.append({"event": event, "timestamp": ts, "location": loc})
            # 按时间倒序取前 10
            memories = memories[-10:]

            # 性格标签
            personality_tags = []
            for tag in (getattr(npc, 'persona_tags', []) or []):
                if hasattr(tag, 'tag'):
                    personality_tags.append(tag.tag)

            # 角色
            role = ent.get_attr("role") or ""

            # 同区域的其他 NPC（通过图遍历：NPC → zone → 同 zone 的其他 NPC）
            npc_zone_id = ent.get_attr("zone_id") or npc.position.zone_id
            zone_eid = f"zone_{npc_zone_id}"
            zone_ent = self.graph_engine.get_entity(zone_eid)
            zone_npcs = []
            if zone_ent:
                for connected_id in zone_ent.connected_entity_ids:
                    if connected_id == neid:
                        continue
                    other = self.graph_engine.get_entity(connected_id)
                    if other and other.entity_type == "npc":
                        other_role = other.get_attr("role") or "?"
                        zone_npcs.append({"name": other.name, "role": other_role})

            prompt = build_one_npc_prompt(
                npc_entity=ent,
                npc_name=npc.name,
                npc_role=role,
                memories=memories,
                personality_tags=personality_tags,
                inventory=inv,
                zone_npcs=zone_npcs,
            )

            # 记录 prompt 摘要
            inv_items = list(inv.keys()) if inv else []
            mem_count = len(memories)
            tag_count = len(personality_tags)
            logger.info(
                f"[PROMPT] {npc.name}({role}) @ {ent.get_attr('zone_id') or '?'} | "
                f"物品: {inv_items[:4]}{'...' if len(inv_items) > 4 else ''} | "
                f"记忆: {mem_count}条 | 标签: {tag_count}个"
            )

            npc_prompts.append((neid, prompt))
            npc_info[neid] = {
                "name": npc.name,
                "model": npc,
                "entity": ent,
                "zone": ent.get_attr("zone_id") or npc.position.zone_id,
                "zone_npcs": zone_npcs,
            }

        # 2. LLM 调用（合并为一个请求，但保持 NPC 独立推理）
        llm_results = await asyncio.wait_for(
            self._resolver.resolve_all_npcs_async(npc_prompts),
            timeout=300.0,
        )
        logger.info(f"LLM 返回 {len(llm_results)}/{len(npc_prompts)} 条指令")

        # 3. LLM #2: 自然语言 → 交互目标（只输出 interact_with，不做数据更新）
        #    IntentExecutor: 执行拓扑变更（区域移动、NPC/物体连接）
        #    LLM #3 PostProcessor: 根据执行结果生成属性/库存/记忆/关系更新
        results = []
        intent_resolver = IntentResolver(resolver=self._resolver)
        intent_exec = IntentExecutor(self.graph_engine)
        post_proc = PostProcessor(resolver=self._resolver)

        # 收集执行结果，先不做数据更新
        exec_results = []
        for neid, _ in npc_prompts:
            info = npc_info.get(neid, {})
            ent = info.get("entity")
            if not ent:
                continue

            npc_model = info["model"]
            raw_text = llm_results.get(neid)  # LLM #1 输出的是自然语言字符串

            if raw_text and isinstance(raw_text, str) and raw_text.strip():
                inv_before = self.graph_engine.get_inventory_view(neid)
                zone = info.get("zone", "?")
                npc_name = info.get("name", "?")
                npc_role = ent.get_attr("role") or "?"

                # 同区域的其他 NPC
                zone_npcs = info.get("zone_npcs", [])

                # LLM #2: 自然语言 → 交互目标（只输出 interact_with）
                intent = intent_resolver.resolve_intent(
                    npc_name=npc_name,
                    npc_role=npc_role,
                    current_zone=ent.get_attr("zone_id") or zone,
                    inventory=inv_before,
                    raw_text=raw_text,
                    nearby_npcs=zone_npcs,
                )

                # 执行拓扑（IntentExecutor 只做拓扑，返回执行结果）
                er = intent_exec.execute(
                    neid, npc_name, npc_role, intent, raw_intent=raw_text
                )
                exec_results.append({
                    "result": er.to_dict(),
                    "model": npc_model,
                    "zone_npcs": zone_npcs,
                    "inv_before": inv_before,
                })
            else:
                # LLM 无效/缺失 → 降级为快速兜底
                zone = info.get("zone", "?")
                vitality = ent.get_attr("vitality") or 100
                role = ent.get_attr("role") or ""
                quick = self._fallback_decide_and_exec(neid, npc_model, zone, vitality, role)
                zone_now = ent.get_attr("zone_id") or zone
                vitality_now = ent.get_attr("vitality") or 100
                inv_view = self.graph_engine.get_inventory_view(neid)
                results.append({
                    "npc_id": npc_model.id,
                    "npc_name": info["name"],
                    "zone": zone_now,
                    "action": "等待",
                    "action_text": quick.get("action_text", "兜底行动"),
                    "vitality": int(vitality_now),
                    "inventory": dict(inv_view),
                    "tick": self.tick_count,
                })
                logger.info(f"[FALLBACK] {info.get('name','?')} @ {zone} | 兜底")

        # 4. LLM #3 PostProcessor：根据执行结果生成数据更新
        #    先收集所有 PP 输出，再批量应用（去重）
        reset_applied_ops()
        all_pp_updates = []
        all_involved_models = {}

        for er_data in exec_results:
            er = er_data["result"]
            model = er_data["model"]
            npc_name = er["npc_name"]
            npc_eid = er["npc_eid"]
            ent = self.graph_engine.get_entity(npc_eid)
            if not ent:
                continue

            all_involved_models[npc_name] = model
            for nname in er.get("interacted_npcs", []):
                for nid, nfo in npc_info.items():
                    if nfo["name"] == nname:
                        all_involved_models[nname] = nfo["model"]

            current_state = {
                "vitality": ent.get_attr("vitality") or 100,
                "hunger": ent.get_attr("hunger") or 50,
                "mood": ent.get_attr("mood") or 50,
                "inventory": self.graph_engine.get_inventory_view(npc_eid),
            }

            updates = post_proc.process(
                npc_name=npc_name,
                npc_role=er["npc_role"],
                zone_id=er["zone_after"],
                raw_decision=er["raw_intent"],
                execution_result=er,
                current_state=current_state,
                nearby_npcs=er_data["zone_npcs"],
            )
            if updates:
                all_pp_updates.extend(updates)

        # 批量应用所有 PP 更新（去重由 apply_updates 内部处理）
        if all_pp_updates:
            # ConservationValidator：校验后 + 类型标记
            validator = ConservationValidator()
            v_out = validator.validate(all_pp_updates)
            if v_out.passed:
                logs = apply_updates(all_pp_updates, all_involved_models, self.graph_engine)
                for log in logs:
                    logger.info(log)
            else:
                logger.warning(f"[Engine] Conservation validation failed: {v_out.message}")
                for d in v_out.details:
                    logger.warning(f"[Engine]   {d}")

        # 构建结果行
        for er_data in exec_results:
            er = er_data["result"]
            model = er_data["model"]
            npc_eid = er["npc_eid"]
            ent = self.graph_engine.get_entity(npc_eid)
            if not ent:
                continue
            zone_now = ent.get_attr("zone_id") or "?"
            vitality_now = ent.get_attr("vitality") or 100
            inv_view = self.graph_engine.get_inventory_view(npc_eid)
            results.append({
                "npc_id": model.id,
                "npc_name": er["npc_name"],
                "zone": zone_now,
                "action": er.get("raw_intent", "")[:50],
                "action_text": er.get("narrative", ""),
                "vitality": int(vitality_now),
                "inventory": dict(inv_view),
                "tick": self.tick_count,
            })

        logger.info(f"逐 NPC 指令执行完成: {len(results)} 条")
        return results

    def _fallback_derive_and_execute(self, npcs) -> list[dict]:
        results = []
        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue
            zone_id = ent.get_attr("zone_id") or npc.position.zone_id
            vitality = ent.get_attr("vitality") or 100
            role = ent.get_attr("role") or ""
            result = self._fallback_decide_and_exec(neid, npc, zone_id, vitality, role)
            result["npc_id"] = npc.id
            result["npc_name"] = npc.name
            result["tick"] = self.tick_count
            results.append(result)
        return results

    def _fallback_decide_and_exec(self, neid, npc, zone_id, vitality, role) -> dict:
        result = {"zone": zone_id, "action": "等待中", "action_text": "等待中", "vitality": vitality, "inventory": {}}

        hunger = getattr(npc, 'hunger', 50)
        mood = getattr(npc, 'mood', 50)

        # 1) 体力低 → 休息
        if vitality < 30:
            wait_edge = self._find_edge(neid, "_wait")
            if wait_edge:
                self.graph_engine.execute_effects([{
                    "edge_id": wait_edge.edge_id,
                    "effects": [{"target_entity_id": neid, "attribute_name": "vitality",
                                 "operation": "add", "value": 20, "description": "休息恢复体力"}],
                    "result_text": "休息中"
                }])
                result["action"] = "休息中"
                result["action_text"] = "休息中"
                rest_ent = self.graph_engine.get_entity(neid)
                result["vitality"] = rest_ent.get_attr("vitality") if rest_ent else 0
                return result

        # 2) 饥饿 → 去 tavern
        if hunger > 60 and zone_id != "tavern":
            move_edge = self._find_edge_to_zone(neid, "tavern")
            if move_edge:
                self._exec_move(neid, move_edge, "tavern", result)
                return result

        # 3) 心情差 → 去广场社交
        if mood < 20 and zone_id != "village_square":
            move_edge = self._find_edge_to_zone(neid, "village_square")
            if move_edge:
                self._exec_move(neid, move_edge, "village_square", result)
                return result

        # 如果在 tavern
        if zone_id == "tavern":
            is_owner = neid in self._ownership_map.values()
            if is_owner:
                # 所有者：开关门 / 休息（只使用第一个 tavern 物体）
                all_edges = self._find_all_edges_to_zone(neid, "tavern")
                owner_edge = None
                for e in all_edges:
                    oeid = e.target_entity_id
                    if self._ownership_map.get(oeid) == neid:
                        owner_edge = e
                        break
                if not owner_edge and all_edges:
                    owner_edge = all_edges[0]
                if owner_edge:
                    bar_ent = self.graph_engine.get_entity(owner_edge.target_entity_id)
                    is_closed = bar_ent and bar_ent.get_attr("state") == "locked"
                    if is_closed:
                        self.graph_engine.execute_effects([{
                            "edge_id": owner_edge.edge_id,
                            "effects": [
                                {"target_entity_id": owner_edge.target_entity_id,
                                 "attribute_name": "state", "operation": "set",
                                 "value": "available", "description": "打开酒馆"},
                                {"target_entity_id": neid, "attribute_name": "vitality",
                                 "operation": "sub", "value": 2, "description": "开门"},
                            ],
                            "result_text": "打开酒馆开始营业"
                        }])
                        result["action"] = "打开酒馆"
                    else:
                        self.graph_engine.execute_effects([{
                            "edge_id": owner_edge.edge_id,
                            "effects": [
                                {"target_entity_id": neid, "attribute_name": "hunger",
                                 "operation": "sub", "value": 10,
                                 "description": "老板吃饭"},
                                {"target_entity_id": neid, "attribute_name": "mood",
                                 "operation": "add", "value": 5,
                                 "description": "休息提升心情"},
                            ],
                            "result_text": "在自家酒馆休息"
                        }])
                        result["action"] = "在自家酒馆休息"
                    result["vitality"] = self.graph_engine.get_entity(neid).get_attr("vitality") or 0
                    return result
            # 非所有者：不拦截，让逻辑继续走到工作区域步骤

        # 4) 移动到起始工作区域（ROLE_ZONE_MAP 只在初始化时用，此处直接用 NPC 起始区域）
        npc_start_zone = getattr(npc, 'position', None) and npc.position.zone_id
        target_zone = npc_start_zone or "village_square"
        if zone_id != target_zone:
            move_edge = self._find_edge_to_zone(neid, target_zone)
            if move_edge:
                self._exec_move(neid, move_edge, target_zone, result)
                return result

        # 5) 在当前区域工作
        if zone_id in self.ZONE_ACTIONS:
            interact_edge = self._find_edge_to_zone_object(neid, zone_id)
            if interact_edge:
                self._exec_work(neid, interact_edge, zone_id, result)
                return result

        # 6) 兜底休息
        wait_edge = self._find_edge(neid, "_wait")
        if wait_edge:
            self.graph_engine.execute_effects([{
                "edge_id": wait_edge.edge_id,
                "effects": [{"target_entity_id": neid, "attribute_name": "vitality",
                             "operation": "add", "value": 5, "description": "等待恢复少量体力"}],
                "result_text": "等待中(1)"
            }])
            result["action"] = "等待中"
            return result

        return result

    def _exec_move(self, neid, edge, target_zone, result):
        self.graph_engine.execute_effects([{
            "edge_id": edge.edge_id,
            "effects": [
                {"target_entity_id": neid, "attribute_name": "zone_id", "operation": "set",
                 "value": target_zone, "description": f"移动到{target_zone}"},
                {"target_entity_id": neid, "attribute_name": "vitality", "operation": "sub",
                 "value": 5, "description": "移动消耗体力"}
            ],
            "result_text": f"移动到了 {target_zone}"
        }])
        ent = self.graph_engine.get_entity(neid)
        result["zone"] = target_zone
        result["action"] = f"移动到了 {target_zone}"
        result["vitality"] = ent.get_attr("vitality") if ent else 100

    def _exec_work(self, neid, edge, zone_id, result):
        act_name, cost, items = self.ZONE_ACTIONS.get(zone_id, ("等待中", 3, []))
        effects = [{"target_entity_id": neid, "attribute_name": "vitality",
                    "operation": "sub", "value": cost, "description": "工作消耗体力"}]
        # tavern 特殊效果：减少饥饿，提升心情
        if zone_id == "tavern":
            effects.append({"target_entity_id": neid, "attribute_name": "hunger",
                           "operation": "sub", "value": 20, "description": "吃饭减少饥饿"})
            effects.append({"target_entity_id": neid, "attribute_name": "mood",
                           "operation": "add", "value": 10, "description": "吃饭提升心情"})
        instr = {
            "edge_id": edge.edge_id,
            "effects": effects,
            "edge_qty_changes": [
                {"source_entity_id": neid, "target_entity_id": item_id, "delta": qty}
                for item_id, qty in items
            ],
            "result_text": act_name
        }
        self.graph_engine.execute_effects([instr])
        ent = self.graph_engine.get_entity(neid)
        result["action"] = act_name
        result["vitality"] = ent.get_attr("vitality") if ent else 100
        result["inventory"] = dict(self.graph_engine.get_inventory_view(neid))

    def _find_edge(self, src_eid, suffix):
        for e in self.graph_engine.graph.edges:
            if e.source_entity_id == src_eid and e.source_interface_id.endswith(suffix):
                return e
        return None

    def _find_edge_to_zone(self, src_eid, zone_id):
        for e in self.graph_engine.graph.edges:
            if e.source_entity_id == src_eid \
               and e.source_interface_id.endswith("_move") \
               and e.target_entity_id == f"zone_{zone_id}":
                return e
        return None

    def _find_edge_to_zone_object(self, src_eid, zone_id):
        for e in self.graph_engine.graph.edges:
            if e.source_entity_id == src_eid \
               and e.source_interface_id.endswith("_interact") \
               and e.target_entity_id.startswith("obj_"):
                target = self.graph_engine.get_entity(e.target_entity_id)
                if target and target.get_attr("zone_id") == zone_id:
                    return e
        return None

    def _find_all_edges_to_zone(self, src_eid, zone_id):
        """查找 src 到指定区域所有物体的交互边"""
        results = []
        for e in self.graph_engine.graph.edges:
            if e.source_entity_id == src_eid \
               and e.source_interface_id.endswith("_interact") \
               and e.target_entity_id.startswith("obj_"):
                target = self.graph_engine.get_entity(e.target_entity_id)
                if target and target.get_attr("zone_id") == zone_id:
                    results.append(e)
        return results

    # ─── 记忆 & 衰减 ───

    def _apply_memories_and_decay(self, npcs, results):
        """为每个 NPC 写入记忆 + 被动衰减（饥饿↑ 心情↓）"""
        from datetime import datetime
        result_map = {r.get("npc_id"): r for r in results}

        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue

            # 被动衰减：每 tick 饥饿 +1，心情 -0.5
            hunger = ent.get_attr("hunger")
            mood = ent.get_attr("mood")
            if hunger is not None:
                ent.set_attr("hunger", min(100, hunger + 1))
            if mood is not None:
                ent.set_attr("mood", max(0, mood - 0.5))

            # 构建记忆
            r = result_map.get(npc.id, {})
            action = r.get("action", "等待中")
            zone = r.get("zone", npc.position.zone_id)
            inv = r.get("inventory", {})

            inv_desc = "、".join(f"{k}x{v}" for k, v in inv.items()) if inv else "空手"
            memory_text = f"在{zone} {action}，持有 {inv_desc}"

            # 添加记忆
            from ..models.npc import MemoryEntry
            npc.memory.append(MemoryEntry(
                event=memory_text,
                timestamp=datetime.now(),
                importance=0.3,
                related_npc_ids=[],
                location=zone,
            ))
            if len(npc.memory) > 20:
                npc.memory = npc.memory[-20:]

    # ─── 写回 ───

    def _sync_back_to_npcs(self, npcs):
        for npc in npcs:
            neid = f"npc_{npc.id[:8]}"
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue
            # 位置
            z = ent.get_attr("zone_id")
            if z:
                npc.position.zone_id = z
            # 体力
            v = ent.get_attr("vitality")
            if v is not None:
                npc.vitality = max(0, min(100, int(v)))
            # 饥饿 & 心情
            h = ent.get_attr("hunger")
            if h is not None:
                npc.hunger = max(0, min(100, int(h)))
            m = ent.get_attr("mood")
            if m is not None:
                npc.mood = max(0, min(100, int(m)))
            # 同步库存（NPC 模型使用 list[str]，按数量重复添加）
            inv_view = self.graph_engine.get_inventory_view(neid)
            if inv_view:
                new_inv = []
                for item_name, qty in inv_view.items():
                    new_inv.extend([item_name] * int(qty))
                npc.inventory = new_inv

    # ─── 运行循环 ───

    async def run(self):
        """启动 tick 主循环"""
        self._running = True
        self.tick_count = 0
        self._ensure_world_initialized()

        logger.info("GraphNPCEngine 启动 (每 10 分钟 LLM 更新, 游戏内 30 分钟/tick)")

        while self._running:
            try:
                tick_start = time.time()
                results = await self.tick()
                tick_end = time.time()

                if results:
                    # 只输出简短的每 NPC 一行结果
                    for r in results:
                        nm = r.get("npc_name", "?")
                        act = r.get("action", "")
                        vt = r.get("vitality", 0)
                        z = r.get("zone", "?")
                        logger.info(f"[Tick] {nm} | {act} @ {z} | vit={vt}")
                else:
                    logger.info("[Tick] 无NPC")

                await self._notify(results)
                elapsed = tick_end - tick_start
                await asyncio.sleep(max(0, 600 - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tick 异常: {e}")
                await asyncio.sleep(600)

        logger.info("GraphNPCEngine 停止")

    def stop(self):
        self._running = False
