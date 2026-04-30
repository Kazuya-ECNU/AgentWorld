"""
图引擎 NPC 服务 —— 4-LLM 纯拓扑流水线

每 tick 流程：
  1. 从数据库加载 NPC → 构建纯拓扑图（无接口边）
  2. LLM #1: 拓扑子图 → NPC 自然语言计划
  3. LLM #2: 计划 + 拓扑 → 拓扑结构变更（connect/disconnect/set_qty）
  4. IntentExecutor: 执行拓扑结构变更 → exec_results
  5. LLM #3 (InteractionLayer): exec_results + 拓扑 → 故事文本
  6. LLM #4 (PostProcessor): 故事 + 拓扑 → 数值增量（{src, tgt, delta}）
  7. GraphEngine: 执行数值增量 → 拓扑更新
  8. 同步回 NPC 模型 → 写回数据库
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable

from ..db import NPCDB, WorldDB, get_session
from ..entities.manager import get_entity_manager, init_entity_manager
from ..models.npc import NPC, Position
from ..models.world import World, Zone

from ..cognition.npc_prompt_builder import build_one_npc_prompt

from .graph_engine import GraphEngine
from .graph_adapter import build_world_graph, _make_eid
from .interaction_resolver import InteractionResolver
from .intent_executor import IntentResolver
from .post_processor import PostProcessor
from .interaction_layer import InteractionLayer


# ─── 数值→文字辅助（供 LLM #3 叙事使用） ───

def _val_to_mood_text(val):
    if val is None:
        return "未知"
    if val < 30:
        return "很低落"
    if val < 50:
        return "有点低落"
    if val < 70:
        return "一般"
    return "不错"

def _val_to_sat_text(val):
    if val is None:
        return "未知"
    if val < 30:
        return "很饿"
    if val < 50:
        return "有点饿"
    if val < 70:
        return "还行"
    return "吃饱了"

def _val_to_vit_text(val):
    if val is None:
        return "未知"
    if val < 30:
        return "很疲惫"
    if val < 50:
        return "有些累"
    if val < 70:
        return "还行"
    return "精力充沛"


logger = logging.getLogger("graph_npc_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)


class GraphNPCEngine:
    """
    基于 4-LLM 纯拓扑流水线的 NPC 引擎。

    设计原则：
      - LLM 从不直接写入数据
      - GraphEngine 是唯一的数据写入路径
      - LLM #2 输出拓扑结构变更，LLM #4 输出数值增量
      - 边无类型标签，语义从节点描述推断
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
        self._current_world_time_str = ""
        self._current_time_of_day = ""
        self._tick_duration_str = ""

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
        """确保实体管理器已初始化 + 数据库中有 NPC"""
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

    # ═══════════════════════════════════════════
    # Tick 核心
    # ═══════════════════════════════════════════

    async def tick(self) -> list[dict]:
        """执行一次交互图 tick"""
        self.tick_count += 1
        self._ensure_world_initialized()

        with get_session() as conn:
            npc_db = NPCDB(conn)
            db_npcs = npc_db.get_all_npcs()

            world_db = WorldDB(conn)
            world = world_db.get_world()
            if world:
                # 动态时间加速：21:00-06:00 夜间每 tick 跳 6 小时
                h = world.world_time.hour
                if h >= 21 or h < 6:
                    tick_minutes = 360
                    tick_duration_label = "6 小时"
                else:
                    tick_minutes = 30
                    tick_duration_label = "30 分钟"
                world.world_time.tick(minutes=tick_minutes)
                world_db.save_world(world)
                self._current_world_time_str = world.world_time.to_display_str()
                self._current_time_of_day = world.world_time.get_time_of_day()
                self._tick_duration_str = tick_duration_label
                logger.info(f"[WorldTime] → {self._current_world_time_str} ({self._current_time_of_day}) +{tick_duration_label}")

        if not db_npcs:
            return []

        # 1. 获取世界对象和区域
        mgr = get_entity_manager()
        all_objects = mgr.all()
        zones = self._get_zones()

        # 2. 从头构建纯拓扑图（无接口边）
        entities = build_world_graph(db_npcs, all_objects, zones, mgr)
        self.graph_engine = GraphEngine()
        for ent in entities.values():
            self.graph_engine.register_entity(ent)

        # 3. 创建初始拓扑边（库存 + 区域 + 区域互联）
        from .graph_adapter import init_graph_edges_from_adapter
        init_graph_edges_from_adapter(self.graph_engine, db_npcs, zones)

        # 4. 4-LLM 流水线
        if self.llm_available:
            self._init_resolver()
        results = await self._execute_4llm_pipeline(db_npcs)

        # 5. 被动衰减 + 记忆
        self._decay_and_sync(db_npcs)

        # 6. 同步回 NPC 模型
        self._sync_back_to_nodes(db_npcs)

        # 7. 写回数据库
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



    # ═══════════════════════════════════════════
    # 4-LLM 流水线
    # ═══════════════════════════════════════════

    async def _execute_4llm_pipeline(self, npcs) -> list[dict]:
        """
        4-LLM 流水线主流程。

        每个阶段独立：
          LLM #1 — 读拓扑 → 写计划（自然语言）
          LLM #2 — 读计划 + 拓扑 → 写拓扑结构变更
          IntentExecutor — 执行结构变更
          LLM #3 — 读更新后拓扑 + 计划 → 写故事（纯自然语言）
          LLM #4 — 读故事 + 拓扑 → 写数值增量
          GraphEngine — 执行数值增量
        """
        if not self.llm_available or not self._resolver:
            raise RuntimeError("LLM 不可用 — 引擎停止")

        # ─── Step 1: LLM #1 — 拓扑 → 计划 ───
        npc_plans, npc_info = await self._build_npc_plans(npcs)
        if not npc_plans:
            return []

        logger.info(f"[LLM #1] {len(npc_plans)} 个 NPC 的计划已生成")

        # ─── Step 2: LLM #2 — 计划 + 拓扑 → 拓扑结构变更 ───
        intent_resolver = IntentResolver(
            graph_engine=self.graph_engine,
            resolver=self._resolver,
        )
        topology_ops = intent_resolver.resolve_all_intents(npc_plans)
        logger.info(f"[LLM #2] {len(topology_ops)} 个拓扑结构操作")

        # ─── Step 3: IntentExecutor — 执行拓扑结构变更 ───
        exec_results = self._execute_intents(topology_ops, npcs, npc_info, npc_plans)

        # ─── Step 4: LLM #3 (InteractionLayer) — 故事生成 ───
        il = InteractionLayer(resolver=self._resolver)
        all_er_dicts = [er for er in exec_results]
        edge_results = il.process(
            all_er_dicts,
            graph_engine=self.graph_engine,
            world_time_str=self._current_world_time_str,
            tick_duration_str=self._tick_duration_str,
        )
        stories = [e.description for e in edge_results]
        logger.info(f"[LLM #3] {len(edge_results)} 条边故事")

        # ─── Step 5: LLM #4 (PostProcessor) — 故事 + 拓扑 → 数值增量 + 近况投影 ───
        pp = PostProcessor(resolver=self._resolver)
        delta_ops, recent_info_map = pp.resolve_topology_deltas(
            npc_plans=npc_plans,
            stories=stories,
            graph_engine=self.graph_engine,
            world_time_str=self._current_world_time_str,
            tick_duration_str=self._tick_duration_str,
        )
        logger.info(f"[LLM #4] {len(delta_ops)} 个数值增量, {len(recent_info_map)} 条近况")

        # 写入近况投影到实体（类型无关，根据 has_recent_info 过滤）
        if recent_info_map:
            from ..config.node_ontology import has_recent_info
            written = 0
            for eid, text in recent_info_map.items():
                ent = self.graph_engine.get_entity(eid)
                if ent and has_recent_info(ent.type_id):
                    ent.recent_info = text
                    written += 1
            if written:
                logger.info(f"[LLM #4b] 近况投影写入 {written} 个实体")

        # 保存本轮 delta_ops 供后续使用
        self._last_delta_ops = delta_ops

        # ─── Step 6: GraphEngine — 执行数值增量 ───
        if delta_ops:
            result = self.graph_engine.apply_edge_operations(delta_ops)
            logger.info(f"[Engine] 数值增量执行: {result['status']} ({len(result['results'])} 条)")
            if result.get("errors"):
                for err in result["errors"]:
                    logger.warning(f"[Engine]   增量错误: {err}")

        # ─── 构建返回结果 ───
        tick_results = []
        for er_data in exec_results:
            npc_eid = er_data.get("npc_eid", "")
            npc_name = er_data.get("npc_name", "?")
            ent = self.graph_engine.get_entity(npc_eid)
            zone_now = "?"
            vitality_now = 100
            inv_view = {}

            if ent:
                # 通过 1-hop 子图找当前区域
                for conn in ent.connected_entity_ids:
                    e = self.graph_engine.get_entity(conn)
                    if e and e.entity_type == "zone":
                        zone_now = e.name
                        break
                vitality_now = int(ent.attributes.get("vitality", 100))
                inv_view = {
                    iv["item_name"]: iv["quantity"]
                    for iv in self.graph_engine.get_inventory_view(npc_eid)
                }

            plan_text = npc_plans.get(npc_eid, npc_name)
            tick_results.append({
                "npc_id": er_data.get("npc_id", ""),
                "npc_name": npc_name,
                "zone": zone_now,
                "action": plan_text[:50],
                "action_text": plan_text,
                "vitality": vitality_now,
                "inventory": inv_view,
                "tick": self.tick_count,
            })

        return tick_results

    async def _build_npc_plans(self, npcs) -> tuple[dict[str, str], dict[str, dict]]:
        """
        LLM #1: 为每个 NPC 构建独立 prompt → 生成自然语言计划。

        返回:
            npc_plans: {npc_eid: 自然语言计划}
            npc_info:  {npc_eid: {name, model, entity, zone, zone_npcs}}
        """
        npc_plans: dict[str, str] = {}
        npc_info: dict[str, dict] = {}

        npc_prompts: list[tuple[str, str]] = []

        for npc in npcs:
            neid = _make_eid("npc", npc.name)
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue

            # 1-hop 子图文本（包含自身描述 + 连接信息）
            subgraph_text = self.graph_engine.get_1hop_subgraph_text(neid)

            # 库存
            inv = self.graph_engine.get_inventory_view(neid)

            # 记忆（由 LLM #4b recent_info 代替，prompt 优先显示 ent.recent_info）
            # 旧 NPC.memory 系统已移除，memories 参数保留为空供回退
            memories = []

            # 性格标签
            personality_tags = []
            for tag in (getattr(npc, 'persona_tags', []) or []):
                if hasattr(tag, 'tag'):
                    personality_tags.append(tag.tag)

            # 同区域的其他 NPC
            zone_npcs = []
            for conn in ent.connected_entity_ids:
                e = self.graph_engine.get_entity(conn)
                if e and e.entity_type == "zone":
                    for other_ent in self.graph_engine.all_entities():
                        if other_ent.entity_type == "npc" and other_ent != ent \
                           and other_ent.is_connected_to(e.entity_id):
                            zone_npcs.append({"name": other_ent.name, "role": other_ent.role or "?"})
                    break

            # 构建 LLM #1 prompt
            prompt = build_one_npc_prompt(
                npc_entity=ent,
                npc_name=npc.name,
                npc_role=ent.role or "",
                memories=memories,
                personality_tags=personality_tags,
                inventory=inv,
                zone_npcs=zone_npcs,
                world_time_str=self._current_world_time_str,
                tick_duration_str=self._tick_duration_str,
                recipes=None,
                topology_subgraph=subgraph_text,
            )

            npc_prompts.append((neid, prompt))
            npc_info[neid] = {
                "name": npc.name,
                "model": npc,
                "entity": ent,
                "zone_npcs": zone_npcs,
            }

        if not npc_prompts:
            return {}, {}

        # LLM #1 调用
        raw_plans = await asyncio.wait_for(
            self._resolver.resolve_all_npcs_async(npc_prompts),
            timeout=300.0,
        )

        for neid, _ in npc_prompts:
            plan = raw_plans.get(neid, "")
            if isinstance(plan, str) and plan.strip():
                npc_plans[neid] = plan
            else:
                # 降级：生成一个简单的描述
                info = npc_info.get(neid, {})
                ent = info.get("entity")
                zone_name = "?"
                if ent:
                    for conn in ent.connected_entity_ids:
                        e = self.graph_engine.get_entity(conn)
                        if e and e.entity_type == "zone":
                            zone_name = e.name
                            break
                npc_plans[neid] = f"我在{zone_name}看看有什么可以做的。"

        logger.info(f"[LLM #1] 返回 {len(npc_plans)}/{len(npc_prompts)} 条计划")
        return npc_plans, npc_info

    def _execute_intents(
        self,
        topology_ops: list[dict],
        npcs,
        npc_info: dict[str, dict],
        npc_plans: dict[str, str],
    ) -> list[dict]:
        """
        执行 LLM #2 的拓扑结构变更。
        只改变图结构（connect/disconnect/set_qty），不改变数据值。

        返回：list[exec_result dict] （供 LLM #3 使用）
        """
        # 按 NPC 分组操作
        npc_ops: dict[str, list[dict]] = {}
        for op in topology_ops:
            src = op.get("src", "")
            npc_ops.setdefault(src, []).append(op)

        # 执行
        results = []
        for neid, ops in sorted(npc_ops.items()):
            info = npc_info.get(neid)
            if not info:
                continue

            # 只在 NPC 有 LLM #1 计划时执行
            plan = npc_plans.get(neid, "")
            if not plan:
                continue

            # 执行操作
            self.graph_engine.apply_edge_operations(ops)

            # 构建 exec_result
            ent = info.get("entity")
            zone_name = "?"
            if ent:
                for conn in ent.connected_entity_ids:
                    e = self.graph_engine.get_entity(conn)
                    if e and e.entity_type == "zone":
                        zone_name = e.name
                        break

            # 提取交互过的实体
            interacted_entities = []
            for op in ops:
                tgt = op.get("tgt", "")
                if tgt != neid:
                    tgt_ent = self.graph_engine.get_entity(tgt)
                    if tgt_ent:
                        if tgt_ent.entity_type == "zone":
                            zone_name = tgt_ent.name if op.get("op") == "connect" else zone_name
                        elif tgt_ent.entity_type in ("npc", "object"):
                            interacted_entities.append(tgt_ent.name)

            # 为 LLM #3 准备节点信息
            model = info.get("model")
            mem_text = model.attributes.get("_recent_info", "") if model and hasattr(model, 'attributes') else ""

            attrs = ent.attributes if ent else {}
            mood_val = attrs.get("mood")
            sat_val = attrs.get("satiety")
            vit_val = attrs.get("vitality")
            mood_txt = _val_to_mood_text(mood_val)
            sat_txt = _val_to_sat_text(sat_val)
            vit_txt = _val_to_vit_text(vit_val)

            # 性格特质（from entity traits）
            traits_list = ent.traits if ent and hasattr(ent, 'traits') else []

            results.append({
                "npc_name": info["name"],
                "npc_eid": neid,
                "npc_role": info["entity"].role if info.get("entity") else "?",
                "npc_id": info["model"].id if info.get("model") else "",
                "zone_after": zone_name,
                "zone_changed": False,
                "interacted_npcs": [n for n in interacted_entities
                                    if self.graph_engine.get_entity(f"npc_{n[:8]}") is not None],
                "interacted_objects": [n for n in interacted_entities
                                       if self.graph_engine.get_entity(f"object_{n[:8]}") is not None],
                "raw_intent": plan,
                "narrative": "",
                # 节点信息（供 LLM #3 叙事使用）
                "memories": mem_text,
                "mood_text": mood_txt,
                "satiety_text": sat_txt,
                "vitality_text": vit_txt,
                "traits": traits_list,
            })

        # 那些有 LLM #1 计划但 LLM #2 没有产生操作的 NPC
        for neid, plan in npc_plans.items():
            if neid not in npc_ops:
                info = npc_info.get(neid)
                if not info:
                    continue
                ent = info.get("entity")
                zone_name = "?"
                if ent:
                    for conn in ent.connected_entity_ids:
                        e = self.graph_engine.get_entity(conn)
                        if e and e.entity_type == "zone":
                            zone_name = e.name
                            break

                attrs = ent.attributes if ent else {}
                model = info.get("model")
                mem_text = model.attributes.get("_recent_info", "") if model and hasattr(model, 'attributes') else ""
                traits_list = ent.traits if ent and hasattr(ent, 'traits') else []

                results.append({
                    "npc_name": info["name"],
                    "npc_eid": neid,
                    "npc_role": info["entity"].role if info.get("entity") else "?",
                    "npc_id": info["model"].id if info.get("model") else "",
                    "zone_after": zone_name,
                    "zone_changed": False,
                    "interacted_npcs": [],
                    "interacted_objects": [],
                    "raw_intent": plan,
                    "narrative": "",
                    "memories": mem_text,
                    "mood_text": _val_to_mood_text(attrs.get("mood")),
                    "satiety_text": _val_to_sat_text(attrs.get("satiety")),
                    "vitality_text": _val_to_vit_text(attrs.get("vitality")),
                    "traits": traits_list,
                })

        return results

    # ═══════════════════════════════════════════
    # 记忆 & 衰减
    # ═══════════════════════════════════════════

    def _decay_and_sync(self, npcs):
        """被动衰减 + 同步。
        
        记忆写入已移除（由 LLM #4b recent_info 代替）。
        仅保留属性衰减逻辑。
        """
        for npc in npcs:
            neid = _make_eid("npc", npc.name)
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue

            # 被动衰减
            satiety = ent.attributes.get("satiety")
            mood = ent.attributes.get("mood")
            if satiety is not None:
                ent.attributes["satiety"] = max(0, satiety - 1)
            if mood is not None:
                ent.attributes["mood"] = max(0, mood - 0.5)

    # ═══════════════════════════════════════════
    # 写回
    # ═══════════════════════════════════════════

    def _sync_back_to_nodes(self, npcs):
        """
        将 entity 状态同步回持久层。
        当前支持 NPC 的位置/属性/库存/近况投影。
        后续扩展其他类型时在此函数统一处理。
        """
        from .graph_adapter import _make_eid
        for npc in npcs:
            neid = _make_eid("npc", npc.name)
            ent = self.graph_engine.get_entity(neid)
            if not ent:
                continue

            # 位置：从 1-hop 子图中找 zone 连接
            for conn in ent.connected_entity_ids:
                e = self.graph_engine.get_entity(conn)
                if e and e.entity_type == "zone":
                    npc.position.zone_id = e.name
                    break

            # 属性
            v = ent.attributes.get("vitality")
            if v is not None:
                npc.vitality = max(0, min(100, int(v)))
            s = ent.attributes.get("satiety")
            if s is not None:
                npc.satiety = max(0, min(100, int(s)))
            m = ent.attributes.get("mood")
            if m is not None:
                npc.mood = max(0, min(100, int(m)))

            # 库存：从出边中取 qty>0 的
            inv_view = self.graph_engine.get_inventory_view(neid)
            if inv_view:
                new_inv = []
                for item in inv_view:
                    new_inv.extend([item["item_name"]] * item["quantity"])
                npc.inventory = new_inv

            # 近况投影（统一接口，类型无关）
            if ent.recent_info:
                if not hasattr(npc, 'attributes') or npc.attributes is None:
                    npc.attributes = {}
                npc.attributes["_recent_info"] = ent.recent_info

    # ═══════════════════════════════════════════
    # 运行循环
    # ═══════════════════════════════════════════

    async def run(self):
        self._running = True
        self.tick_count = 0
        self._ensure_world_initialized()

        logger.info("GraphNPCEngine 启动 (纯拓扑 4-LLM 流水线)")

        while self._running:
            try:
                tick_start = time.time()
                results = await self.tick()
                tick_end = time.time()

                if results:
                    for r in results:
                        nm = r.get("npc_name", "?")
                        act = r.get("action", "")
                        vt = r.get("vitality", 0)
                        z = r.get("zone", "?")
                        inv = r.get("inventory", {})
                        inv_str = " ".join(f"{k}x{v}" for k, v in inv.items())[:60]
                        logger.info(f"[Tick] {nm} | {act[:40]} @ {z} | vit={vt} | {inv_str}")
                else:
                    logger.info("[Tick] 无NPC")

                await self._notify(results)
                elapsed = tick_end - tick_start
                await asyncio.sleep(max(0, 600 - elapsed))

            except asyncio.CancelledError:
                break
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Tick 异常: {e}")
                raise

        logger.info("GraphNPCEngine 停止")

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════
# 记忆重要度计算
# ═══════════════════════════════════════════════════


