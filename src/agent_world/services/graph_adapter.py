"""
适配器 —— 把现有 NPC/WorldObject/Zone 模型包装为 Entity 接口。
"""

from __future__ import annotations

from ..entities.base_entity import Entity
from ..entities.base import WorldObject
from ..entities.world_objects import WorldObjectManager
from ..models.npc import NPC
from ..models.world import World, Zone
from ..models.interaction import EntityInterface


def npc_to_entity(npc: NPC) -> Entity:
    """把 NPC 模型包装成 Entity（原子动作接口）"""
    eid = f"npc_{npc.id[:8]}"
    ent = Entity(eid, npc.name, "npc")

    # 属性
    ent.set_attr("vitality", getattr(npc, 'vitality', 100),
                 min_value=0, max_value=100, description="体力/活力值")
    ent.set_attr("hunger", getattr(npc, 'hunger', 50),
                 min_value=0, max_value=100, description="饥饿度，0=饱 100=极饿")
    ent.set_attr("mood", getattr(npc, 'mood', 50),
                 min_value=0, max_value=100, description="心情，0=极差 100=极好")
    ent.set_attr("zone_id", getattr(npc.position, 'zone_id', 'village_square'),
                 description="当前所在区域")
    ent.set_attr("role", npc.role.value if hasattr(npc.role, 'value') else str(getattr(npc, 'role', '')),
                 description="角色")

    # 角色标签
    for tag in getattr(npc, 'persona_tags', []) or []:
        t = tag.tag if hasattr(tag, 'tag') else (tag.get('tag', '') if isinstance(tag, dict) else str(tag))
        if t:
            ent.set_attr(f"tag_{t}", True, description=f"性格标签: {t}")

    # 物理属性
    pa = getattr(npc, 'physical', None)
    if pa:
        for field in ['recovery_speed', 'energy_capacity']:
            val = getattr(pa, field, None)
            if val is not None:
                ent.set_attr(field, val)

    # 原子接口（只有 4 种通用动作）
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_move",
        entity_id=eid,
        name="移动",
        description="移动到另一个区域",
    ))
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_interact",
        entity_id=eid,
        name="交互",
        description="与当前区域的物体或其他人互动",
    ))
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_hold",
        entity_id=eid,
        name="持有",
        description="持有或携带物品",
    ))
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_wait",
        entity_id=eid,
        name="等待",
        description="什么都不做，等待",
    ))

    return ent


def object_to_entity(obj: WorldObject) -> Entity:
    """把 WorldObject 包装成 Entity（只暴露「可交互」一个接口）"""
    eid = f"obj_{obj.id[:8]}"
    ent = Entity(eid, obj.name, "object")

    # 属性
    ent.set_attr("state", obj.state.value if hasattr(obj.state, 'value') else str(obj.state),
                 description=f"物体状态: {obj.state}")
    ent.set_attr("zone_id", obj.zone_id, description="所在区域")
    ent.set_attr("object_type", obj.object_type.value if hasattr(obj.object_type, 'value') else str(obj.object_type),
                 description="物体类型")
    ent.set_attr("description", getattr(obj, 'description', obj.name),
                 description="物体描述")

    # 接口1：可交互（任何人都可以用它）
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_interactable",
        entity_id=eid,
        name="可交互",
        description="可以被互动（LLM 根据该物体属性推导具体交互内容）",
    ))

    # 接口2：可持有（谁有 qty>0 就代表控制/拥有此物体）
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_holdable",
        entity_id=eid,
        name="可持有",
        description="NPC 持有此物 qty>0 表示拥有/控制该物体",
    ))

    return ent


def zone_to_entity(zone: Zone) -> Entity:
    """把 Zone 包装成 Entity（只暴露「可抵达」一个接口）"""
    eid = f"zone_{zone.id}"
    ent = Entity(eid, zone.id, "zone")

    ent.set_attr("zone_type", getattr(zone, 'zone_type', zone.id),
                 description="区域类型")
    ent.set_attr("weather", getattr(zone, 'weather', '晴朗'),
                 description="当前天气")
    for conn in getattr(zone, 'connected_zones', []) or []:
        ent.set_attr(f"connects_to_{conn}", True, description=f"可通往 {conn}")

    # 唯一接口：可抵达
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_reachable",
        entity_id=eid,
        name="可抵达",
        description=f"可以被移动到 {zone.id} 区域",
    ))

    return ent


def create_item_entity(item_name: str, total_qty: int,
                        location_zone: str = "world") -> Entity:
    """创建物品类型实体（每种物品一个节点，不按实例重复创建）"""
    eid = f"item_{item_name}"
    ent = Entity(eid, item_name, "item")
    ent.set_attr("total_quantity", total_qty, description="总库存量")
    ent.set_attr("location", location_zone, description="存放区域")
    ent.set_attr("description", f"{item_name}，世界中有{total_qty}单位",
                 description="物品描述")

    # 暴露两个接口：可持有（NPC 持有用）、可被消耗（物体消耗用）
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_holdable",
        entity_id=eid,
        name="可持有",
        description="可以被 NPC 持有/携带",
    ))
    ent.add_interface(EntityInterface(
        interface_id=f"{eid}_produce",
        entity_id=eid,
        name="可产出",
        description="可以被生产/收获出来",
    ))

    return ent


def build_world_graph(
    npcs: list[NPC],
    objects: list[WorldObject],
    zones: list[Zone],
    object_manager: WorldObjectManager,
) -> list[Entity]:
    """
    从 NPC / 物体 / 区域 / 物品类型 构建完整的交互图实体集合。
    返回 Entity 列表，由调用方注册到 GraphEngine。
    """
    entities: list[Entity] = []
    eid_map: dict[str, Entity] = {}

    # 1. 创建 NPC 实体
    for npc in npcs:
        ent = npc_to_entity(npc)
        entities.append(ent)
        eid_map[ent.entity_id] = ent

    # 2. 创建物体实体
    for obj in objects:
        ent = object_to_entity(obj)
        entities.append(ent)
        eid_map[ent.entity_id] = ent

    # 3. 创建区域实体
    for zone in zones:
        ent = zone_to_entity(zone)
        entities.append(ent)
        eid_map[ent.entity_id] = ent

    # 4. 创建物品类型实体（去重）
    item_registry: dict[str, int] = {}  # item_name -> total_qty
    # ... 后续扩展从数据库/配置读取
    default_items = {
        "小麦": 200,
        "金币": 1000,
        "铁锭": 50,
        "药水": 30,
        "蔬菜": 80,
        "面包": 60,
        "皮毛": 30,
        "草药": 60,
        "纸张": 20,
        "货物": 50,
        "武器": 20,
        "书籍": 40,
        "酒": 100,
        "佛经": 10,
    }
    for iname, qty in default_items.items():
        if iname not in item_registry:
            ent = create_item_entity(iname, qty, "world")
            entities.append(ent)
            eid_map[ent.entity_id] = ent
            item_registry[iname] = qty

    # 5. 建立拓扑连接
    # NPC ↔ 同区域的物体
    for npc in npcs:
        npc_eid = f"npc_{npc.id[:8]}"
        ent = eid_map.get(npc_eid)
        if not ent:
            continue
        npc_zone = getattr(npc.position, 'zone_id', '')

        for obj in objects:
            if obj.zone_id == npc_zone:
                obj_eid = f"obj_{obj.id[:8]}"
                if obj_eid in eid_map:
                    ent.connect_to(obj_eid)
                    eid_map[obj_eid].connect_to(npc_eid)

        # Zone 连接
        for zone in zones:
            if zone.id == npc_zone:
                zone_eid = f"zone_{zone.id}"
                if zone_eid in eid_map:
                    ent.connect_to(zone_eid)

    # 物体 ↔ Zone
    for obj in objects:
        obj_eid = f"obj_{obj.id[:8]}"
        obj_ent = eid_map.get(obj_eid)
        if not obj_ent:
            continue
        for zone in zones:
            if zone.id == obj.zone_id:
                zone_eid = f"zone_{zone.id}"
                if zone_eid in eid_map:
                    obj_ent.connect_to(zone_eid)

    # Zone ↔ Zone（通过 connected_zones）
    for zone in zones:
        zone_eid = f"zone_{zone.id}"
        z_ent = eid_map.get(zone_eid)
        if not z_ent:
            continue
        for conn in getattr(zone, 'connected_zones', []) or []:
            conn_eid = f"zone_{conn}"
            if conn_eid in eid_map:
                z_ent.connect_to(conn_eid)

    return entities
