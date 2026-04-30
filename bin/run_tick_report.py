#!/usr/bin/env python3
"""
单 Tick 测试报告生成器（新架构版）。
跑 1 tick，输出：
  - 总体变化（库存 delta 表格）
  - 每个 NPC 的属性变化增量
  - 逐层 LLM 调用摘要
  - LLM #4 后处理原始输出
  - 指定 NPC 的完整原始输入输出

用法:
  python3 run_tick_report.py                    # 跑 1 tick，默认输出到终端
  python3 run_tick_report.py --count 3          # 连续跑 3 tick
  python3 run_tick_report.py --npc 老张         # 指定高亮 NPC
  python3 run_tick_report.py --save             # 同时保存 trace JSON
"""
import sys, os, json, asyncio, logging
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/Documents/01_Projects/05_AgentWorld/src"))
logging.disable(logging.CRITICAL)

LLM_CAPTURE = {"calls": []}
from agent_world.services.interaction_resolver import InteractionResolver
_orig_call_llm = InteractionResolver._call_llm

def _captured_call_llm(self, prompt):
    call_id = len(LLM_CAPTURE["calls"])
    result = _orig_call_llm(self, prompt)
    LLM_CAPTURE["calls"].append({
        "id": call_id, "prompt": prompt, "response": result,
        "model": getattr(self, 'model', '?'),
    })
    return result
InteractionResolver._call_llm = _captured_call_llm


def classify(prompt: str) -> str:
    """推断 LLM 层编号（新架构提示词）"""
    if "我想做什么？" in prompt and "基本生存需求" in prompt:
        return "LLM #1 (意图规划)"
    if "拓扑结构变更" in prompt and "connect" in prompt:
        return "LLM #2 (拓扑结构)"
    if "请根据当前拓扑状态" in prompt and "叙事" in prompt:
        return "LLM #3 (叙事生成)"
    if "数据变更" in prompt and "src/tgt/delta" in prompt:
        return "LLM #4 (数值增量)"
    if "base64" in prompt:
        return f"(base64 {len(prompt)//3}c)"
    return f"(λ {len(prompt)}c)"


async def run_one_tick(engine, tick_num: int,
                       highlight_npc: str | None = None,
                       save: bool = False) -> dict:
    """执行一个 tick 并输出格式化报告"""
    LLM_CAPTURE["calls"].clear()

    # 记录 tick 前的图状态或 DB 状态
    from agent_world.db import get_session, NPCDB
    prev_inventories = {}
    prev_attrs = {}
    with get_session() as conn:
        db_npcs_before = NPCDB(conn).get_all_npcs()
        for n in db_npcs_before:
            from agent_world.services.graph_adapter import _make_eid
            neid = _make_eid("npc", n.name)
            from agent_world.services.graph_adapter import _normalize_inventory
            inv = _normalize_inventory(n)
            prev_inventories[neid] = inv
            prev_attrs[neid] = {'vitality': getattr(n,'vitality',100),
                                'satiety': getattr(n,'satiety',100),
                                'mood': getattr(n,'mood',50)}

    t0 = datetime.now()
    results = await engine.tick()
    elapsed = (datetime.now() - t0).total_seconds()

    wt = engine._current_world_time_str or "?"
    wt_before = getattr(engine, '_previous_world_time_str', 'Day ? ??:??')
    # 存储当前时间供下次使用
    engine._previous_world_time_str = wt

    # 获取新的图状态
    ge = engine.graph_engine
    npc_results = {}
    if results:
        for r in results:
            nm = r.get("npc_name", "?")
            npc_results[nm] = r

    # ── 报告头 ──
    print(f"\n{'='*65}")
    print(f"  Tick {tick_num}: {wt_before} → {wt} | {elapsed:.1f}s | LLM {len(LLM_CAPTURE['calls'])}次")
    print(f"{'='*65}")

    # ── NPC 属性变化表 ──
    print(f"\n{'NPC':<10} {'位置':<14} {'V':>6} {'S':>6} {'M':>6} {'库存变化':<35}")
    print(f"{'─'*10} {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*35}")

    for eid, ent in sorted(ge._entities.items(), key=lambda x: x[1].name or x[0]):
        if ent.entity_type != 'npc':
            continue
        cur_inv = ge.get_inventory_view(eid)
        cur_dict = {i['item_name']: i['quantity'] for i in cur_inv}
        prev = prev_inventories.get(eid, {})

        # 位置
        zone_name = "?"
        for conn in ent.connected_entity_ids:
            e = ge.get_entity(conn)
            if e and e.entity_type == 'zone':
                zone_name = e.name
                break
        prev_zone = prev_attrs.get(eid, {}).get('zone', None)

        # 属性变化
        cur_attrs = {k: v for k, v in ent.attributes.items() if v is not None}
        prev_a = prev_attrs.get(eid, {})
        dv = cur_attrs.get('vitality', 0) - prev_a.get('vitality', 0)
        ds = cur_attrs.get('satiety', 0) - prev_a.get('satiety', 0)
        dm = cur_attrs.get('mood', 0) - prev_a.get('mood', 0)

        dv_s = f"{dv:+.0f}" if abs(dv) > 0.5 else "—"
        ds_s = f"{ds:+.0f}" if abs(ds) > 0.5 else "—"
        dm_s = f"{dm:+.0f}" if abs(dm) > 0.5 else "—"

        # 库存变化摘要
        inv_changes = []
        all_items = set(list(prev.keys()) + list(cur_dict.keys()))
        for item in sorted(all_items):
            old_q = prev.get(item, 0)
            new_q = cur_dict.get(item, 0)
            if old_q != new_q:
                inv_changes.append(f"{item}:{old_q}→{new_q}")
        inv_str = ", ".join(inv_changes[:5]) if inv_changes else "—"

        print(f"{ent.name:<10} {zone_name:<14} {dv_s:>6} {ds_s:>6} {dm_s:>6} {inv_str:<35}")

    # ── 逐层 LLM 调用摘要 ──
    print(f"\n{'─'*65}")
    print("LLM 调用明细")
    for c in LLM_CAPTURE["calls"]:
        layer = classify(c["prompt"])
        npc_names = []
        for line in c["prompt"].split("\n"):
            if line.startswith("## NPC:"):
                npc_names.append(line.replace("## NPC:", "").strip())
        name_tag = f" [{', '.join(npc_names)}]" if npc_names else ""
        rp = (c["response"] or "∅").replace("\n", " ")[:120]
        print(f"  #{c['id']} {layer}{name_tag}")
        print(f"    → {rp}")

    # ── LLM #4 全量输出 ──
    for c in LLM_CAPTURE["calls"]:
        if "数据变更" in c["prompt"] and "src/tgt/delta" in c["prompt"]:
            print(f"\n{'─'*65}")
            print("LLM #4（数值增量）全量输出:")
            print(c["response"])
            break

    for c in LLM_CAPTURE["calls"]:
        if "拓扑结构变更" in c["prompt"] and "connect" in c["prompt"]:
            print(f"\n{'─'*65}")
            print("LLM #2（拓扑结构）全量输出:")
            print(c["response"])
            break

    # ── 高亮 NPC 完整 I/O ──
    if highlight_npc:
        print(f"\n{'='*65}")
        print(f"🔍 {highlight_npc} — 完整原始输入输出")
        print(f"{'='*65}")
        for c in LLM_CAPTURE["calls"]:
            p, r = c["prompt"], c["response"]
            if highlight_npc not in p and highlight_npc not in str(r):
                continue
            layer = classify(p)
            print(f"\n{'─'*65}")
            print(f"  📞 #{c['id']} — {layer}")
            print(f"{'─'*65}")
            p_display = p[:2000] + ("\n...(truncated)" if len(p) > 2000 else "")
            print(f"\n【输入】\n{p_display}\n")
            print(f"\n【输出】\n{r}\n")

    # ── 保存 ──
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_dir = os.path.expanduser("~/Documents/01_Projects/05_AgentWorld/traces_oneshot")
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"tick{tick_num}_{ts}.json")

        npc_snapshots = []
        for eid, ent in sorted(ge._entities.items(), key=lambda x: x[1].name or x[0]):
            if ent.entity_type != 'npc':
                continue
            cur_inv = ge.get_inventory_view(eid)
            inv_dict = {i['item_name']: i['quantity'] for i in cur_inv}
            zone_name = "?"
            for conn in ent.connected_entity_ids:
                e = ge.get_entity(conn)
                if e and e.entity_type == 'zone':
                    zone_name = e.name
                    break
            snap = {"name": ent.name, "zone": zone_name,
                    "vitality": ent.attributes.get('vitality', 0),
                    "satiety": ent.attributes.get('satiety', 0),
                    "mood": ent.attributes.get('mood', 0),
                    "inventory": inv_dict}
            npc_snapshots.append(snap)

        trace = {
            "tick": tick_num,
            "elapsed": round(elapsed, 1),
            "world_time": wt,
            "npcs": npc_snapshots,
            "llm_calls": LLM_CAPTURE["calls"],
        }
        with open(path, "w") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nTrace saved: {path}")

    return results


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent World — 单 Tick 报告生成器（新架构）")
    parser.add_argument("--npc", default=None, help="高亮某个 NPC 展示完整 I/O")
    parser.add_argument("--save", action="store_true", help="保存 trace JSON")
    parser.add_argument("--count", type=int, default=1, help="连续跑 N 个 tick")
    args = parser.parse_args()

    from agent_world.db.db import init_db
    from agent_world.entities.recipe import RecipeRegistry
    from agent_world.services.graph_npc_engine import GraphNPCEngine

    init_db()
    RecipeRegistry.init_defaults()

    engine = GraphNPCEngine(llm_available=True, llm_callback=lambda r: None)
    engine._ensure_world_initialized()
    engine._init_resolver()

    # 检测当前 tick 数
    start_tick = getattr(engine, 'tick_count', 1)

    for i in range(args.count):
        await run_one_tick(engine, start_tick + i,
                          highlight_npc=args.npc, save=args.save)

    print(f"\n✅ 完成 {args.count} tick(s)")


if __name__ == "__main__":
    asyncio.run(main())
