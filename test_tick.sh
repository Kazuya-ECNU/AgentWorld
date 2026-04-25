#!/bin/bash
pkill -f 'world_viewer.py' 2>/dev/null
sleep 1
cd /home/asher/Documents/01_Projects/05_AgentWorld
nohup python3 web/world_viewer.py > /tmp/wv3.log 2>&1 &
sleep 7
curl -s http://localhost:8765/api/npc | python3 -c "
import sys, json
d = json.load(sys.stdin)
for n in d['data']:
    print(f\"{n['name']}: status={n['status']} zone={n['position']['zone_id']}\")
    print(f\"  inv={n['inventory']}\")
    print(f\"  _debug: plan_idx={n['_debug_plan_idx']} cooldown={n['_debug_cooldown']}\")
"
cat /tmp/wv3.log | grep -E "\[Tick\]|\[NPCEngine\]" | head -20