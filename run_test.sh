#!/bin/bash
cd /home/asher/Documents/01_Projects/05_AgentWorld
python3 web/world_viewer.py > /tmp/wv_llm.log 2>&1 &
PID=$!
echo "Server PID: $PID"
sleep 12

echo "=== World API ==="
curl -s http://localhost:8765/api/world | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Time:', d['world_time'])
print('Active NPCs:', d['active_npcs'])
"

echo ""
echo "=== NPC State ==="
curl -s http://localhost:8765/api/npc | python3 -c "
import sys, json
d = json.load(sys.stdin)
for n in d['data']:
    print(f'{n[\"name\"]}: status={n[\"status\"]} zone={n[\"position\"][\"zone_id\"]} inv={len(n[\"inventory\"])}')
    for m in n['memory'][-3:]:
        print(f'  [{m['timestamp'][11:19]}] {m['event']}')
"

echo ""
echo "=== Tick Log ==="
cat /tmp/wv_llm.log | grep -E "\[NPCEngine\]|\[GoalReasoner\]|\[Tick\]" | head -20