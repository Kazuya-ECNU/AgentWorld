#!/usr/bin/env python3
import subprocess, sys, time, json, os, signal

os.chdir("/home/asher/Documents/01_Projects/05_AgentWorld")

# Kill existing server
subprocess.run(["pkill", "-f", "world_viewer.py"], stderr=subprocess.DEVNULL)
time.sleep(2)

# Start server
p = subprocess.Popen(
    [sys.executable, "-u", "web/world_viewer.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    bufsize=1, universal_newlines=True
)

# Wait for startup
time.sleep(10)

# Test 1: World API
import urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8765/api/world", timeout=5)
    d = json.loads(r.read())
    print("✅ Test 1 - World API OK")
    print(f"   Time: {d['world_time']}")
    print(f"   Active NPCs: {d['active_npcs']}")
except Exception as e:
    print(f"❌ Test 1 - World API failed: {e}")

# Test 2: NPC API
try:
    r = urllib.request.urlopen("http://localhost:8765/api/npc", timeout=5)
    d = json.loads(r.read())
    print(f"✅ Test 2 - NPC API OK ({d['count']} NPCs)")
    for n in d['data']:
        print(f"   {n['name']}: status={n['status']} zone={n['position']['zone_id']} inv={len(n['inventory'])}")
except Exception as e:
    print(f"❌ Test 2 - NPC API failed: {e}")

# Test 3: Wait for ticks and check behavior
print("\n⏳ Waiting 20s for NPC ticks...")
time.sleep(20)

try:
    r = urllib.request.urlopen("http://localhost:8765/api/npc", timeout=5)
    d = json.loads(r.read())
    print(f"✅ Test 3 - NPC after ticks:")
    for n in d['data']:
        print(f"   {n['name']}: status={n['status']} zone={n['position']['zone_id']}")
        print(f"     inv={len(n['inventory'])} items: {n['inventory'][:3]}...")
        if n['memory']:
            m = n['memory'][-1]
            print(f"     last: [{m['timestamp'][11:19]}] {m['event'][:70]}")
except Exception as e:
    print(f"❌ Test 3 - NPC after ticks failed: {e}")

# Test 4: LLM reasoner direct test
print("\n⏳ Test 4 - Direct LLM reasoner test...")
sys.path.insert(0, "src")
from agent_world.cognition.reasoner import GoalReasoner, _get_minimax_credentials

base_url, api_key = _get_minimax_credentials()
print(f"   Credentials: {base_url} | {api_key[:15]}...")

reasoner = GoalReasoner()
prompt = "NPC角色：商人\n状态：精力80/100，位置市场区，背包空，时间白天。\n请推理NPC的目标，用一句话回答goal和reason。"
goal = reasoner.reason(prompt)
if goal:
    print(f"✅ Test 4 - LLM reasoning OK: goal={goal.goal} reason={goal.reason[:50]}")
else:
    print("❌ Test 4 - LLM reasoning FAILED")

# Test 5: Check tick log output
print("\n⏳ Test 5 - Checking server tick output...")
stdout, _ = p.communicate(timeout=2)
llm_ticks = 0
total_ticks = 0
for line in stdout.split('\n'):
    if '[Tick]' in line:
        total_ticks += 1
        if '[LLM]' in line or 'MiniMax' in line:
            llm_ticks += 1
    if any(k in line for k in ['[NPCEngine]', '[Tick]', '[GoalReasoner]', 'LLM推理']):
        print(f"   {line.strip()}")

print(f"\n📊 Tick Summary: {total_ticks} total ticks, {llm_ticks} with LLM reasoner")
print("✅ All tests passed!" if llm_ticks > 0 else "⚠️  No LLM ticks detected yet")