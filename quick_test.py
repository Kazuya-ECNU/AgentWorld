#!/usr/bin/env python3
import subprocess, sys, time, json, os

os.chdir("/home/asher/Documents/01_Projects/05_AgentWorld")

# Kill existing
subprocess.run(["pkill", "-f", "world_viewer.py"], capture_output=True)
time.sleep(2)

# Start server
p = subprocess.Popen([sys.executable, "web/world_viewer.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(12)

import urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8765/api/npc", timeout=5)
    npcs = json.loads(r.read())["data"]
    print(f"NPCs: {len(npcs)}")
    for n in npcs:
        print(f"  {n['name']}: status={n['status']} zone={n['position']['zone_id']}")
        print(f"    goal={n['goal']} reason={n['goal_reason']}")
        print(f"    inv={len(n['inventory'])} items")
        if n["memory"]:
            m = n["memory"][-1]
            print(f"    last mem: [{m['timestamp'][11:19]}] {m['event']}")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== Server stdout ===")
stdout, _ = p.communicate(timeout=2)
for line in stdout.decode().split('\n'):
    if any(k in line for k in ['[NPCEngine]', '[GoalReasoner]', '[Tick]', '[LLM']):
        print(' ', line)