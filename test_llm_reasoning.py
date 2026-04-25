#!/usr/bin/env python3
"""Test LLM goal reasoning for NPC"""
import sys
sys.path.insert(0, 'src')

from agent_world.cognition.reasoner import GoalReasoner, _get_minimax_credentials

# Check credentials
base_url, api_key = _get_minimax_credentials()
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:15]}..." if api_key else "No API key")

# Create reasoner
reasoner = GoalReasoner()

# Test cases
test_cases = [
    ("商人，白天，精力充足，市场摆摊", "merchant"),
    ("矿工，夜间，精力一般，需要休息", "miner"),
    ("农民，白天，精力充沛，在农场", "farmer"),
]

print("\n=== LLM Goal Reasoning Test ===")
for desc, role in test_cases:
    prompt = f"""NPC角色：{role}
状态：精力100/100，位置市场区，背包空，无记忆，时间白天。

请根据以上信息推理NPC的目标。"""
    
    print(f"\n[{role}] {desc}")
    print(f"  Prompt: {prompt[:60]}...")
    
    goal = reasoner.reason(prompt)
    if goal:
        print(f"  ✅ goal={goal.goal} reason={goal.reason[:40]} plan={goal.plan}")
    else:
        print(f"  ❌ Failed to get goal")