# 20 × 1 Tick 批量运行任务

## 核心规则

每次运行前通读一遍！

### 1. 运行方式
- 脚本: `bin/run_20ticks.py`
- 每次只跑 1 tick（内部调用 `bin/run_tick_report.py --count 1`）
- 共跑 20 次，不能一次跑 20 tick
- 日志保存到 `logs/batch_20tick/tickXX_{BATCH_ID}.log`

### 2. 时间加速规则
```
if hour >= 21 or hour < 6:
    tick_minutes = 360    # 夜间：6 小时
    tick_duration = "6 小时"
else:
    tick_minutes = 30     # 白天：30 分钟
    tick_duration = "30 分钟"
```
- 在 `graph_npc_engine.py` 中实现（已修改 ✅）

### 3. 异常处理
- 某个 tick 出错 → 打出错误日志继续下一轮
- 全部跑完后汇总报告

### 4. 报告格式
- Markdown 表格：Tick | 世界时间 | LLM 数 | 耗时 | 结果
- 交易记录
- 世界时间线
- 错误记录

## 每次运行前检查清单
- [ ] `graph_npc_engine.py` 的时间加速逻辑还在（没有被覆盖）
- [ ] 日志目录 `logs/batch_20tick/` 存在
- [ ] 当前 DB 状态正常（`PYTHONPATH=src python3 -c "from agent_world.db.db import get_session; from agent_world.models.npc import NPC; with get_session() as s: rows = s.execute('SELECT data FROM npcs').fetchall(); print(f'{len(rows)} NPCs')"`）
- [ ] 上一个 tick 没有遗留问题
