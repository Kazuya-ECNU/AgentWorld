# Patch: LLM #4 双输出 + Story Projection

**日期:** 2026-05-01 00:31
**触发:** 市场三人连续 3 tick 0 交易死循环——老陈说卖小麦→LLM #3 写讨价还价→LLM #4 不出 delta（故事没成交）→老陈记忆只存了"我说了要卖"→下 tick 又说卖小麦

---

## 问题

### 死循环链路

```
LLM #1: 我决定在market卖掉2单位小麦
  ↓ (存记忆: "在market 我决定在market卖掉...，持小麦x2")
LLM #3: 老陈和王老板在讨价还价，他出10金币一单位...
  ↓ (不存任何地方)
LLM #4: 故事里没成交 → 不出 delta → []
  ↓ (存记忆: 还是"在market 我决定在market卖掉...")
LLM #1 (下 tick): 记忆里看到自己在市场要卖小麦但库存没变 →
  以为自己是第一次说"我要卖小麦"
```

**根因：LLM #3 的故事从不投影回节点，LLM #1 永远看不到上 tick 发生了什么。**

### 结构问题

当前 `_apply_memories_and_decay()` 写的记忆 = LLM #1 自己的计划原文，等价于「老陈说：我要卖小麦」。这不是叙事——只是 echo，不提供新信息。NPC 的「最近经历」里只有自己说过的话，没有世界的反馈。

---

## 改动内容

### 新增

| 位置 | 内容 | 作用 |
|------|------|------|
| `Entity.story_projection: str` | 新字段 | 每 tick LLM #4b 投影故事到此字段 |
| `PostProcessor._parse_output()` | 取代 `_parse_ops()` | 解析 LLM #4 双输出 |

### 修改

| 文件 | 改动 | 说明 |
|------|------|------|
| `post_processor.py` — `resolve_topology_deltas()` | 返回 `(list[dict], dict[str,str])` | 原本只返回 operations，增加 memory_map |
| `post_processor.py` — Prompt 输出格式段 | 从 `array` → `object {operations, projections}` | 输出结构变化 |
| `post_processor.py` — Prompt 新增一段 | "节点近况投影"指令段 | 描述 LLM #4b 任务 |
| `graph_npc_engine.py` — Step 5 | 接收 `projections` 并写入 entity 字段 | 连接 LLM #4b → 节点 |
| `graph_npc_engine.py` — `_apply_memories_and_decay()` | 保留衰减 + 旧记忆 fallback | 改用 `story_projection` 替代 `action` 作为记忆主文本 |
| `npc_prompt_builder.py` — "最近经历"段 | 优先展示 `entity.story_projection` | LLM #1 读到叙事记忆 |

---

## LLM #4 的新输出格式

旧格式（纯数组）：
```json
[
  {"op": "delta", "src": "老陈", "tgt": "item_金币", "delta": 10},
  {"op": "attr", "target": "刘猎户", "attr": "vitality", "delta": -10}
]
```

新格式（对象，双段）：
```json
{
  "operations": [
    {"op": "delta", "src": "老陈", "tgt": "item_金币", "delta": 10},
    {"op": "attr", "target": "刘猎户", "attr": "vitality", "delta": -10}
  ],
  "projections": {
    "老陈": "在market和王老板讨价还价，10金币没谈拢",
    "王老板": "想低价收老陈的小麦他嫌价高没成",
    "market": "春日的市集，老陈和王老板在讨价还价，铁匠王在旁边观望",
    "铁匠王": "在market看老陈和王老板谈粮食生意",
    "小麦": "被老陈拿来和王老板谈价，没成交",
    "刘猎户": "在森林打猎收获2张皮毛，心情不错"
  }
}
```

兼容旧格式：如果 LLM 输出纯数组，`_parse_output` 自动按旧路径解析，`projections` = 空 dict。

---

## 设计思想

### 1. Story Projection ≠ Memory

`Entity.story_projection` 不是"记忆系统"——它只是每 tick 被 LLM #3 的故事刷新一次的快照。

类比：
- **记忆** = NPC 的个人日记（历史积累、压缩、衰减）
- **投影** = 世界对 NPC 的**最近反馈**（叙事结果）

NPC 的现有 `NPC.memory` 列表（LLM #1 计划原文 + 衰减 + 压缩）**保留不动**。`story_projection` 是新增的字段，LLM #1 prompt 优先读取。

### 2. 所有节点都是投影目标

不新增 `memory_target` 标签，不区分节点类型：

```python
class Entity:
    story_projection: str = ""
```

Zone 收到投影 → `"market: 春日的市集，老陈和王老板在讨价还价"`
Item 收到投影 → `"小麦: 被老陈拿来和王老板谈价，没成交"`

Engine 写入时不加过滤——LLM #4 输出什么就写什么。以后新增节点类型（如 Faction、Quest）自动受益。

### 3. LLM #1 读投影 = 自然闭环

```
LLM #4b 写 "老陈: 在market和王老板讨价还价没谈拢"
  ↓
LLM #1 读出 → 「上次没谈拢，这次降价试试」
  ↓
LLM #3 写 "老陈降价到11，王老板接受，成交"
  ↓
LLM #4a 出 delta: {老陈 -2小麦, +22金币}, {王老板 +2小麦, -22金币}
```

### 4. Prompt 不写"记忆"这个词

LLM #4 prompt 中指令段的措辞：

```
==== 节点近况投影 ====
请根据本轮故事，为故事中涉及到的每个节点生成近况摘要。
近况会作为该节点下个 tick 的上下文。

- 按节点写，每个节点一条，20-60 字
- 从该节点的视角：NPC写"我做了什么"，zone写"这里发生了什么"
- 没有事件可以不写
```

不出现「记忆」「记住」等词。LLM #1 拿到投影后用"最近信息"展示，LLM #1 自然把它当作自己的近期经历。

---

## 数据流（改造后）

```
LLM #1
  │ 读 npc_prompt_builder (含 entity.story_projection)
  │ 写 plan (存 NPC.memory)
  ▼
LLM #2
  │ 读 plan
  │ 写 topology (NPC↔Zone 移动)
  ▼
LLM #3
  │ 读 topology (BFS 分组)
  │ 写 story (自然语言)
  ▼
LLM #4 (同一个调用)
  │
  ├─ #4a: 读 story → 写 operations (delta/attr/set_qty)
  └─ #4b: 读 story → 写 entity.story_projection  ← 新
  ▼
_apply_memories_and_decay ← 保留（衰减 + 旧记忆写回兜底）
  │ 衰减 satiety/mood
  │ 回退：如果没有 story_projection，用 LLM #1 plan 原文
  ▼
(下个 tick) LLM #1 读到 story_projection → 闭环
```

---

## 如何打断死循环

| Tick | 改前 | 改后 |
|------|------|------|
| T5 | LLM #1: "卖小麦" → LLM #3: "讨价还价" → #4: [] → 记忆: "在market卖小麦" | LLM #1: "卖小麦" → LLM #3: "讨价还价" → #4: [] + 投影: "在market和王老板谈价没谈拢" |
| T6 | LLM #1: 记忆=「我在market卖小麦」→又重复"卖小麦" | LLM #1: 投影=「在market和王老板谈价没谈拢」→**降价或换策略** |

---

## 优势

| 维度 | 改前 | 改后 |
|------|------|------|
| **NPC 看到的世界反馈** | 无（只看到自己的计划） | 有（故事的叙事投影） |
| **信息断链** | LLM #3 → LLM #1 断连 | LLM #3 → #4b → 投影 → LLM #1 全链路 |
| **新节点类型** | 需加记忆逻辑 | 自动受益（所有节点都投影） |
| **LLM #4 调用次数** | 1次 | 1次（不变） |
| **代码复杂度** | 旧格式简单 | 兼容新旧两种格式 + 投影写入 |
| **配置复杂度** | 无 | 无（不新增标签） |
