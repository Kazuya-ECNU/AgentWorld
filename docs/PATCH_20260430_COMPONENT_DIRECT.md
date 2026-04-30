# Patch: Component-Direct Story Assignment

**日期:** 2026-04-30 23:00
**触发:** BFS 阻断 zone↔zone 后，`_assign_stories` 的 `【场景:名称】` 解析路由彻底成为冗余

---

## 改动内容

### 删除

| 函数 | 行数 | 原因 |
|------|------|------|
| `_assign_stories()` | ~50 行 | 不再需要解析 LLM 输出做路由 |
| `_group_by_zone()` | ~30 行 | 被 `_group_by_zone_only()` 替代 |
| `_SCENE_HEAD_RE` | 1 行 | `【场景:xxx】` 正则 |
| `_STORY_HEAD_RE` | 1 行 | 旧式 `【源↔目标】` 正则 |

### 新增

| 函数/结构 | 行数 | 作用 |
|-----------|------|------|
| `@dataclass Component` | 5 行 | 连通子图的唯一载体：entity_ids + npc_names + edges |
| `_group_by_zone_only()` | 15 行 | 无 graph_engine 时的兜底分组 |
| `process()` 的 per-component 循环 | ~3 行 | 逐子图调 LLM，直接分配故事 |

### 修改

- `_group_by_subgraph()` — 返回类型从 `dict[str, list[EdgeResult]]` 改为 `list[Component]`
- `_build_story_prompt()` — 签名从 `(edges, exec_results, zone_edges, graph_engine, ...)` 改为 `(component, exec_results, graph_engine, ...)`
- `_legacy_build_prompt()` — 签名同步简化
- `process()` — 删除 `_assign_stories` 调用和 zone_edges 中转变量

---

## 改动原因

### 之前的问题

`_assign_stories` 的存在理由：LLM #3 一次收到所有子图的 context，一次返回多条故事，每条用 `【场景:name】` 标记归属。引擎通过正则解析 LLM 输出，把故事路由回对应的 zone 的边。

这个设计的根本矛盾：**分组是确定性事实（BFS 算出来的），却依赖 LLM 的文本输出重新确认**。

具体问题积累：

1. **name 不匹配** — LLM 写 `【场景:temple】`，zone 实际叫 market，正则匹配失败
2. **模糊匹配脆弱** — `"market" in "temple"` 为 False，兜底生成占位符
3. **结构冗余** — BFS 已经知道每个边属于哪个子图，解析是在重复确认已知事实
4. **prompt 约束负担** — 必须要求 LLM "名称必须与场所一致"，这是架构问题的症状不是根因

### 现在的方案

```
BFS → list[Component]
        ↓
循环 Component:
  建 prompt → 调 LLM → 故事 → 直接放 comp.edges 上
        ↓
无解析、无正则、无路由
```

**本质变化：** 路由从"LLM 告诉我某个故事属于哪"变成"引擎已经知道，直接放过去"。

---

## 设计思想

### 1. 拓扑事实不应由 LLM 确认

分组是拓扑层面的计算，是确定性任务。LLM #3 的职责是讲故事——它不应该参与确认"这个故事属于什么 zone"这样的拓扑事实。这是引擎工作的分界。

### 2. 一个子图 = 一次 LLM 调用

之前把全部子图打包进一次 prompt 让 LLM 一次写多个故事，然后靠解析拆分，本质是在**用 LLM 做引擎的分工**——让 LLM 做路由决策。现在改为每子图独立调 LLM，LLM 只需要关注当前子图内的故事。

调用次数增加但 prompt 变小（每个只带当前子图的 context），总 token 量基本不变。

### 3. 节点统一呈现，不预分类

`_build_story_prompt` 不再为 NPC、zone、item 创建互不相同的区块。所有节点用统一的格式列出：

```
· node_name
  类型: zone/npc/item
  描述: ...
  身份: ...
  体力: ...
  想法: ...
```

**引擎不替 LLM 分类，让节点的自有属性说话。** LLM 通过 `entity_type` / `role` / `desc` 自然理解"这是一个场所还是一个角色"。

---

## 优势

| 维度 | 之前 | 现在 |
|------|------|------|
| **代码量** | ~50 行解析 + 正则 + 模糊匹配 | 0 行路由代码 |
| **依赖 LLM** | LLM 必须写正确的场景名 | LLM 只需写故事 |
| **容错性** | LLM 写错名 → 模糊匹配失败 → 占位符 | 不依赖 LLM 确认拓扑事实 |
| **prompt 复杂度** | 需要约束"名称必须与场所一致" | 无场景名约束 |
| **维护面** | 正则 2 套 + 解析逻辑 2 条路径 | 无正则、无解析 |
| **扩展性** | 加新节点类型可能影响路由逻辑 | 加新类型完全不影响 |

---

## 验证

```
连续 2 轮 tick 测试 ✅
  第 1 轮: 【场景】午后的春风裹挟着青草香气... ✅
  第 2 轮: 【场景】春日的午后，阳光斜斜地洒落在market的石板路上... ✅
  BFS 不跨 zone ✅
  交易数据 ConservationValidator 通过 ✅
```
