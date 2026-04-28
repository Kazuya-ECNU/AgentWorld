# Agent World 项目全景

> 基于 LLM 驱动的多 NPC 世界模拟引擎

---

## 一、系统架构

### 整体流程（每 Tick）

```
现实 DB ─→ 实体构建 ─→ LLM 管线 ─→ 验证 ─→ 写回 DB
                 ↑                        |
                 └──── 图引擎状态持久化 ────┘
```

### 4 层 LLM 管线

```
                 LLM #1 (决策)
 每个 NPC 读取自身状态 → 独立输出自然语言行动计划
                    ↓
                 LLM #2 (意图)
 自然语言 → {interact_with[], narrative}
                    ↓
            IntentExecutor (拓扑执行)
 移动NPC/连接NPC/创建边 — 不更新属性
                    ↓
           InteractionLayer ─→ LLM #3 (故事)
 收集所有边 → 统一生成每条边的自然语言故事
                    ↓
             PostProcessor ─→ LLM #4 (批处理)
 所有NPC状态 + 故事 → 批量生成属性/库存/记忆/关系更新
                    ↓
            ConservationValidator
 检查所有交易守恒（Σgold=0, Σitem=0）
                    ↓
               Apply Updates
 回写图引擎 + NPC 模型 + DB
```

### 核心文件职责

| 文件 | 角色 |
|------|------|
| `graph_npc_engine.py` | 主编排引擎，驱动整个 tick 管线 |
| `graph_engine.py` | 图引擎：实体管理、边拓扑、库存视图 |
| `graph_adapter.py` | 从 DB+Entity Manager 构建交互图 |
| `npc_prompt_builder.py` | LLM #1 的 prompt 构建 |
| `intent_executor.py` | IntentResolver（LLM #2）+ 拓扑执行 |
| `interaction_layer.py` | 交互边管理 + LLM #3 故事生成 |
| `post_processor.py` | LLM #4 批量更新 + 自动对称 + 合并 |
| `conservation_validator.py` | 守恒校验器 |
| `interaction_resolver.py` | LLM 调用封装 |

### 数据模型

```
World (主世界)
  ├── zones[8] (farm/market/tavern/barracks/library/temple/forest/village_square)
  └── world_time (春秋周天 时:分)

NPC (16 个)
  ├── attributes: vitality/satiety/mood (0-100)
  ├── position.zone_id → zone
  ├── inventory: List[str] (物品名重复)
  ├── memory: List[MemoryEntry] (最多 20 条)
  └── relationships: Dict[npc_name -> int]

Entity Manager (运行时图)
  ├── NPC 节点 (npc_xxx)
  ├── Zone 节点 (zone_farm)
  ├── Object 节点 (obj_xxx)
  └── Item 节点 (item_小麦)
```

---

## 二、设计哲学

### 1. LLM 是大脑，代码是骨架

```
代码负责：构建上下文、约束输出格式、执行 LLM 的决定
LLM 负责：理解状态、做出判断、创造叙事
代码不替 LLM 做判断，只提供判断所需的信息和边界
```

### 2. 自然语言 > 硬编码阈值

**反模式（已废弃）：**
```python
if npc.vitality < 30:
    go_rest()  # 硬编码阈值，NPC 没有选择
```

**当前模式：**
```python
# Prompt 注入：
# ⚠️  体力 < 30：极度疲劳，必须休息恢复
# NPC 自主决定：是否休息？去哪里休息？休息多久？
```

### 3. 分层解耦

每一层只做一件事，输出给下一层：

```
决策层 (LLM#1)  →  意图解析 (LLM#2)  →  拓扑执行 (Executor)
→  故事层 (LLM#3)  →  批更新 (LLM#4)  →  校验 (Validator)
```

**为什么 4 层而不是 1 层？**
- 每一层的上下文集中且清晰（不是把所有 NPC 的所有信息塞给一次 LLM 调用）
- 每层可以独立错误处理和降级
- 故事和生产更新分离 → 故事可以自由发挥，不污染数据一致性

### 4. 图结构作为"共享状态"

- Zone / NPC / Object 都是图节点
- 边代表可达性（移动、交互、持有）
- 库存用边权重（quantity）表示，不是独立列表
- 所有权用 special edge（qty=1）表示

### 5. 自动对称 + 守恒校验

```
LLM 产出：方统领 +50 金币（忘记写对手方）
  ↓
_auto_symmetry 自动补全：王老板 -50 金币
  ↓
ConservationValidator 检查 Σ=0 ✅
  ↓
不可能漏掉交易对手（即使 LLM 忘记）
```

---

## 三、关键改进历程

### ✅ 阶段 1：从单层到 4 层管线

| 改进前 | 改进后 |
|--------|--------|
| 一次 LLM 调用决定所有 | 4 层分工：决策→意图→故事→更新 |
| 属性/库存/故事混在一起 | 拓扑执行（IntentExecutor）先跑，再生成故事和更新 |
| 没有独立校验 | ConservationValidator 校验守恒 |

### ✅ 阶段 2：硬编码阈值 → 自然语言警戒

**改了什么：**
- 移除 `npc_prompt_builder.py`, `post_processor.py`, `graph_npc_engine.py` 中全部 `if vitality < X` 判断
- 替换为 prompt 中 `⚠️  属性警戒值` 描述块
- 兜底路径统一去 tavern，不替 LLM 做分支判断

**效果：** NPC 开始自主做出生存决策：
- 铁匠王（⚡16）→ "在 tavern 躺下休息到 50 以上"
- 老陈（低饱腹）→ "去点碗热汤面"
- 王老板（⚡32）→ "在市场摆摊零售"

### ✅ 阶段 3：LLM 不可用 → 崩溃而非静默降级

- `_derive_and_execute()` 在 LLM 不可用时 `raise RuntimeError`
- 静默降级（fallback 兜底）会掩盖问题
- 兜底路径保留但变为死代码（BFS 可达性所需）

### ✅ 阶段 4：Hunger → Satiety 重构

- `hunger: int`（0=不饿）→ `satiety: int`（0=饿死, 100=饱）
- 7 个源文件 + DB migration
- 语义更直观（饱腹值从 100 向下衰减）

### ✅ 阶段 5：故事匹配修复

**问题：** `_assign_stories()` 按 `paragraphs[i] → edges[i]` 顺序匹配。LLM 输出乱序时，故事和边错位。

**修复：**
- Prompt 要求每条故事以 `【源↔目标】` 开头
- `_assign_stories()` 改为按标记匹配（不只是 index）
- 正则兜底：如果 `---` 拆分覆盖不足，直接从原文按 `【X↔Y】` 提取

---

## 四、从项目中学习到的

### 🔑 1. LLM 不可控是特性不是 Bug

LLM 会：
- 乱序输出故事 → 不要依赖顺序
- 编造物品/交易 → 用 Validator 兜住
- 忘记写交易对手 → 用 `_auto_symmetry` 补全
- 批量模板化输出 → 接受或增加区分信号

**应对原则：**
- 接受 LLM 的不可靠性
- 用代码层（校验器/自动对称/regex 兜底）兜住不可靠性
- 不要试图让 LLM "完美"——而是让系统在 LLM 不完美时依然正确

### 🔑 2. 硬编码阈值 vs LLM 判断

```
硬编码阈值：
  ✓ 确定性强（if vital < 30: rest）
  ✗ NPC 没有个性（所有人都一样）
  ✗ 违背"模拟"的初衷

LLM 判断：
  ✗ 不确定性（有时 LLM 忽视低属性）
  ✓ NPC 有个性选择（铁匠王去 tavern，老陈吃面）
  ✓ 更接近真实世界模拟
```

**取舍：** 选择 LLM 判断，接受偶尔的"不合理"行为作为真实感的一部分。

### 🔑 3. 图结构的威力

- 同一个图：支持移动（NPC↔Zone）、交互（NPC↔NPC）、持有（NPC↔Item）
- 边的 quantity 天然支持库存计数
- 区域拓扑决定了 NPC 社交网络
- 所有权边自动隔离非所有者对物体的访问

### 🔑 4. 4 层管线的合理性证明

每一层的输入和输出清晰定义，可以独立替换：
- LLM #1 可以换规则引擎（如果不想用 LLM 做决策）
- LLM #2 就是 JSON parser
- LLM #3 可以关掉（用模板故事）
- LLM #4 是整个系统的核心——它连接"发生了什么"和"属性怎么变"

### 🔑 5. 测试策略

- 8-NPC 快速验证（~3min）→ 16-NPC 全量（~6min）
- Validator 是最后一道防线（校验失败不会静默）
- 每次改一行代码就跑一次 tick（迭代速度 > 完美度）

### 🔑 6. LLM Prompt 工程经验

**好的 prompt 设计：**
- 告诉 LLM 后果（体力<30:会累倒），而不是告诉它具体行为（体力<30:去休息）
- 给例子（比如交易格式必须包含 zone + npc）
- 在输出端做校验，不在输入端做过滤

**Prompt 的"硬约束"：**
- `每条故事的第一行必须以【X↔Y】开头`
- 标 3 个感叹号强调，并在后面加例子
- 解析端同样支持降级（LLM 不遵守时 regex 兜底）

---

## 五、当前状态

| 层 | 状态 |
|----|------|
| LLM #1 决策 | ✅ 16/16 NPC 稳定产出 |
| LLM #2 意图 | ✅ 自然语言→结构化交互 |
| IntentExecutor | ✅ 移动/连接/建边 |
| LLM #3 故事 | ✅ `【X↔Y】` 标记匹配修复完成 |
| LLM #4 批更新 | ✅ 集中式处理，含自动对称 |
| Validator | ✅ 守恒校验 |
| DB 持久化 | ✅ 每 tick 写回 |

### 已知未修复的问题

- **追逐系统平衡**：移动 -5⚡ vs 恢复 +0/+3（用户认为非关键）
- **LLM #4 批量模板化**：无交互 NPC 属性更新趋于一致（非关键）
- **LLM 幻觉交易**：NPC 编造不存在的交易（非关键，现实也可接受）

### 下一步

- 连续 3+ tick 验证行为持续正确
- 生产 50-tick 仿真
- 进一步优化 prompt（减少 LLM 幻觉）
- 经济系统平衡调整
