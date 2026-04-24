# Agent World — 项目规划

## 目录结构

```
agent-world/
├── src/
│   └── agent_world/           # 项目根包
│       ├── __init__.py
│       ├── models/            # 数据模型层
│       │   ├── world.py
│       │   └── npc.py
│       ├── db/                # 数据库层
│       │   ├── db.py
│       │   └── schemas.py
│       ├── api/                # API 层（路由）
│       │   ├── world.py
│       │   └── npc.py
│       ├── services/           # 核心业务/NPC引擎
│       │   └── npc_engine.py
│       ├── cognition/          # [NEW] 认知/推理引擎 (Phase 3.5a)
│       │   ├── persona.py       # 角色标签（可扩展）
│       │   ├── memory.py        # 记忆管理
│       │   ├── context.py       # 上下文构建
│       │   ├── reasoner.py      # LLM 推理引擎
│       │   └── fallbacks.py     # 规则引擎兜底
│       ├── entities/           # [NEW] 世界实体系统 (Phase 3.5b)
│       │   ├── base.py          # 实体基类
│       │   └── world_objects.py # 具体物体（摊位/农田/矿脉...）
│       └── config.py
├── web/                       # 浏览器可视化
│   ├── world_viewer.py        # Python 静态文件服务器 + WebSocket
│   ├── index.html             # 可视化主页面
│   └── assets/
├── data/                      # SQLite
├── requirements.txt
└── README.md
```

---

## 阶段规划

### Phase 1 — 数据层 + 可视化骨架

**工具：** `pydantic` / `sqlite3` / `html/css/js`

**功能：**
- NPC 数据结构定义完毕（models/）
- 数据库初始化（空世界）
- 浏览器可打开 `index.html`，看到空白地图和 NPC 列表（无数据则为空状态）
- **此阶段没有 API、没有行为**，纯数据 + 可视化框架

**产出文件：**
- `src/agent_world/models/npc.py`
- `src/agent_world/models/world.py`
- `src/agent_world/db/db.py`
- `src/agent_world/db/schemas.py`
- `web/index.html`
- `web/world_viewer.py`（骨架）
- `data/agent_world.db`（空库）

---

### Phase 2 — API 层 + 实时推送

**工具：** `FastAPI` / `websockets` / `uvicorn`

**功能：**
- REST API：创建/查询/更新/删除 NPC
- 浏览器自动刷新显示世界状态（无刷新）
- WebSocket 连接管理

**产出文件：**
- `src/agent_world/api/npc.py`
- `src/agent_world/api/world.py`
- `web/world_viewer.py`（完整 WebSocket）
- 新增 API 接口可访问

---

### Phase 3 — NPC Engine（行为树版）

**工具：** `openai` / `langchain` / `asyncio` / `行为树`

**功能：**
- NPC 根据时间 tick 自动行动（种田/交易/社交）
- 行为树驱动 NPC 状态转换
- NPC 记忆积累，影响后续行为
- **可视化跟随：** NPC 在地图上移动、交互

**产出文件：**
- `src/agent_world/services/npc_engine.py`
- NPC 行为自动执行
- 地图上 NPC 可见移动

**状态：** ⚠️ 已运行但行为空转（随机游走，缺乏目的驱动）

---

### Phase 3.5a — Cognition Module（认知模块）✅

**工具：** `openai` / LLM / Pydantic

**功能：**
- **Persona Tags** — NPC 角色标签（可扩展，支持自动生成）
- **Memory Store** — 记忆管理（追加/检索/摘要/滑动窗口）
- **Context Builder** — 上下文构建（给 LLM 的输入 prompt）
- **Goal Reasoner** — LLM 推理引擎，从数据推导 Goal
- **Fallback Engine** — LLM 不可用时的规则引擎兜底

**数据流向：**
```
PersonaTags + Memory + ContextBuilder → Prompt → GoalReasoner (LLM) → GoalOutput
                                                           ↓ (fallback)
                                                    FallbackEngine (规则兜底)
```

**产出文件：**
- `src/agent_world/cognition/persona.py`
- `src/agent_world/cognition/memory.py`
- `src/agent_world/cognition/context.py`
- `src/agent_world/cognition/reasoner.py`
- `src/agent_world/cognition/fallbacks.py`

**状态：** ✅ 已完成（2026-04-24）

---

### Phase 3.5b — Entities Module（实体系统）

**工具：** Pydantic / `asyncio`

**功能：**
- **WorldObject 基类** — 所有可交互实体的基类（状态机 + 交互接口）
- **实体类型** — Stall（商人摊位）、FarmPlot（农田）、OreVein（矿脉）、BarCounter（酒馆吧台）等
- **WorldObjectManager** — 实体管理器（创建/查询/状态更新）
- **交互接口** — `can_interact()` / `interact()` / `get_affordances()`
- NPC 的行为围绕"找可交互物体"而不是"随机选区域"

**核心概念：**
```
Zone (空间)
  └── Objects (实体)
       ├── 类型 (stall, ore_vein, farmland, bar_counter...)
       ├── 状态 (可用/占用/损坏/空置...)
       ├── 交互接口 (work, use, collect, trade...)
       └── 当前使用者 (谁在用这个物体)
```

**产出文件：**
- `src/agent_world/entities/base.py`
- `src/agent_world/entities/world_objects.py`

**状态：** ○ 未开始

---

### Phase 3.5c — NPC Engine 重构（Goal-Driven 版）

**工具：** cognition + entities + asyncio

**功能：**
- NPC Engine 接 cognition 模块（GoalReasoner → GoalOutput）
- NPC Engine 接 entities 模块（Goal → 找可交互物体 → 交互）
- NPC 行为从"规则/状态机"变为"目的驱动"
- 行为循环：评估状态 → 形成 Goal → 找 Target → 移动 → 交互 → 回到步骤1

**重构前后对比：**

| | Phase 3（规则驱动） | Phase 3.5c（目的驱动） |
|---|---|---|
| 行为触发 | 随机/基于状态 | 基于 LLM 推理的 Goal |
| 移动理由 | "随机去相邻区域" | "我要去完成 Goal X" |
| 交互对象 | 无（只有 Zone） | 找能完成 Goal 的物体/NPC |
| 行为来源 | 硬编码条件分支 | Persona + Memory + Context → LLM |

**产出文件：**
- 重构 `src/agent_world/services/npc_engine.py`

**状态：** ○ 未开始（依赖 3.5a ✅ + 3.5b ○）

---

### Phase 4 — 外部 Agent 接入协议

**工具：** 注册协议 / `API Key` 认证 / REST 接口

**功能：**
- 外部 AI Agent 可通过 API 加入世界
- 外部 Agent 与 NPC 交互
- 世界状态对外部 Agent 可见

**产出文件：**
- `src/agent_world/api/agent.py`（注册/认证/act）
- 外部 Agent 可接入

---

### Phase 5 — 持久化 + NPC 演进

**工具：** SQLite / `schedule` / NPC 记忆系统

**功能：**
- 服务重启后世界状态保留
- NPC 根据历史发展（升级/技能变化）
- 世界时间独立推进（日夜循环）

**产出文件：**
- 完整持久化
- NPC 自动演进系统
- 世界时间推进

---

## 设计原则

1. **每阶段独立验证** — 不影响上一步
2. **可视化从第一天就集成** — 不事后补救
3. **NPC Engine 和 API 层分离** — 便于单独测试
4. **新增不覆盖** — 已完成代码不被改写
5. **数据驱动目的** — NPC 行为来源：Persona Tags + Memory + Recent Context + Knowledge → Goal

---

## 当前阶段

**✅ Phase 1** — 数据层 + 可视化骨架
**✅ Phase 2** — API 层 + 实时推送
**✅ Phase 3** — NPC Engine（行为树版，已运行但需重构）
**✅ Phase 3.5a** — Cognition Module（已完成）
**○ Phase 3.5b** — Entities Module（未开始）
**○ Phase 3.5c** — NPC Engine 重构（未开始，依赖 3.5b）
**○ Phase 4** — 外部 Agent 接入协议
**○ Phase 5** — 持久化 + NPC 演进

---

_最后更新：2026-04-24_