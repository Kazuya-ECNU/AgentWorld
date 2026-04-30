# AgentWorld

> 基于 LLM 驱动的多 NPC 世界模拟引擎

一个由 **4 层 LLM 管线**驱动的多智能体世界模拟系统。NPC 在村庄/森林/集市/酒馆等区域中自主生活、交易、社交，所有行为由大模型实时决策，代码仅负责拓扑约束和守恒验证。

## 核心架构

### 4 层 LLM 管线（每 Tick）

```
DB / 图引擎 ──→ LLM #1 (决策) ──→ 每个 NPC 输出自然语言计划
                    ↓
LLM #2 (拓扑) ──→ 移动/连接/断连 — 只改结构不改数值
                    ↓
LLM #3 (叙事) ──→ 为每个交互子图生成故事
                    ↓
LLM #4 (执行) ──→ 批量输出属性/库存/关系更新
                    ↓
ConservationValidator → 验证 Σgold=0, Σitem=0
                    ↓
            回写 DB + 图引擎
```

### 设计原则

- **LLM 是大脑，代码是骨架**：代码提供上下文和约束，LLM 做判断
- **自然语言 > 硬编码阈值**：不写 `if vitality < 30: rest()`，而是注入到 prompt 让 LLM 自行判断
- **纯拓扑边**：NPC 之间只有数量连接（连通/断连），不携带接口语义
- **守恒校验**：所有交易必须通过 ConservationValidator（物资/金币守恒）

## 项目结构

```
src/agent_world/
├── api/              # HTTP API 层
├── cognition/        # LLM prompt 构建、记忆管理
│   └── npc_prompt_builder.py  # LLM #1 的决策 prompt
├── config/           # 节点本体、世界配置
│   └── node_ontology.py       # 节点类型标签系统
├── db/               # SQLite 持久化
├── entities/         # 实体模型（NPC/Zone/Item/Object）
├── models/           # Pydantic 数据模型
│   ├── npc.py                # NPC 模型
│   ├── npc_defaults.py       # 16 个预设 NPC
│   └── world.py              # 世界时间系统
└── services/         # 核心服务层
    ├── graph_npc_engine.py   # 主编排引擎
    ├── graph_engine.py       # 图拓扑引擎
    ├── graph_adapter.py      # DB → 图适配
    ├── intent_executor.py    # LLM #2 拓扑执行
    ├── interaction_layer.py  # LLM #3 故事生成
    ├── post_processor.py     # LLM #4 批量更新
    ├── conservation_validator.py  # 守恒校验
    └── interaction_resolver.py    # LLM 调用封装

bin/
├── run_tick_report.py    # 单 tick 报告生成器
└── run_20ticks.py        # 20 tick 批量运行器

docs/                     # 架构决策记录 (ADR)
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python3 -c "from agent_world.db.db import init_db; init_db()"

# 运行 1 个 tick
python3 bin/run_tick_report.py

# 运行完整报告（含 LLM 调用明细）
python3 bin/run_tick_report.py --npc 老陈 --save

# 批量运行 20 tick
python3 bin/run_20ticks.py
```

## 世界设定

- **时间系统**: 春夏秋冬四季，每个 tick 推进 30 分钟（夜间 6 小时）
- **区域**: farm / market / tavern / barracks / library / temple / forest / village_square
- **NPC**: 16 个预设角色（农夫、商人、工匠、猎户等），各有背景故事和属性
- **属性**: vitality(体力) / satiety(饱腹) / mood(心情) — 0-100，随时间自然衰减

## LLM 管线详解

| 层级 | 输入 | 输出 | 职责 |
|------|------|------|------|
| #1 决策 | NPC 状态 + 库存 + 位置 + 最近信息 | 自然语言计划 | 决定做什么 |
| #2 拓扑 | 所有 NPC 计划 | `connect/disconnect` 操作 | 空间/社交移动 |
| #3 叙事 | 拓扑变更后的子图结构 | 自然语言故事 | 为每个交互生成叙事 |
| #4 执行 | 故事 + 全局状态 | 属性/库存 delta | 执行交易和属性更新 |

## 技术栈

- **Python 3.12+** with Pydantic
- **LLM**: 支持 MiniMax / Anthropic 兼容 API
- **图引擎**: 自定义内存级图拓扑（非 NetworkX）
- **持久化**: SQLite
- **验证**: 守恒校验器保证经济系统不出错

## License

MIT
