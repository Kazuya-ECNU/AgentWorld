# RISC-like Entity Minimalization Axiom

## Principle

In AgentWorld's interactive simulation, when the LLM deduces the existence of a new
entity during its reasoning (e.g., referencing an item not yet registered in the
world), the engine follows a **3-step minimalization axiom**, inspired by RISC
philosophy: introduce no entity unless truly irreducible.

## The 3-Step Algorithm

### Step 1: Derive via Process Chain

Try to find an existing **recipe / process chain** that produces the target entity
from entities that already exist in the world.

- Example: "面包" (bread) is referenced but does not exist.
  RecipeRegistry has `烘焙面包: 小麦x2 → 面包x1`.
  "小麦" exists in the world → register "面包" by deriving from the chain.

If all inputs to the chain already exist, **no new basic entities are created**.

### Step 2: Recursive Decomposition

If the inputs to the chain themselves do not exist, recurse on each input.

- Example: "面粉" is referenced, chain `碾磨面粉: 小麦x3 → 面粉x2` exists.
  Recursively check "小麦" → it exists. Register "面粉" via the chain.
- Example: "衣物" is referenced, chain `加工皮毛: 皮毛x2 → 衣物x1` exists.
  Recursively check "皮毛" → does "皮毛" exist? If not, recurse further.

The recursion bottoms out at **basic entities** (Step 3) or existing entities.

### Step 3: Register as Irreducible Basic

If no process chain can produce the entity (or its inputs cannot be recursively
resolved), register it as a **basic entity**. Basic entities are the world's
irreducible primitives — they cannot be manufactured via any known process.

**Default basic entities:**
- 水 (water)
- 石头 (stone)
- 木材 (wood)
- 铁矿石 (iron ore)
- 砂 (sand)
- 陶土 (clay)
- 种子 (seeds)
- 野果 (wild fruit)

These entities exist from the beginning and are not produced by any recipe.

### Step 0 (Implicit): Already Exists?

If the entity already exists in the world graph, do nothing.

## Design Rationale

### Why "RISC-like"?

RISC (Reduced Instruction Set Computer) minimizes the instruction set to the
bare essentials — everything else is composed from those primitives through
well-defined sequences. Similarly, the entity model reduces the world's entity
set to:

1. **Basic primitives** — irreducible materials/items that cannot be made.
2. **Derived entities** — everything else produced by process chains from basics.

This eliminates redundancy (no two ways to represent the same "product entity"),
forces explicit modeling of production chains, and makes the world self-evident:
given the primitives and recipes, the full entity space is determined.

### Traceability

Every entity created through this axiom carries a derivation log:
- Direct registration (already existed)
- Chain-derived (which chain, what inputs)
- Basic-registered (irreducible)

This traceability is essential for debugging LLM behavior and understanding how
the world evolved during simulation.

---

# Energy Conservation Principle (First Principle)

## Statement

AgentWorld is a **closed material system** with open environmental boundaries.
Every inventory quantity change must be explainable by one of four interaction
types. If none applies, the change is a system error.

This is not a numeric zero-sum check — it's a **type-based validation**:
wheat −2 and beer +1 is not a numeric match, but if a recipe says
"2 wheat + 1 vegetable → 2 beer", the change is **explainable** and
thus valid.

## The Four Valid Interaction Types

### Type 1: Transfer (NPC ↔ NPC, same item)

An item moves from one NPC to another without transformation:

```
老张: 小麦 −5
王老板: 小麦 +5
```

**Validation rule:** For the same `item_name`, Σ(delta) across all NPCs = 0.
**Marking in updates:** `"type": "transfer"` or automatically detected when
same item has both + and - entries.

### Type 2: Craft (Recipe Execution)

Items are consumed as recipe inputs; other items are produced as outputs:

```
田嫂: 小麦 −2, 蔬菜 −1, 酒 +2
```

**Validation rule:** The ratio must match a known recipe in RecipeRegistry:
- Inputs: {小麦: 2, 蔬菜: 1}
- Outputs: {酒: 2}
- Match found → ✅ valid transformation
**Marking in updates:** `"type": "craft"` with `"recipe": "酿酒"`.

If no matching recipe exists, the validator issues a **soft warning** —
the NPC might have discovered a new recipe that hasn't been registered yet.

### Type 3: Consume (Item → Attribute)

An item is consumed to affect NPC's own attributes (vitality, hunger, mood):

```
老张: 酒 −1
老张: mood +20, vitality +5
```

**Validation rule:** The item disappears from the system (no compensation).
The attribute changes are recorded separately — validator only checks that
the item's disappearance is marked as `"type": "consume"`.
**Marking in updates:** `"type": "consume"`.

### Type 4: Gather (Open Environmental Boundary)

An item enters the system from the environment without compensation:

```
猎户: 肉 +5
```

**Validation rule:** The item appears from nowhere — only allowed if marked
as `"type": "gather"` and the NPC is in an appropriate zone (forest, mine).
**Marking in updates:** `"type": "gather"` with optional `"source": "环境"`.

### Catch-All: Unknown Change

```
田嫂: 蔬菜 −2, 王老板: 金币 +10  (no recipe, no consume tag)
```

**Validation rule:** No known type matches → **hard violation.** The change
is rolled back and a warning is logged. This catches LLM hallucination where
it invents quantity changes that have no physical or environmental basis.

## Validator Architecture

The ConservationValidator checks inventory changes by interaction type:

```
PostProcessor output → updates[]
    ↓
Validator(type="transfer"): CHECK Σ(delta) per item_name = 0
Validator(type="craft"):    CHECK matches known recipe ratios
Validator(type="consume"):  CHECK marked as consume (always passes)
Validator(type="gather"):   CHECK marked as gather (always passes)
Validator(type="unknown"):  HARD VIOLATION → rollback + alarm
```

**Why `type` must be explicit in PostProcessor output:**
The LLM #3 PostProcessor already knows what type of interaction it's
processing — it saw the LLM #1 decision and the execution result. It should
simply tag each `inventory_changes` entry with its interaction type.
The validator doesn't guess; it checks based on what the LLM declared.

### Transfer Validation

```
def validate_transfers(updates):
    items = {}
    for u in updates:
        for ic in u.get("inventory_changes", []):
            if ic.get("type") == "transfer":
                delta = ic["quantity"]
                delta = -delta if ic["action"] == "remove" else delta
                items[ic["item_name"]] = items.get(ic["item_name"], 0) + delta
    return all(abs(v) < 0.001 for v in items.values())
```

### Craft Validation

```
def validate_craft(updates):
    inputs, outputs = {}, {}
    recipe_name = ""
    for u in updates:
        for ic in u.get("inventory_changes", []):
            if ic.get("type") == "craft":
                recipe_name = ic.get("recipe", recipe_name)
                if ic["action"] == "remove":
                    inputs[ic["item_name"]] = inputs.get(ic["item_name"], 0) + ic["quantity"]
                else:
                    outputs[ic["item_name"]] = outputs.get(ic["item_name"], 0) + ic["quantity"]
    recipe = registry.get_by_name(recipe_name)
    if not recipe: return ("SOFT", "未知配方")
    return ("PASS", "") if recipe.inputs == inputs and recipe.outputs == outputs \
           else ("HARD", f"配方 {recipe_name} 比例不匹配")
```

## Architectural Implications

### What Each Layer Knows About Conservation

| Layer | Knows | Doesn't Know |
|---|---|---|
| **GraphEngine** | edge topology + qty | Conservation entirely |
| **IntentExecutor** | Topology changes only | Quantity, conservation — zero |
| **PostProcessor** | Full NPC states + decision | Must output `type` tags |
| **ConservationValidator** | Validation rules per type | Game mechanics, economy |

### The Three-LLM Pipeline

```
LLM #1: "我想把小麦卖给王老板换金币"  → 自然语言
LLM #2: {interact_with: [...]}       → 只关心交互对象
IntentExecutor: connect(老张↔王老板)  → 拓扑操作
LLM #3: [...] + type="transfer"     → 数据层 + 类型标记
Validator: transfer→Σ小麦=0 ✅        → 纯校验
```

## Design Decisions

### 1. PostProcessor Must Tag Types

**Validator does not infer types.** PostProcessor declares them.
This keeps the validator deterministic and pushes reasoning to LLM #3.

### 2. `allow_npcs` Dedup Policy

One PostProcessor generates BOTH sides of a transfer. `allow_npcs` allows
both halves to be applied.

### 3. Topology vs. Quantity vs. Validation

Three separate concerns, three separate components.

### 4. Market Value: Future Module

Energy = physical truth. Market = economic truth. Validator never
references market prices.

## Integration Points

| Component | Role |
|---|---|
| `DerivationRegistry` | Basic entities + process chains |
| `RecipeRegistry` | Recipes as process chains |
| `EntityDerivationEngine` | 3-step derivation bridge |
| `GraphEngine._auto_register_missing` | Entry: LLM deduces unknown entity |
| `ConservationValidator` | Validates PostProcessor output by type |

## Example Derivation Flows

```
[LLM references "面包"]
  1. Already exists? No
  2. Chain? 烘焙面包: 小麦x2 → 面包x1
  3. Input "小麦" exists? Yes
  4. → Register entity "item_面包", register chain "烘焙面包"
```

```
[LLM references "绸缎"]
  1. Already exists? No
  2. Chain? None found
  3. → Register "item_绸缎" as new basic entity
```
