"""
Node Ontology — Declarative type definition for the topology engine.

This is the only place where node types are mapped to numeric IDs and
traversal/prompt behavior. No engine code should branch on semantic strings.

All switches default to False. A type not listed here gets no special
traversal behavior (i.e., full BFS expansion).
"""

from __future__ import annotations
from typing import Any

# ─── Numeric Type IDs ───
# These are pure numbers — the engine never interprets them semantically.
TYPE_NPC: int = 1
TYPE_ZONE: int = 2
TYPE_ITEM: int = 3
TYPE_OBJECT: int = 4

# Semantic string → type_id (for bootstrap/migration only; engine never reads)
TYPE_NAME_TO_ID: dict[str, int] = {
    "npc": TYPE_NPC,
    "zone": TYPE_ZONE,
    "item": TYPE_ITEM,
    "object": TYPE_OBJECT,
}

# Entity ID prefix → type string → type_id (for _infer_type replacement)
ENTITY_PREFIX_TO_ID: dict[str, int] = {
    "npc_": TYPE_NPC,
    "zone_": TYPE_ZONE,
    "item_": TYPE_ITEM,
    "obj_": TYPE_OBJECT,
}


# ─── Ontology Table ───
# Each entry maps type_id → {traversal switches, prompt rules, properties}
# All switches are off by default for unlisted types.

NODE_ONTOLOGY: dict[int, dict[str, Any]] = {
    TYPE_ITEM: {
        "terminal": True,           # BFS stops here — leaf node
        "has_recent_info": True,
        "prompt": {
            "category": "物品",
            "order": 3,
        },
    },
    TYPE_OBJECT: {
        "terminal": True,           # BFS stops here — leaf node
        "has_recent_info": False,
        "prompt": {
            "category": "物件",
            "order": 3,
        },
    },
    TYPE_ZONE: {
        "same_type_block": True,    # BFS doesn't cross zone↔zone edges
        "has_recent_info": True,
        "prompt": {
            "category": "场所",
            "order": 1,
            "fields": ["name", "description"],
        },
    },
    TYPE_NPC: {
        # default — no terminal, no same_type_block → full traversal
        "has_recent_info": True,
        "prompt": {
            "category": "角色",
            "order": 2,
            "fields": ["name", "role", "mood", "satiety", "vitality", "memories", "traits", "intent"],
        },
        "properties": {
            "has_attributes": True,
            "has_inventory": True,
            "has_memory": True,
        },
    },
}


# ─── Lookup Helpers ───

def get_ontology(type_id: int) -> dict[str, Any]:
    """Get ontology entry for a type_id. Returns empty dict for unknown types."""
    return NODE_ONTOLOGY.get(type_id, {})


def is_terminal(type_id: int) -> bool:
    """Is this type a terminal/leaf node for BFS?"""
    return bool(NODE_ONTOLOGY.get(type_id, {}).get("terminal", False))


def is_same_type_blocked(type_id: int) -> bool:
    """Should BFS stop when encountering a neighbor of the same type?"""
    return bool(NODE_ONTOLOGY.get(type_id, {}).get("same_type_block", False))


def type_name_to_id(name: str) -> int:
    """Convert semantic type string to numeric type_id."""
    return TYPE_NAME_TO_ID.get(name, 0)


def prefix_to_type_id(prefix: str) -> int:
    """Infer type_id from entity ID prefix (e.g., 'npc_' → TYPE_NPC)."""
    for pfx, tid in ENTITY_PREFIX_TO_ID.items():
        if prefix.startswith(pfx):
            return tid
    return 0


def has_recent_info(type_id: int) -> bool:
    """Does this type support recent_info projection?"""
    return bool(NODE_ONTOLOGY.get(type_id, {}).get("has_recent_info", False))
