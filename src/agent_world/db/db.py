# Database Layer - SQLite Storage

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from ..models.npc import NPC, NPCRole, NPCStatus, Position
from ..models.world import World, Zone, WorldTime, ZoneType, DEFAULT_ZONES


DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "agent_world.db"


def get_db_path() -> Path:
    """获取数据库路径"""
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "agent_world.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


class WorldDB:
    """世界数据库操作"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def get_world(self) -> World | None:
        """获取世界数据"""
        cursor = self.conn.execute("SELECT data FROM world WHERE id = 'main_world'")
        row = cursor.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return World(**data)
    
    def save_world(self, world: World):
        """保存世界数据"""
        data = world.model_dump_json()
        self.conn.execute(
            "INSERT OR REPLACE INTO world (id, data, updated_at) VALUES ('main_world', ?, ?)",
            (data, datetime.now().isoformat())
        )
        self.conn.commit()


class NPCDB:
    """NPC 数据库操作"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_npc(self, npc: NPC) -> NPC:
        """创建 NPC"""
        data = npc.model_dump_json()
        self.conn.execute(
            "INSERT INTO npcs (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (npc.id, data, datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.conn.commit()
        return npc
    
    def get_npc(self, npc_id: str) -> NPC | None:
        """获取单个 NPC"""
        cursor = self.conn.execute("SELECT data FROM npcs WHERE id = ?", (npc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return NPC(**data)
    
    def get_all_npcs(self) -> list[NPC]:
        """获取所有 NPC"""
        cursor = self.conn.execute("SELECT data FROM npcs")
        return [NPC(**json.loads(row[0])) for row in cursor.fetchall()]
    
    def update_npc(self, npc: NPC) -> NPC:
        """更新 NPC"""
        data = npc.model_dump_json()
        self.conn.execute(
            "UPDATE npcs SET data = ?, updated_at = ? WHERE id = ?",
            (data, datetime.now().isoformat(), npc.id)
        )
        self.conn.commit()
        return npc
    
    def delete_npc(self, npc_id: str):
        """删除 NPC"""
        self.conn.execute("DELETE FROM npcs WHERE id = ?", (npc_id,))
        self.conn.commit()
    
    def list_npcs(self, zone_id: str | None = None, role: str | None = None, limit: int = 100) -> list[NPC]:
        """筛选 NPC 列表"""
        query = "SELECT data FROM npcs WHERE 1=1"
        params = []
        if zone_id:
            query += " AND json_extract(data, '$.position.zone_id') = ?"
            params.append(zone_id)
        if role:
            query += " AND json_extract(data, '$.role') = ?"
            params.append(role)
        query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query, params)
        return [NPC(**json.loads(row[0])) for row in cursor.fetchall()]


@contextmanager
def get_session() -> Generator:
    """获取数据库会话（上下文管理器）"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """初始化数据库"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    # 创建表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS world (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS npcs (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS world_objects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            object_type TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            position_x REAL DEFAULT 0,
            position_y REAL DEFAULT 0,
            state TEXT DEFAULT 'available',
            current_user TEXT DEFAULT NULL,
            capacity INTEGER DEFAULT 1,
            current_goods TEXT DEFAULT NULL,
            growth_stage TEXT DEFAULT NULL,
            resources_left REAL DEFAULT NULL,
            uses_left INTEGER DEFAULT NULL,
            metadata TEXT DEFAULT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # 初始化默认世界
    cursor = conn.execute("SELECT id FROM world WHERE id = 'main_world'")
    if not cursor.fetchone():
        default_world = World(
            id="main_world",
            name="Agent World",
            description="一个 AI Agent 与 NPC 共存的世界",
            zones=DEFAULT_ZONES,
            world_time=WorldTime()
        )
        data = default_world.model_dump_json()
        conn.execute(
            "INSERT INTO world (id, data, updated_at) VALUES ('main_world', ?, ?)",
            (data, datetime.now().isoformat())
        )
    
    conn.commit()
    conn.close()
    
    print(f"数据库已初始化: {db_path}")