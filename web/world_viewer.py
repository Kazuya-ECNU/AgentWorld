# Agent World - FastAPI Server + WebSocket + NPC Engine

"""
Phase 2 + Phase 3: REST API + WebSocket + NPC Engine
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from agent_world.db import init_db, get_session, WorldDB, NPCDB
from agent_world.models.world import World
from agent_world.models.npc import NPC, NPCRole, NPCStatus
from agent_world.services.npc_engine import NPCEngine

# === FastAPI App ===

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    asyncio.create_task(start_npc_engine())
    yield
    # 关闭时
    print("[NPC Engine] 关闭中...")

app = FastAPI(title="Agent World API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).parent
app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

# === WebSocket 连接管理器 ===

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for client in self.active_connections:
            try:
                await client.send_json(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self.disconnect(client)


manager = ConnectionManager()


# === WebSocket 端点 ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print(f"[WS] Client connected: {websocket.client}")

    try:
        with get_session() as conn:
            world_db = WorldDB(conn)
            npc_db = NPCDB(conn)
            world = world_db.get_world()
            npcs = npc_db.get_all_npcs()

        await websocket.send_json({
            "type": "init",
            "world": world.model_dump() if world else None,
            "npcs": [npc.model_dump() for npc in npcs],
            "timestamp": datetime.now().isoformat()
        })

        async for message in websocket.iter_text():
            data = json.loads(message)
            print(f"[WS] Received: {data}")

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif data.get("type") == "subscribe":
                await websocket.send_json({"type": "subscribed", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"[WS] Client disconnected: {websocket.client}")


# === NPC Engine 集成 ===

async def broadcast_tick_results(results: list):
    """NPC Engine tick 结果广播到所有 WS 客户端"""
    await manager.broadcast({
        "type": "tick",
        "data": results,
        "timestamp": datetime.now().isoformat()
    })


# === REST API 路由 ===

from agent_world.api import world as world_api
from agent_world.api import npc as npc_api

app.include_router(world_api.router, prefix="/api")
app.include_router(npc_api.router, prefix="/api")


@app.get("/")
async def root():
    return {"msg": "Agent World API", "version": "0.1.0", "phase": 3}


# === 后台任务：NPC Engine ===

async def start_npc_engine():
    """启动 NPC Engine 作为后台任务"""
    engine = NPCEngine()
    engine.add_listener(broadcast_tick_results)
    await engine.run()





# === 主入口 ===

def main():
    print("=" * 50)
    print("Agent World - Phase 3 Server")
    print("=" * 50)

    print("初始化数据库...")
    init_db()
    print()

    print("🌐 HTTP Server: http://0.0.0.0:8765")
    print("📡 WebSocket: ws://0.0.0.0:8765/ws")
    print("📖 API Docs: http://0.0.0.0:8765/docs")
    print("🤖 NPC Engine: 运行中 (每 5s tick)")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
