# World REST API

from datetime import datetime
from fastapi import APIRouter, HTTPException

from ..db import WorldDB, get_session
from ..db.schemas import WorldResponse, SuccessResponse
from ..models.world import World

router = APIRouter(prefix="/world", tags=["World"])


@router.get("", response_model=WorldResponse)
def get_world():
    """获取世界信息"""
    with get_session() as conn:
        db = WorldDB(conn)
        world = db.get_world()
        if not world:
            raise HTTPException(status_code=404, detail="World not found")
        return WorldResponse(**world.model_dump())


@router.post("/tick", response_model=WorldResponse)
def tick_world(minutes: int = 1):
    """推进世界时间"""
    with get_session() as conn:
        db = WorldDB(conn)
        world = db.get_world()
        if not world:
            raise HTTPException(status_code=404, detail="World not found")
        
        world.world_time.tick(minutes)
        world.updated_at = datetime.now()
        db.save_world(world)
    return WorldResponse(**world.model_dump())


@router.post("/refresh")
async def refresh_world():
    """
    手动触发一次世界刷新（World Update）。
    
    执行内容：
    - 推进世界时间
    - LLM/规则评估世界状态
    - 生成世界事件（天气、经济、社交）
    - 广播事件
    """
    from ..services.world_updater import get_world_updater
    updater = get_world_updater()
    result = updater.refresh()
    return {"success": True, **result}


@router.get("/events")
def get_world_events(limit: int = 10):
    """获取最近的世界事件"""
    from ..services.world_updater import get_world_updater
    updater = get_world_updater()
    return {
        "success": True,
        "events": [e.to_dict() for e in updater._event_history[-limit:]],
        "current_weather": updater._current_weather,
    }
