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
