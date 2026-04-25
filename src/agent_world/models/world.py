# World Data Model

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ZoneType(str, Enum):
    """区域类型"""
    VILLAGE_SQUARE = "village_square"  # 村庄广场
    MARKET = "market"                   # 市场
    TAVERN = "tavern"                   # 酒馆
    FARM = "farm"                       # 农场
    MINE = "mine"                       # 矿场
    FOREST = "forest"                   # 森林
    LIBRARY = "library"                 # 图书馆
    TEMPLE = "temple"                   # 神庙
    BARRACKS = "barracks"               # 兵营
    OUTSKIRTS = "outskirts"             # 城郊


class Zone(BaseModel):
    """世界中的一个区域"""
    id: str
    name: str
    zone_type: ZoneType
    description: str = ""
    
    # 地理信息
    bounds: dict = Field(default_factory=lambda: {
        "min_x": 0, "max_x": 100,
        "min_y": 0, "max_y": 100
    })
    
    # 区域内 NPC 上限
    capacity: int = 20
    
    # 连接的区域
    connected_zones: list[str] = Field(default_factory=list)


class WorldTime(BaseModel):
    """世界时间系统"""
    year: int = 1
    month: int = 1
    day: int = 1
    hour: int = 8  # 游戏内时间（24小时制）
    minute: int = 0
    
    def tick(self, minutes: int = 1):
        """推进时间"""
        self.minute += minutes
        while self.minute >= 60:
            self.minute -= 60
            self.hour += 1
        while self.hour >= 24:
            self.hour -= 24
            self.day += 1
        while self.day >= 31:
            self.day -= 30
            self.month += 1
        while self.month >= 13:
            self.month -= 12
            self.year += 1
    
    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute
        }

    def get_time_of_day(self) -> str:
        """返回 'dawn' | 'day' | 'dusk' | 'night' | 'midnight'"""
        h = self.hour
        if 5 <= h < 8:    return "dawn"
        if 8 <= h < 17:   return "day"
        if 17 <= h < 20:  return "dusk"
        if 20 <= h < 24:  return "night"
        return "midnight"

    def is_night(self) -> bool:
        return self.hour >= 20 or self.hour < 5

    def get_season(self) -> str:
        """返回 'spring' | 'summer' | 'autumn' | 'winter'"""
        season_map = {1: "spring", 2: "spring", 3: "spring",
                      4: "summer", 5: "summer", 6: "summer",
                      7: "autumn", 8: "autumn", 9: "autumn",
                      10: "winter", 11: "winter", 12: "winter"}
        return season_map.get(self.month, "spring")

    def to_display_str(self) -> str:
        """如 '春·第 3 天 14:30'"""
        seasons = {"spring": "春", "summer": "夏", "autumn": "秋", "winter": "冬"}
        season_name = seasons.get(self.get_season(), "春")
        return f"{season_name}·第 {self.day} 天 {self.hour:02d}:{self.minute:02d}"


class World(BaseModel):
    """整个游戏世界"""
    id: str = "main_world"
    name: str = "Agent World"
    description: str = "一个 AI Agent 与 NPC 共存的世界"
    
    # 世界分区
    zones: list[Zone] = Field(default_factory=list)
    
    # 世界时间
    world_time: WorldTime = Field(default_factory=WorldTime)
    
    # 统计
    active_npcs: int = 0
    total_events: int = 0
    
    # 创建时间
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def is_night(self) -> bool:
        return self.world_time.is_night()

    def get_time_str(self) -> str:
        return self.world_time.to_display_str()


# 默认世界模板
DEFAULT_ZONES = [
    Zone(id="village_square", name="村庄广场", zone_type=ZoneType.VILLAGE_SQUARE,
         description="村庄的中心广场，NPC 们在这里交流与交易。",
         bounds={"min_x": 40, "max_x": 60, "min_y": 40, "max_y": 60},
         connected_zones=["market", "tavern"]),
    Zone(id="market", name="市场区", zone_type=ZoneType.MARKET,
         description="热闹的集市，商贩们在叫卖商品。",
         bounds={"min_x": 60, "max_x": 80, "min_y": 40, "max_y": 60},
         connected_zones=["village_square", "mine"]),
    Zone(id="tavern", name="酒馆", zone_type=ZoneType.TAVERN,
         description="温暖的酒馆，旅人和村民们在这里休息闲聊。",
         bounds={"min_x": 20, "max_x": 40, "min_y": 40, "max_y": 60},
         connected_zones=["village_square", "forest"]),
    Zone(id="farm", name="农场", zone_type=ZoneType.FARM,
         description="绿油油的农田，农民们在耕作。",
         bounds={"min_x": 60, "max_x": 80, "min_y": 20, "max_y": 40},
         connected_zones=["market"]),
    Zone(id="mine", name="矿场", zone_type=ZoneType.MINE,
         description="阴暗的矿场，矿工们挖掘珍贵的矿石。",
         bounds={"min_x": 80, "max_x": 100, "min_y": 60, "max_y": 80},
         connected_zones=["market"]),
    Zone(id="forest", name="森林", zone_type=ZoneType.FOREST,
         description="茂密的森林，偶尔有野兽出没。",
         bounds={"min_x": 0, "max_x": 20, "min_y": 60, "max_y": 80},
         connected_zones=["tavern", "outskirts"]),
    Zone(id="library", name="图书馆", zone_type=ZoneType.LIBRARY,
         description="宁静的图书馆，学者们在此研究。",
         bounds={"min_x": 40, "max_x": 60, "min_y": 20, "max_y": 40},
         connected_zones=["village_square", "temple"]),
    Zone(id="temple", name="神庙", zone_type=ZoneType.TEMPLE,
         description="庄严的神庙，治疗师在这里治愈伤痛。",
         bounds={"min_x": 20, "max_x": 40, "min_y": 20, "max_y": 40},
         connected_zones=["library", "barracks"]),
    Zone(id="barracks", name="兵营", zone_type=ZoneType.BARRACKS,
         description="士兵们训练的地方。",
         bounds={"min_x": 0, "max_x": 20, "min_y": 20, "max_y": 40},
         connected_zones=["temple"]),
    Zone(id="outskirts", name="城郊", zone_type=ZoneType.OUTSKIRTS,
         description="安静的城郊，偶尔有流浪者经过。",
         bounds={"min_x": 0, "max_x": 20, "min_y": 80, "max_y": 100},
         connected_zones=["forest"]),
]