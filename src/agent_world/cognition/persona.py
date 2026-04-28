"""
Persona Tags - NPC 角色标签系统

Tags 定义了 NPC 的核心思维方式与行为倾向。
设计为可扩展：支持冷启动时手动设置，也支持从行为/记忆中自动总结生成。

Tags 格式：dict[str, Any]
  - str: 标签名 (如 "cautious", "ambitious")
  - Any: 值，可以是 bool / float / str，视标签类型而定
"""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime


class PersonaTags(BaseModel):
    """
    NPC 的角色标签，决定其思维方式与行为风格。
    
    可扩展字段，可根据需要增加新的标签类型。
    当前字段为冷启动默认值，实际使用时由 reasoner 读取并传给 LLM。
    
    使用 alias 实现中文标签名（如 alias="勤奋程度"）。
    """

    # === 核心性格标签 ===
    cautious: float = 0.5     # 0.0~1.0 谨慎程度（高=保守，低=冒险）
    ambitious: float = 0.5     # 0.0~1.0 野心程度（高=追求目标，低=随遇而安）
    social: float = 0.5        # 0.0~1.0 社交倾向（高=喜欢互动，低=倾向于独处）

    # === 行动倾向标签（中文名，alias 兼容）===
    diligence: float = Field(0.5, alias="勤奋程度")     # 0.0~1.0 工作vs休息的倾向
    patience: float = Field(0.5, alias="耐心值")        # 0.0~1.0 能接受多久无聊/等待
    risk_tolerance: float = Field(0.5, alias="风险承受")  # 0.0~1.0 对危险的接受程度

    # === 社交标签 ===
    trust: float = Field(0.5, alias="信任度")          # 0.0~1.0 对陌生 NPC 的信任程度
    generosity: float = Field(0.5, alias="慷慨度")        # 0.0~1.0 分享资源/信息的意愿

    # === 特殊倾向（可扩展）===
    extra_tags: dict[str, Any] = Field(default_factory=dict, alias="标签集合")

    # 元信息
    updated_at: datetime = Field(default_factory=datetime.now)

    def set_tag(self, key: str, value: Any):
        """动态设置标签（支持中文 key）"""
        # 中文 key -> 英文映射
        chinese_map = {
            "勤奋程度": "diligence",
            "耐心值": "patience",
            "风险承受": "risk_tolerance",
            "信任度": "trust",
            "慷慨度": "generosity",
            "标签集合": "extra_tags",
        }
        english_key = chinese_map.get(key, key)
        if hasattr(self, english_key):
            setattr(self, english_key, value)
        else:
            self.extra_tags[key] = value
        self.updated_at = datetime.now()

    def get_tag(self, key: str, default: Any = None) -> Any:
        """获取标签值（支持中文 key）"""
        chinese_map = {
            "勤奋程度": "diligence",
            "耐心值": "patience",
            "风险承受": "risk_tolerance",
            "信任度": "trust",
            "慷慨度": "generosity",
            "标签集合": "extra_tags",
        }
        english_key = chinese_map.get(key, key)
        if hasattr(self, english_key):
            return getattr(self, english_key)
        return self.extra_tags.get(key, default)

    def to_prompt_string(self) -> str:
        """
        将标签转为 LLM prompt 字符串。
        例：cautious=0.8 → "性格谨慎(0.8/1.0)，决策前会深思熟虑"
        """
        tag_descriptions = [
            ("cautious", "谨慎", "冒进"),
            ("ambitious", "有野心", "随遇而安"),
            ("social", "善于社交", "倾向于独处"),
            ("diligence", "勤奋", "懒惰"),
            ("patience", "有耐心", "急躁"),
            ("risk_tolerance", "敢于冒险", "规避风险"),
            ("trust", "信任他人", "多疑"),
            ("generosity", "慷慨", "吝啬"),
        ]
        
        parts = []
        for field_name, high_desc, low_desc in tag_descriptions:
            value = getattr(self, field_name, 0.5)
            if value >= 0.7:
                parts.append(high_desc)
            elif value <= 0.3:
                parts.append(low_desc)
        
        return "、".join(parts) if parts else "普通性格"

    @classmethod
    def from_dict(cls, data) -> "PersonaTags":
        """从字典或 Pydantic 模型创建（支持中文 key）"""
        # 如果是 Pydantic 模型，转换为 dict
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        elif not isinstance(data, dict):
            data = dict(data)

        # 中文 -> 英文映射
        chinese_map = {
            "勤奋程度": "diligence",
            "耐心值": "patience",
            "风险承受": "risk_tolerance",
            "信任度": "trust",
            "慷慨度": "generosity",
            "标签集合": "extra_tags",
            "工作伦理": "work_ethic",
            "社会阶层": "social_class",
            "声誉": "reputation",
            "兴趣爱好": "interests",
            "性格特点": "personality",
            "特殊特质": "special_traits",
        }

        converted = {}
        for k, v in data.items():
            if k in chinese_map:
                converted[chinese_map[k]] = v
            elif k not in ("updated_at",):
                converted[k] = v

        return cls(**converted)


def generate_persona_from_behavior(
    memories: list[dict],
    recent_actions: list[str],
    role: str,
) -> PersonaTags:
    """
    根据行为历史自动生成/更新 Persona Tags。
    
    这是一个 stub 实现，后续会接入 LLM 进行更智能的总结。
    
    当前简单逻辑：
    - 分析最近 actions 的模式，推断勤奋程度、冒险程度等
    """
    tags = PersonaTags()

    if not recent_actions:
        return tags

    # 简单启发式推断
    travel_actions = sum(1 for a in recent_actions if "移动" in str(a))
    work_actions = sum(1 for a in recent_actions if any(k in str(a) for k in ["交易", "收获", "挖掘", "治疗"]))

    if work_actions / max(len(recent_actions), 1) > 0.6:
        tags.diligence = min(1.0, tags.diligence + 0.2)
    if travel_actions / max(len(recent_actions), 1) > 0.7:
        tags.risk_tolerance = max(0.0, tags.risk_tolerance - 0.1)

    # 角色默认值
    role_defaults = {
        "merchant": {"cautious": 0.6, "social": 0.8, "ambitious": 0.7},
        "farmer":   {"diligence": 0.8, "cautious": 0.7},
        "miner":    {"risk_tolerance": 0.8, "cautious": 0.3},
        "guard":    {"cautious": 0.7, "risk_tolerance": 0.6},
        "scholar":  {"cautious": 0.8, "social": 0.3},
        "healer":   {"generosity": 0.9, "social": 0.7},
    }
    if role in role_defaults:
        for k, v in role_defaults[role].items():
            setattr(tags, k, v)

    return tags