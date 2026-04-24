"""
Goal Reasoner - LLM 推理引擎

将 NPC 的上下文数据发给 LLM，推理出目标（Goal）。

Goal 格式：
  - goal: str          # 目标类型 (trade, rest, farm, socialize, explore...)
  - reason: str        # 推理原因
  - target_object: str | None  # 目标物体
  - target_npc: str | None    # 目标 NPC
  - urgency: float     # 紧迫度 0.0~1.0
  - plan: list[str]   # 执行计划（可选）
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import os


class GoalOutput(BaseModel):
    """LLM 输出的 Goal 结构"""
    goal: str = Field(..., description="目标类型，如 trade/rest/farm/explore/socialize")
    reason: str = Field(..., description="推理原因，为什么是这个目标")
    target_object: Optional[str] = Field(None, description="目标物体 ID")
    target_npc: Optional[str] = Field(None, description="目标 NPC ID")
    urgency: float = Field(0.5, ge=0.0, le=1.0, description="紧迫度")
    plan: list[str] = Field(default_factory=list, description="执行计划步骤")
    created_at: datetime = Field(default_factory=datetime.now)


class GoalReasoner:
    """
    使用 LLM 推理 NPC 的目标。
    
    流程：
    1. ContextBuilder 构建输入
    2. 发送给 LLM
    3. 解析响应为 GoalOutput
    4. 失败时 fallback 到规则引擎
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None  # 延迟初始化

    def _get_client(self):
        """延迟初始化 LLM client"""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def reason(self, prompt: str) -> GoalOutput | None:
        """
        给定 prompt，让 LLM 推理目标。
        
        Returns:
            GoalOutput if success, None if failed.
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个生活模拟世界的 AI NPC 推理引擎。"
                            "根据 NPC 的性格标签、记忆、当前状态等信息，"
                            "推理出 NPC 当前最合理的目标。\n\n"
                            "输出格式要求：\n"
                            "goal: <目标类型>\n"
                            "reason: <推理原因>\n"
                            "urgency: <0.0~1.0>\n"
                            "plan: <步骤1>; <步骤2>; ..."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return self._parse_response(response.choices[0].message.content)

        except Exception as e:
            print(f"[GoalReasoner] LLM 调用失败: {e}")
            return None

    def _parse_response(self, text: str) -> GoalOutput | None:
        """解析 LLM 输出文本为 GoalOutput"""
        try:
            lines = text.strip().split("\n")
            data = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "goal":
                    data["goal"] = value
                elif key == "reason":
                    data["reason"] = value
                elif key == "urgency":
                    data["urgency"] = float(value)
                elif key == "plan":
                    data["plan"] = [p.strip() for p in value.split(";") if p.strip()]

            if "goal" in data:
                return GoalOutput(
                    goal=data.get("goal", "idle"),
                    reason=data.get("reason", ""),
                    target_object=data.get("target_object"),
                    target_npc=data.get("target_npc"),
                    urgency=data.get("urgency", 0.5),
                    plan=data.get("plan", []),
                )
        except Exception as e:
            print(f"[GoalReasoner] 解析失败: {e}")
        return None

    def reason_for_npc(self, formatted_prompt: str) -> GoalOutput | None:
        """封装后的 NPC 推理接口"""
        return self.reason(formatted_prompt)


# === 内置 Goal 类型常量 ===

class GoalType:
    """预定义的 Goal 类型"""
    IDLE = "idle"
    TRADE = "trade"
    FARM = "farm"
    MINE = "mine"
    REST = "rest"
    SOCIALIZE = "socialize"
    EXPLORE = "explore"
    WORK = "work"
    MOVE = "move"


def is_goal_valid(goal: GoalOutput) -> bool:
    """检查 Goal 输出是否有效"""
    return (
        goal.goal is not None
        and goal.goal != ""
        and len(goal.goal) <= 30
    )