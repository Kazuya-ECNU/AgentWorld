"""
Goal Reasoner - LLM 推理引擎

支持 MiniMax（Anthropic 兼容格式）和 OpenAI 两种后端。
优先级：OPENAI_API_KEY > MINIMAX_API_KEY（从 openclaw.json 获取）
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


def _get_minimax_credentials() -> tuple[str, str]:
    """获取 MiniMax 凭证，优先级：环境变量 > agent models.json > openclaw.json"""
    # 先检查环境变量
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    base_url = os.environ.get("MINIMAX_BASE_URL", "").strip()
    
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        # 尝试从 agent 配置读取（与当前 agent 相同配置）
        import json
        for config_path in [
            os.path.expanduser("~/.openclaw/agents/coder/agent/models.json"),
            os.path.expanduser("~/.openclaw/agents/life/agent/models.json"),
            os.path.expanduser("~/.openclaw/openclaw.json"),
        ]:
            try:
                with open(config_path) as f:
                    raw = f.read()
                # 找到 providers 对象（以 { 开头，在 providers 关键字之后）
                idx = raw.find('"providers"')
                if idx < 0:
                    continue
                brace_idx = raw.find('{', idx)
                if brace_idx < 0:
                    continue
                partial = raw[brace_idx:]
                depth = 0
                end = 0
                for i, c in enumerate(partial):
                    if c == '{': depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end == 0:
                    continue
                providers = json.loads(partial[:end])
                minimax = providers.get("minimax", {})
                api_key = minimax.get("apiKey", "") or minimax.get("api_key", "")
                base_url = minimax.get("baseUrl", "") or minimax.get("base_url", "")
                if api_key:
                    break
            except Exception:
                continue
    
    if not base_url:
        base_url = "https://api.minimaxi.com/anthropic"
    
    return base_url, api_key


class GoalReasoner:
    """
    使用 LLM 推理 NPC 的目标。
    
    优先使用 MiniMax（Anthropic 兼容 API），无配置则使用 OpenAI。
    """

    def __init__(self, model: str | None = None):
        # 自动选择模型
        if model is None:
            model = "MiniMax-M2.7"
        self.model = model
        self._client = None  # httpx client（延迟初始化）
        self._provider: str = ""  # "minimax" | "openai"

    def _init_client(self):
        if self._client is not None:
            return
        
        # 尝试 MiniMax
        base_url, api_key = _get_minimax_credentials()
        if api_key and "minimax" in base_url.lower():
            import httpx
            self._client = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "anthropic-version": "2023-06-01",
                    "anthropic-dangerous-direct-browser-access": "true",
                },
                timeout=15.0,
            )
            self._provider = "minimax"
            return
        
        # 回退到 OpenAI
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                self._provider = "openai"
                return
            except ImportError:
                pass
        
        raise ValueError("No API key configured for LLM推理")

    def reason(self, prompt: str) -> GoalOutput | None:
        """
        给定 prompt，让 LLM 推理目标。
        
        Returns:
            GoalOutput if success, None if failed.
        """
        try:
            self._init_client()
        except ValueError as e:
            print(f"[GoalReasoner] 未配置 LLM: {e}")
            return None

        try:
            if self._provider == "minimax":
                return self._reason_minimax(prompt)
            else:
                return self._reason_openai(prompt)
        except Exception as e:
            print(f"[GoalReasoner] LLM 调用失败 ({self._provider}): {e}")
            return None
        finally:
            print(f"[GoalReasoner] reason() 结束, provider={self._provider}")

    def _reason_minimax(self, prompt: str) -> GoalOutput | None:
        """MiniMax Anthropic API"""
        response = self._client.post(
            "/v1/messages",
            json={
                "model": self.model,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "你是一个 AI NPC 目标推理引擎。根据 NPC 的状态信息，"
                            "推理出其当前最合理的目标。\n\n"
                            "输出格式：\n"
                            "goal: <目标类型>\n"
                            "reason: <推理原因>\n"
                            "urgency: <0.0~1.0>\n"
                            "plan: <步骤1>; <步骤2>"
                        )
                    },
                    {"role": "assistant", "content": "goal: rest\nreason: 能量低需要休息\nurgency: 0.7\nplan: 移动到 tavern; 在酒馆休息"},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        if response.status_code != 200:
            print(f"[GoalReasoner] MiniMax API 错误: {response.status_code} {response.text[:100]}")
            return None
        
        data = response.json()
        # MiniMax Anthropic 格式：content 是 list
        content_blocks = data.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                break
        
        return self._parse_response(text)

    def _reason_openai(self, prompt: str) -> GoalOutput | None:
        """OpenAI Chat Completions API"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个 AI NPC 目标推理引擎。根据 NPC 的状态信息，"
                        "推理出其当前最合理的目标。\n\n"
                        "输出格式：\n"
                        "goal: <目标类型>\n"
                        "reason: <推理原因>\n"
                        "urgency: <0.0~1.0>\n"
                        "plan: <步骤1>; <步骤2>"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return self._parse_response(response.choices[0].message.content)

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
                    try:
                        data["urgency"] = float(value)
                    except ValueError:
                        pass
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