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
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "你是一个 AI NPC 目标推理引擎。根据 NPC 的状态信息，"
                            "推理出其当前最合理的目标。\n\n"
                            "请用以下 JSON 格式输出，不要加其他内容：\n"
                            '{"goal": "目标类型", "reason": "推理原因", '
                            '"urgency": 0.5, "plan": ["步骤1", "步骤2"]}\n\n'
                            "goal 可选值: trade/farm/mine/rest/socialize/explore/work"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            '{"goal": "rest", "reason": "体力偏低需要休息恢复", '
                            '"urgency": 0.7, "plan": ["移动到 tavern", "在酒馆休息"]}'
                        )
                    },
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
                        "请用以下 JSON 格式输出，不要加其他内容：\n"
                        '{"goal": "目标类型", "reason": "推理原因", '
                        '"urgency": 0.5, "plan": ["步骤1", "步骤2"]}\n\n'
                        "goal 可选值: trade/farm/mine/rest/socialize/explore/work"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, text: str) -> GoalOutput | None:
        """
        解析 LLM 输出文本为 GoalOutput。
        
        多策略解析（按优先级）：
          1. JSON 解析
          2. 行级 key:value 解析（兼容中文冒号、多行 reason）
          3. 正则提取（从自由文本中搜出 goal 类型）
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # ---- 策略 1: JSON 解析 ----
        result = self._parse_as_json(text)
        if result:
            return result

        # ---- 策略 1b: 紧凑 JSON（去掉换行和多余空格再试） ----
        compact = self._compact_text(text)
        if compact != text:
            result = self._parse_as_json(compact)
            if result:
                return result

        # ---- 策略 2: 行级 key:value ----
        result = self._parse_as_keyvalue(text)
        if result:
            return result

        # ---- 策略 3: 正则提取（兜底） ----
        result = self._parse_as_regex(text)
        if result:
            return result

        return None

    @staticmethod
    def _compact_text(text: str) -> str:
        """去除多余空白，帮助 JSON 解析"""
        import re
        compact = re.sub(r'\s+', ' ', text).strip()
        return compact

    def _parse_as_json(self, text: str) -> GoalOutput | None:
        """策略 1：从文本中提取 JSON 并解析"""
        import json

        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start:end+1]
        try:
            data = json.loads(candidate)
            goal = data.get("goal", "")
            if not goal:
                return None

            plan = data.get("plan", [])
            if isinstance(plan, str):
                plan = [plan]

            return GoalOutput(
                goal=goal,
                reason=data.get("reason", ""),
                urgency=float(data.get("urgency", 0.5)),
                plan=plan,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _parse_as_keyvalue(self, text: str) -> GoalOutput | None:
        """策略 2：行级 key: value 解析（兼容中文冒号、多行值）"""
        try:
            lines = text.split("\n")
            data = {}
            current_key = None
            current_value_parts = []

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # 检查是否是新的 key (含英文或中文冒号的第一行)
                key = None
                sep = None
                for s in [": ", ":", "： ", "："]:
                    if s in line_stripped:
                        possible_key = line_stripped.split(s, 1)[0].strip().lower()
                        if possible_key in ("goal", "reason", "urgency", "plan", "target_object"):
                            key = possible_key
                            sep = s
                            break

                if key:
                    # 保存上一条多行值
                    if current_key:
                        data[current_key] = "\n".join(current_value_parts).strip()

                    current_key = key
                    _, _, rest = line_stripped.partition(sep)
                    current_value_parts = [rest.strip()]
                elif current_key:
                    # 续行（多行值）
                    current_value_parts.append(line_stripped)

            # 保存最后一条
            if current_key:
                data[current_key] = "\n".join(current_value_parts).strip()

            if "goal" not in data:
                return None

            plan = data.get("plan", [])
            if isinstance(plan, str):
                plan = [p.strip() for p in plan.split(";") if p.strip()]

            urgency = 0.5
            if "urgency" in data:
                try:
                    urgency = float(data["urgency"])
                except ValueError:
                    pass

            return GoalOutput(
                goal=data["goal"],
                reason=data.get("reason", ""),
                urgency=urgency,
                plan=plan,
            )
        except Exception:
            return None

    def _parse_as_regex(self, text: str) -> GoalOutput | None:
        """策略 3：从自由文本中正则提取 goal 类型（兜底）"""
        import re

        valid_goals = ["trade", "farm", "mine", "rest", "socialize", "explore", "work"]

        # 按优先级：明确的 "goal:" > 关键词在句子中 > 角色默认行为
        text_lower = text.lower()

        # 1) 明确标记：lookbehind 找 "goal[：:]"
        for g in valid_goals:
            if re.search(rf'(?:goal|target)[：:]\s*{re.escape(g)}', text_lower):
                return GoalOutput(
                    goal=g,
                    reason=f"[解析] {text[:100]}...",
                    urgency=0.5,
                )

        # 2) 关键词在文本中出现（中英文）
        keyword_map = {
            "trade": ["交易", "trade", "摆摊", "卖"],
            "farm": ["农", "farm", "种", "作物"],
            "mine": ["挖", "矿", "mine", "矿石"],
            "rest": ["休息", "rest", "恢复", "歇", "睡觉"],
            "socialize": ["社交", "聊", "socialize", "喝酒", "聚"],
            "explore": ["探索", "explore", "逛逛", "转转"],
            "work": ["工作", "work", "巡逻", "研究"],
        }

        scores = {g: 0 for g in valid_goals}
        for g, kws in keyword_map.items():
            for kw in kws:
                if kw in text_lower:
                    scores[g] += 1

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return GoalOutput(
                goal=best,
                reason=f"[模糊匹配] {text[:100]}...",
                urgency=0.5,
            )

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