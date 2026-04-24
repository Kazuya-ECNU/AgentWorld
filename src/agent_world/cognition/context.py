"""
Context Builder - 为 LLM 构建输入上下文

将 NPC 的数据（Persona Tags, Memory, Recent Context, Knowledge）
组装成 LLM 可处理的 prompt 结构。
"""

from typing import Optional
from .persona import PersonaTags
from .memory import MemoryStore


class ContextBuilder:
    """
    将 NPC 状态数据构建为 LLM 推理所需的上下文。
    
    使用方式：
      builder = ContextBuilder(npc)
      prompt = builder.build()
      
    输出结构供 GoalReasoner 使用。
    """

    def __init__(
        self,
        name: str,
        role: str,
        persona: PersonaTags,
        memory: MemoryStore,
        recent_context_n: int = 5,
    ):
        self.name = name
        self.role = role
        self.persona = persona
        self.memory = memory
        self.recent_context_n = recent_context_n

    def build_system_prompt(self) -> str:
        """
        构建系统级 prompt（角色设定）。
        这部分相对固定，所有 NPC 共享同一模板。
        """
        return (
            f"你是一个生活在一个 AI 世界中的 NPC。你的角色是 {self.role}。\n"
            f"你需要根据当前的状态和目标，决定自己接下来要做什么。\n"
            f"你的决策应该符合你的性格标签，并且基于你过去的记忆和当前的环境。\n"
        )

    def build_persona_string(self) -> str:
        """构建性格标签描述"""
        return f"性格特点：{self.persona.to_prompt_string()}"

    def build_memory_summary(self) -> str:
        """
        构建记忆摘要（给 LLM 阅读的完整记忆流）。
        
        格式：[时间顺序的记忆列表]
        """
        entries = self.memory.get_all()
        if not entries:
            return "这个 NPC 还没有任何记忆。"

        lines = []
        for e in entries:
            time_str = e.timestamp.strftime("%H:%M")
            lines.append(f"[{time_str}] {e.event}")
        return "\n".join(lines)

    def build_recent_context(self) -> str:
        """构建最近上下文（滑动窗口）"""
        return self.memory.recent_context_string(self.recent_context_n)

    def build_knowledge_string(self, known_zones: list[str], known_npcs: list[str]) -> str:
        """构建 NPC 对世界的认知"""
        zone_str = "、".join(known_zones) if known_zones else "尚不了解任何地点"
        npc_str = "、".join(known_npcs) if known_npcs else "尚不认识其他 NPC"
        return f"熟悉的地点：{zone_str}；认识的其他 NPC：{npc_str}"

    def build_state_summary(self, **state_fields) -> str:
        """
        构建当前状态摘要。
        
        state_fields 示例：
          energy=50, gold=10, inventory=["铁矿石"], position="market"
        """
        parts = []
        for key, value in state_fields.items():
            if value is None or value == "":
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            parts.append(f"{key}={value}")
        return "、".join(parts) if parts else "状态正常"

    def build_full_prompt(
        self,
        known_zones: list[str],
        known_npcs: list[str],
        **state_fields
    ) -> dict[str, str]:
        """
        构建完整的 prompt 结构（供 reasoner 使用）。
        
        Returns:
            dict with keys: system_prompt, persona, memory, recent_context, knowledge, state
        """
        return {
            "system_prompt": self.build_system_prompt(),
            "persona": self.build_persona_string(),
            "memory_summary": self.build_memory_summary(),
            "recent_context": self.build_recent_context(),
            "knowledge": self.build_knowledge_string(known_zones, known_npcs),
            "state": self.build_state_summary(**state_fields),
        }

    def format_for_llm(self, **kwargs) -> str:
        """
        将所有上下文格式化为一个完整的字符串（供直接发送给 LLM）。
        """
        ctx = self.build_full_prompt(**kwargs)
        return (
            f"{ctx['system_prompt']}\n\n"
            f"【性格标签】{ctx['persona']}\n\n"
            f"【全部记忆】\n{ctx['memory_summary']}\n\n"
            f"【最近发生的事】{ctx['recent_context']}\n\n"
            f"【当前状态】{ctx['state']}\n\n"
            f"【对世界的了解】{ctx['knowledge']}\n\n"
            f"基于以上信息，你现在的目标是什么？请简要说明原因。"
        )