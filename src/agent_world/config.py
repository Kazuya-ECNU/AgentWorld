# Agent World Config

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据库路径
DB_PATH = PROJECT_ROOT / "data" / "agent_world.db"

# API 配置
API_HOST = "0.0.0.0"
API_PORT = 8765

# WebSocket 配置
WS_HOST = "0.0.0.0"
WS_PORT = 8766

# LLM 配置（Phase 3 开始使用）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")