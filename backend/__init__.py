# backend/__init__.py
# 将 backend.backend 子包中的模块暴露到 backend 命名空间下
# 使得 from backend.engine import ... 能正常工作

import sys

# 先导入子包模块
from backend.backend import engine
from backend.backend import llm_chat
from backend.backend import realtime
from backend.backend import weather_alerts

# 注册到 sys.modules，使 from backend.xxx 能正确解析
sys.modules["backend.engine"] = engine
sys.modules["backend.llm_chat"] = llm_chat
sys.modules["backend.realtime"] = realtime
sys.modules["backend.weather_alerts"] = weather_alerts
