# -*- coding: utf-8 -*-
"""
LangChain Messages 综合演示 Demo
基于文档：Docum.txt
"""

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import json

# 1. 加载环境变量 (从 .env 文件)
from dotenv import load_dotenv
load_dotenv() 

# --- 基础设置 ---
# 注意：文档中使用的是 init_chat_model，这里假设使用 GPT-4o (支持多模态)
model = init_chat_model(
    model="qwen3.5-plus",  # 请确保这是百炼支持的模型名
    model_provider="openai"  # 因为我们使用的是 OpenAI 兼容模式
)

# --- 定义工具 (Tool) ---
# 文档中提到的 ToolMessage 需要配合工具使用
def get_weather(location: str) -> str:
    """模拟天气查询工具"""
    # 这里通常会调用真实的 API，为了演示，我们返回模拟数据
    return f"南昌市 {location} 当前天气：晴朗，气温 22°C。"

# 让模型“知道”这个工具
model_with_tools = model.bind_tools([get_weather])

print("=== 场景 1：基础多轮对话 ===")

# 构建对话历史
messages = [
    SystemMessage("你是一个友好的南昌本地导游，说话带有南昌方言特色。"),
    HumanMessage("你好啊，我是来南昌旅游的！"),
    AIMessage("哎呀，欢迎来南昌咯！"),
    HumanMessage("今天南昌天气怎么样？")
]

# 调用模型
response = model_with_tools.invoke(messages)
print("AI 回复:", response.content)
# 输出示例：南昌的天气老好哦，适合出去恰饭！