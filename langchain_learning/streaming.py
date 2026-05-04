import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

load_dotenv()

# ===== 模型 =====
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

# ===== 工具 =====
@tool
def get_weather(city: str) -> str:
    """
    根据城市名称获取天气信息
    
    Args:
        city: 城市名称（例如：北京、上海、San Francisco）
    
    Returns:
        天气描述字符串
    """
    writer = get_stream_writer()
    writer(f"🔍 正在查找 {city} 的数据...")
    writer(f"✅ 已获取 {city} 的数据！")
    return f"☀️ {city} 终年阳光明媚！"

# ===== Agent =====
agent = create_agent(
    model=llm,   # ✅ 修复
    tools=[get_weather],
)

# ===== 输入 =====
input_message = {
    "messages": [{"role": "user", "content": "南昌现在的天气怎么样？"}]
}

print("🤖 Agent 开始工作...\n")

# ===== 流式处理 =====
for chunk in agent.stream(
    input_message,
    stream_mode=["updates", "messages", "custom"],
    version="v2"
):
    t = chunk["type"]
    data = chunk["data"]

    # ===== custom =====
    if t == "custom":
        print(f"\n[自定义] {data}")

    # ===== updates =====
    elif t == "updates":
        for step, info in data.items():
            if "messages" in info:
                msg = info["messages"][-1]

                if getattr(msg, "tool_calls", None):
                    print(f"\n🔄 调用工具: {msg.tool_calls[0]['name']}")
                    print(f"参数: {msg.tool_calls[0]['args']}")

    # ===== messages（核心）=====
    elif t == "messages":
        token, meta = data

        if hasattr(token, "text") and token.text:
            print(token.text, end="", flush=True)