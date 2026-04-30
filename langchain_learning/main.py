import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 1. 加载 .env 文件中的变量
# 这行代码会读取当前目录下的 .env 文件并注入到环境变量中
load_dotenv()

# 2. 获取配置
# 这里读取我们在 .env 里定义的变量名
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME", "qwen-plus") # 如果没配环境变量，默认用 qwen-plus

# 3. 初始化模型(更精细方式，除了这个还可以用from langchain.chat_models import init_chat_model)
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.7
)

# 4. 定义工具
def get_weather(city: str) -> str:
    """
    获取指定城市的当前天气信息。
    
    Args:
        city: 城市名称，例如 "北京" 或 "Shanghai"。
        
    Returns:
        描述天气的字符串。
    """
    return f"杭州（{city}）今天天气晴朗，气温 25°C！"

# 5. 创建并运行 Agent
agent = create_agent(llm, tools=[get_weather], system_prompt="你是一个智能助手。")

result = agent.invoke({"messages": [{"role": "user", "content": "杭州的天气怎么样？"}]})
print(result["messages"][-1].content)