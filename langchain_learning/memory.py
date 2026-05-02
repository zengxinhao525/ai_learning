import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime

# 1. 加载环境变量
load_dotenv()

# 2. 获取配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME", "qwen-plus")

# 3. 初始化模型
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.7
)

# --- 1. 定义自定义状态 ---
class MyAgentState(AgentState):
    user_id: str

# --- 2. 定义记忆裁剪中间件 ---
@before_model
def trim_messages(state: MyAgentState, runtime: Runtime) -> Optional[Dict[str, Any]]:
    messages = state.get("messages", [])
    # 阈值调整为 6，超过 6 条才开始删
    if len(messages) <= 6:
        return None
    
    # 策略：保留最后 4 条消息（即最近 2 轮对话）
    # 这样比只保留 2 条更稳健
    recent_messages = messages[-4:]
    
    # 构建删除指令
    to_remove = []
    for msg in messages[:-4]:
        if msg.id:
            to_remove.append(RemoveMessage(id=msg.id))
            
    return {
        "messages": to_remove + recent_messages
    }

# --- 3. 初始化组件 ---
memory = InMemorySaver()

# 4. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[], 
    state_schema=MyAgentState, 
    middleware=[trim_messages], 
    checkpointer=memory 
)

# --- 5. 模拟对话 ---
config = {"configurable": {"thread_id": "thread_123"}}

print("=== 第一轮对话 ===")
result1 = agent.invoke(
    {
        "messages": HumanMessage(content="你好，我是南昌的小明。"),
        "user_id": "user_123"
    },
    config
)
print("AI:", result1["messages"][-1].content)

print("\n=== 第二轮对话 ===")
result2 = agent.invoke(
    {"messages": HumanMessage(content="我来自哪里？")},
    config
)
print("AI:", result2["messages"][-1].content)

print("\n=== 模拟长对话导致记忆裁剪 ===")
# 循环 10 次，确保触发裁剪逻辑
for i in range(10):
    agent.invoke(
        {"messages": HumanMessage(content=f"这是第 {i} 条无关消息，用来测试记忆裁剪。")},
        config
    )

print("\n=== 第三轮对话 (测试是否被裁剪) ===")
result3 = agent.invoke(
    {"messages": HumanMessage(content="我的名字叫什么？")},
    config
)
print("AI:", result3["messages"][-1].content)
# 注意：因为保留了最后 4 条，如果上面的循环刚好把“我是小明”挤出去了，AI 就会忘记。
# 这是符合预期的行为。