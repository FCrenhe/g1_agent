# agent/graph_builder.py
import os, json, pathlib, uuid, asyncio
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI
from .tools import load_tools

load_dotenv()

# 环境变量
open_ai_kpi_key = os.getenv("open_ai_api_key")
DB_URI = os.getenv("DB_URI")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def route_tools(state: State):
    """判断是否需要调用工具"""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("state 中没有 messages")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

async def build_graph():
    """构建 LangGraph 智能体图"""
    tools = await load_tools()

    llm = ChatOpenAI(
        model="qwen-plus-2025-07-14",
        openai_api_key=open_ai_kpi_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
        timeout=30,
    )

    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "chatbot")

    checkpointer = InMemorySaver()
    graph = graph_builder.compile(checkpointer)
    return graph
