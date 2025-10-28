# agent/tools.py
import asyncio
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.custom_tools import multiply, speak, weather

async def load_tools():
    """加载本地工具 + MCP 远程工具"""
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["utils/rh_mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools_mcp = await client.get_tools()

    search_tool = TavilySearch(max_results=2)
    return [multiply, speak, weather] + tools_mcp
