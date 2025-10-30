# agent/handler.py
import asyncio, json, uuid
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.store.postgres import PostgresStore
from .graph_builder import build_graph, DB_URI
from .tts_streaming import tts_stream
from dotenv import load_dotenv
import os

load_dotenv()
system_msg = os.getenv("SYSTEM_MSG")




async def stream_graph_updates(graph, input_messages, config, recorder):
  #  print("ASSISTANT:", end="", flush=True)
    full_text = "  "

    async for chunk in graph.astream({"messages": input_messages}, config, stream_mode="messages"):
        if isinstance(chunk[0], AIMessageChunk):
           # print(chunk[0].content, end="", flush=True)
            full_text += chunk[0].content
        elif isinstance(chunk[0], ToolMessage):
            print(f"\nTOOL: {chunk[0].content}", flush=True)
          #  print("ASSISTANT:", end="", flush=True)
   # print("\n")  # 换行
    return full_text.strip()  # 把完整回复返回


_graph_instance = None
async def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = await build_graph()
        
    return _graph_instance 
    
async def handle_final(self, text: str):
    """当 ASR 识别到一句完整文本后触发"""
    global system_msg
    user_input = text.strip()
    graph = await get_graph()
    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()
        config = {"configurable": {"thread_id": "2", "user_id": "1"}}
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)

        if "remember" in user_input.lower():
            store.put(namespace, str(uuid.uuid4()), {"data": "用户名字叫神人和"})
        memories = store.search(namespace, query=user_input)
        info = "\n".join([d.value["data"] for d in memories])
        system_msg = system_msg + info
        input_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input},
        ]
        
        reply = await stream_graph_updates(graph, input_messages, config, self)
        print('ASSISTANT:', reply)
        self.recording_enabled =False
        await asyncio.sleep(0.1)

        tts_stream(reply, reference_id="wanwan", temperature=0.3)
        await asyncio.sleep(0.3)

        self.recording_enabled =True




