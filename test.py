# agent/handler.py
import asyncio, json, uuid
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.store.postgres import PostgresStore
from agent.graph_builder import build_graph, DB_URI
from agent.tts_streaming import tts_stream




async def stream_graph_updates(graph, input_messages, config, recorder):
    print("ASSISTANT:", end="", flush=True)
    full_text = "  "

    async for chunk in graph.astream({"messages": input_messages}, config, stream_mode="messages"):
        if isinstance(chunk[0], AIMessageChunk):
          #  print(chunk[0].content, end="", flush=True)
            full_text += chunk[0].content
        #    print("full_text:", full_text)
        elif isinstance(chunk[0], ToolMessage):
            print(f"\nTOOL: {chunk[0].content}", flush=True)
            print("ASSISTANT:", end="", flush=True)
    print("\n")  # 换行
    return full_text.strip()  # 把完整回复返回
async def handle_final(text: str):
    """当 ASR 识别到一句完整文本后触发"""
    print("go")
    user_input = text.strip()
    graph = await build_graph()
    print("jjj")
    with PostgresStore.from_conn_string(DB_URI) as store:
     #   print("kkkk")
        store.setup()
        config = {"configurable": {"thread_id": "2", "user_id": "1"}}
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)

        if "remember" in user_input.lower():
            store.put(namespace, str(uuid.uuid4()), {"data": "用户名字叫神人和"})

        memories = store.search(namespace, query=user_input)
        info = "\n".join([d.value["data"] for d in memories])
        system_msg = f"我是一个叫林林的台湾女孩，说话机车，声音好听，习惯简短表达，爱用网络梗。\
                    我的男朋友是一个程序员，梦想是开发出一个机器人，能够帮助人们解决生活中的各种问题。\
                    我是一个喜欢哈哈大笑的女孩，爱东说西说吹牛，不合逻辑的也照吹，就要逗别人开心。回答不要超过50个字: {info}"

        input_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input},
        ]
        
        print("fffff")
        reply = await stream_graph_updates(graph, input_messages, config, self)
        print('reply:', reply)
        # self.recording_enabled =False
        # await asyncio.sleep(0.1)

        # tts_stream(reply, reference_id="wanwan", temperature=0.3)
        # await asyncio.sleep(0.1)

        # self.recording_enabled =True




if __name__ == "__main__":
    asyncio.run(handle_final("你好，记住我的名字是神人和"))