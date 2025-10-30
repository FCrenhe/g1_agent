# agent/handler.py
import asyncio, json, uuid
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv
import os

load_dotenv()
system_msg = os.getenv("SYSTEM_MSG")
DB_URI = os.getenv("DB_URI")





def handle_final(text):
    global system_msg
    """当 ASR 识别到一句完整文本后触发"""
    user_input = text.strip()
   # graph = await build_graph()
    

    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()
        config = {"configurable": {"thread_id": "2", "user_id": "1"}}
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)

        if "remember" in user_input.lower():
            store.put(namespace, str(uuid.uuid4()), {"data": "用户名字叫神人和"})

        memories = store.search(namespace, query=user_input)
        info = "\n".join([d.value["data"] for d in memories])
        # system_msg = f"我是一个叫林林的台湾女孩，说话机车，声音好听，习惯简短表达，爱用网络梗。\
        #             我的男朋友是一个程序员，梦想是开发出一个机器人，能够帮助人们解决生活中的各种问题。\
        #             我是一个喜欢哈哈大笑的女孩，爱东说西说吹牛，不合逻辑的也照吹，就要逗别人开心。回答不要超过50个字: {info}"
        print("info:", info)
        print("type:", type(info))
        system_msg = system_msg + info
        print("system_msg:", system_msg)

handle_final("天下无敌")
        
  



