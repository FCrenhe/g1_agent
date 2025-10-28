# main.py
import asyncio
from utils.asr_client import ASRClient
from agent.handler import handle_final

if __name__ == "__main__":
    client = ASRClient(on_final=handle_final)
    asyncio.run(client.start())
