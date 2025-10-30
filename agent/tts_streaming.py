import time
import ormsgpack
import pyaudio
import requests
#from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from pydantic import BaseModel, Field, conint, model_validator
from typing_extensions import Annotated
from typing import Literal
import base64
from dotenv import load_dotenv
import os

load_dotenv()

# 环境变量

TTS_URL = os.getenv("TTS_URL")

class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True


def tts_stream(text: str, reference_id: str = "0", temperature: float = 0.6):
    """流式合成 + 播放"""
    data = {
        "text": text,
        "references": [],  # 不传参考音频，使用 reference_id
        "reference_id": reference_id,
        "format": "wav",
        "max_new_tokens": 1024,
        "chunk_length": 300,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": temperature,
        "streaming": True,
        "use_memory_cache": "on",
        "seed": None,
    }

    pydantic_data = ServeTTSRequest(**data)

   # print(f"\n🟢 请求流式合成: {text}")
    start_time = time.time()
    response = requests.post(
        TTS_URL,
        params={"format": "msgpack"},
        data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=True,
        headers={"content-type": "application/msgpack"},
    )

    if response.status_code != 200:
        print("❌ 请求失败:", response.status_code, response.text)
        return

  #  print("✅ 开始接收音频流...\n")

    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    audio_format = pyaudio.paInt16
    stream = p.open(format=audio_format, channels=1, rate=44100, output=True)

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    end_time = time.time()
   # print(f"🎵 播放结束，用时 {end_time - start_time:.2f} 秒\n")


def main():
    print("🎤 Fish-Speech 交互式流式语音合成")
    print("输入文本后按回车即可合成；输入 `exit` 退出。")
    print("-" * 50)

    reference_id = input("请输入参考音色 ID（默认 0）: ").strip() or "0"

    while True:
        text = input("\n🗣️ 请输入要合成的文本：").strip()
        if text.lower() in ("exit", "quit", "q"):
            print("👋 退出程序。")
            break
        if not text:
            continue

        tts_stream(text, reference_id=reference_id, temperature=0.3)

# import re

# if __name__ == "__main__":
#     #main()
#     print("start")
    
#     text = "陈人和，今天天气怎么样啊？我刚在想要不要去夜市逛逛，顺便帮你的机器人找点灵感零件，哈哈！"
#     sentences = re.split(r'(?<=[。！？!?])', text)
#     sentences = [s.strip() for s in sentences if s.strip()]
#     for s in sentences:
#         print("-", s)
#         tts_stream(s, reference_id="wanwan", temperature=0.3)

