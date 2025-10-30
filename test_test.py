import asyncio
import aiohttp
import ormsgpack
import pyaudio
import io
import wave
import time
from pydantic import BaseModel, Field, ConfigDict, conint, model_validator
from typing_extensions import Annotated
from typing import Literal, List, Optional
import base64

# -----------------------------
# Pydantic 模型
# -----------------------------
class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if isinstance(audio, str) and len(audio) > 255:
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                pass
        return values


class ServeTTSRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    format: Literal["wav"] = "wav"           # ✅ 只支持 WAV
    references: List[ServeReferenceAudio] = []
    reference_id: Optional[str] = None
    seed: Optional[int] = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = True
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8


TTS_URL = "http://192.168.1.143:6006/v1/tts"


# -----------------------------
# 异步流式 TTS
# -----------------------------
async def tts_stream_async(text: str, reference_id="wanwan", temperature=0.6):
    data = {
        "text": text,
        "references": [],
        "reference_id": reference_id,
        "format": "wav",
        "streaming": True,
        "use_memory_cache": "on",
        "chunk_length": 300,
        "temperature": temperature,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "max_new_tokens": 1024,
    }

    payload = ormsgpack.packb(ServeTTSRequest(**data), option=ormsgpack.OPT_SERIALIZE_PYDANTIC)

    print(f"\n🟢 异步请求流式合成: {text}")
    start_time = time.time()

    p = pyaudio.PyAudio()
    stream = None
    sample_rate = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TTS_URL,
                params={"format": "msgpack"},
                data=payload,
                headers={"content-type": "application/msgpack"},
            ) as resp:
                if resp.status != 200:
                    print("❌ 请求失败:", resp.status, await resp.text())
                    return

                print("✅ 开始接收音频流...\n")

                async for chunk, _ in resp.content.iter_chunks():
                    if not chunk:
                        continue

                    # 第一次解析 WAV 头获取采样率
                    if sample_rate is None and chunk[:4] == b"RIFF":
                        try:
                            with wave.open(io.BytesIO(chunk)) as wf:
                                sample_rate = wf.getframerate()
                                print(f"🎚️ 检测采样率: {sample_rate} Hz")
                                # 初始化 PyAudio 播放器
                                stream = p.open(
                                    format=pyaudio.paInt16,
                                    channels=wf.getnchannels(),
                                    rate=sample_rate,
                                    output=True,
                                )
                                pcm = wf.readframes(wf.getnframes())
                                await asyncio.to_thread(stream.write, pcm)
                                continue
                        except Exception:
                            pass

                    # 之后的块直接剥掉 header
                    if chunk[:4] == b"RIFF":
                        idx = chunk.find(b"data")
                        if idx != -1:
                            chunk = chunk[idx + 8 :]
                    if stream:
                        await asyncio.to_thread(stream.write, chunk)

    except Exception as e:
        print(f"⚠️ 播放错误: {e}")

    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    print(f"🎵 播放结束，用时 {time.time() - start_time:.2f} 秒\n")



async def main():
    await tts_stream_async("你好，我是异步语音合成测试。", reference_id="wanwan", temperature=0.3)

if __name__ == "__main__":
    asyncio.run(tts_stream_async("你好，我是异步语音合成测试。"))
