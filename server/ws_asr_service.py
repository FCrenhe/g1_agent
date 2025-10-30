import asyncio
import json
from typing import Optional, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import opuslib_next
import uvicorn

# -----------------------------
# 配置
# -----------------------------
MODEL_DIR = "../models/SenseVoiceSmall"
SAMPLE_RATE = 16000        # 仅 16k 识别
CHANNELS = 1
CHUNK_MS = 200             # 建议客户端每 ~200ms 发送一块
INFER_INTERVAL = 0.8       # 多少秒触发一次增量识别
MAX_BUFFER_SECONDS = 30    # 避免越积越多，滑动窗口上限（秒）

# -----------------------------
# 初始化模型（仅加载一次）
# -----------------------------
model = AutoModel(
    model=MODEL_DIR,
    vad_kwargs={"max_single_segment_time": 30000},
    disable_update=True,
    hub="hf",
    # device="cuda:0",  # 如有GPU可启用
)

app = FastAPI(title="WS-ASR Service", version="1.0.0")


class StreamDecoder:
    """将到达的音频帧累积为 PCM（int16, 16k, mono）"""
    def __init__(self, mode: Literal["pcm", "opus"]):
        self.mode = mode
        self._pcm = bytearray()
        if mode == "opus":
            self._decoder = opuslib_next.Decoder(SAMPLE_RATE, CHANNELS)
            self._buffer_size = 960  # 60ms @16k
        else:
            self._decoder = None

    def append(self, data: bytes):
        if not data:
            return
        if self.mode == "pcm":
            # 直接拼接（假设客户端发送就是 int16 小端）
            self._pcm.extend(data)
        else:
            # 每次来一帧 Opus 包就解码
            try:
                pcm_frame = self._decoder.decode(data, self._buffer_size)
                if pcm_frame:
                    self._pcm.extend(pcm_frame)
            except opuslib_next.OpusError:
                # 跳过坏包
                pass

    def take_tail_pcm(self, seconds: float) -> bytes:
        """取末尾 N 秒 pcm（滑动窗口）"""
        bytes_per_sec = SAMPLE_RATE * 2 * CHANNELS
        need = int(bytes_per_sec * seconds)
        if len(self._pcm) <= need:
            return bytes(self._pcm)
        return bytes(self._pcm[-need:])

    def trim_keep_tail(self, seconds: float):
        """只保留末尾 N 秒，限制总长度"""
        bytes_per_sec = SAMPLE_RATE * 2 * CHANNELS
        keep = int(bytes_per_sec * seconds)
        if len(self._pcm) > keep:
            self._pcm = bytearray(self._pcm[-keep:])

    def all_pcm(self) -> bytes:
        return bytes(self._pcm)

    def clear(self):
        self._pcm.clear()


async def infer_loop(ws: WebSocket, decoder: StreamDecoder, stop_evt: asyncio.Event):
    """
    增量识别循环：定期对最近几秒音频做识别（例如最近 6 秒），返回 partial。
    连接关闭或 stop_evt 置位时退出。
    """
    # 为了稳定性：窗口太短不准，太长又慢；6~8s 比较稳妥
    WINDOW_SEC = 6.0

    while not stop_evt.is_set():
        await asyncio.sleep(INFER_INTERVAL)

        # 若连接已断开
        if ws.application_state != WebSocketState.CONNECTED:
            break

        # 从尾部取窗口音频，并做一次识别
        pcm = decoder.take_tail_pcm(WINDOW_SEC)
        if len(pcm) < SAMPLE_RATE * 2 * 1:  # 少于1秒就不跑
            continue

        try:
            result = model.generate(
                input=pcm,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            payload = {"type": "partial", "text": text}
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            # 不中断连接，回报一次错误消息即可
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                break

        # 控制总缓存长度，避免越积越多
        decoder.trim_keep_tail(MAX_BUFFER_SECONDS)


@app.websocket("/ws")
async def ws_asr(
    websocket: WebSocket,
    format: Literal["pcm", "opus"] = Query("pcm"),
):
    """
    WebSocket 实时识别：
    - 连接参数 format=pcm 或 opus（默认 pcm）
    - 客户端发送：二进制音频块（推荐每 ~200ms 一块）
    - 可选发送：文本消息 JSON 作为控制：
        {"event":"start"} | {"event":"end"} | {"event":"reset"}
    - 服务器返回：
        {"type":"partial","text": "..."}
        {"type":"final","text": "..."}
        {"type":"ready"} / {"type":"closed"} / {"type":"error", ...}
    """
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "ready", "format": format}))

    decoder = StreamDecoder(format)
    stop_evt = asyncio.Event()
    task = asyncio.create_task(infer_loop(websocket, decoder, stop_evt))

    try:
        while True:
            msg = await websocket.receive()
           # print("msg:", msg)
            if "bytes" in msg and msg["bytes"] is not None:
                # 音频帧
                decoder.append(msg["bytes"])
                # 继续等待更多帧
                continue

            if "text" in msg and msg["text"] is not None:
                # 控制消息
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    # 非 JSON 文本，忽略
                    continue

                event = data.get("event")
                if event == "start":
                    # 可用于客户端显式标记一段开始
                    await websocket.send_text(json.dumps({"type": "ack", "event": "start"}))
                elif event == "reset":
                    decoder.clear()
                    await websocket.send_text(json.dumps({"type": "ack", "event": "reset"}))
                elif event == "end":
                    # 输出最终结果（对全部缓存做一次最终识别）
                    pcm = decoder.all_pcm()
                    final_text = ""
                    if len(pcm) >= SAMPLE_RATE * 2 * 1:
                        try:
                            result = model.generate(
                                input=pcm,
                                cache={},
                                language="auto",
                                use_itn=True,
                                batch_size_s=60,
                            )
                            final_text = rich_transcription_postprocess(result[0]["text"])
                        except Exception as e:
                            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                    await websocket.send_text(json.dumps({"type": "final", "text": final_text}, ensure_ascii=False))
                    # 一段完成后可清空，继续下一段
                    decoder.clear()
                else:
                    # 其他自定义事件忽略或回显
                    await websocket.send_text(json.dumps({"type": "ack", "event": event}))
            # 其他情况（close 等）交给异常分支处理
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        stop_evt.set()
        try:
            await task
        except Exception:
            pass
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps({"type": "closed"}))
                await websocket.close()
            except Exception:
                pass


@app.get("/")
def root():
    return JSONResponse({"message": "WS-ASR is running. Connect to /ws?format=pcm|opus"})


if __name__ == "__main__":
    uvicorn.run("ws_asr_service:ws_asr:app", host="0.0.0.0", port=8000, reload=False)
