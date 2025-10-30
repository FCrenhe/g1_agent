import asyncio
import json
import sounddevice as sd
import websockets
import numpy as np
import webrtcvad
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# 环境变量

WS_URL = os.getenv("WS_URL")
class ASRClient:
    def __init__(
        self,
        ws_url=WS_URL,
        sample_rate=16000,
        chunk_ms=320,
        frame_ms=32,
        vad_mode=3,
        on_final=None,
    ):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.channels = 1
        self.chunk_ms = chunk_ms
        self.frame_ms = frame_ms

        # 音频参数
        self.frames = int(sample_rate * chunk_ms / 1000)
        self.frame_size = int(sample_rate * frame_ms / 1000) * 2  # bytes (16bit=2)
        self.vad = webrtcvad.Vad(vad_mode)
        
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (_, _, _, VADIterator, _) = utils
        self.vad_iterator = VADIterator(model)
        self.vad_model = model
        
        

        # 状态变量
        self.audio_queue = asyncio.Queue()
        self.silence_frames = 0
        self.speech_detected = False
       # self.loop = asyncio.get_event_loop()
        self.running = False
        self.on_final = on_final
        self.energy_threshold = 300
        
        self.recording_enabled =True

    def _audio_callback(self, indata, frames, time, status, loop):
        
        """录音线程的回调函数"""
        
        
        if not getattr(self, "recording_enabled", True):
           # print("正在播放，停止录音")
            return
        
        if status:
            print("⚠️ 音频状态:", status)

        pcm_bytes = indata.astype(np.int16).tobytes()

        for i in range(0, len(pcm_bytes), self.frame_size):
            frame = pcm_bytes[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
            




            audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)
            result = self.vad_iterator(audio_tensor)
            # if speech_prob is not None:
            #     print("speech_prob:", speech_prob)
            
            if isinstance(result, dict):
                if "start" in result:
                  #  print(f"🎙️ 检测到语音开始")
                    self.speech_detected = True
                elif "end" in result:
                  #  print(f"🔇 检测到语音结束")
                    self.speech_detected = False
                    asyncio.run_coroutine_threadsafe(self.audio_queue.put(None), loop)
            
            if getattr(self, "speech_detected", False):
                #self.audio_queue.put_nowait(chunk.astype(np.int16).tobytes())
                self.audio_queue.put_nowait(frame)
            # is_speech = self.vad.is_speech(frame, self.sample_rate)
            # audio_data = np.frombuffer(frame, dtype=np.int16)
            # energy = np.mean(np.abs(audio_data))
            # is_valid_speech = is_speech and energy > self.energy_threshold

            # if is_valid_speech:
            #     self.silence_frames = 0
            #     self.speech_detected = True
            #     self.audio_queue.put_nowait(frame)
            # else:
            #     self.silence_frames += 1
            #     if self.speech_detected and self.silence_frames > 50:
            #         asyncio.run_coroutine_threadsafe(
            #             self.audio_queue.put(None), loop
            #         )
            #         self.speech_detected = False
            #       #  print("\n 检测到静音 -> 发送 None")
                  
                  
            

    async def _sender(self, ws):
        """负责发送音频数据"""
        while self.running:
            data = await self.audio_queue.get()
            if data is None:
                await ws.send(json.dumps({"event": "end"}))
              #  print("\n 已发送 end 事件")
                continue
            await ws.send(data)
            


    async def _receiver(self, ws):
        """负责接收识别结果"""
        async for msg in ws:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                print("\n[非JSON消息]", msg)
                continue

            msg_type = data.get("type")
            text = data.get("text", "")

            if msg_type == "ready":
                print("服务器就绪，开始传输音频。")
            elif msg_type == "partial":
                continue
               # print(f"\r用户：{text}", end="", flush=True)
            elif msg_type == "final":
                print(f"用户:{text}")
                if self.on_final:
                    await self.on_final(self, text)
            elif msg_type == "error":
                print("\n 错误：", data)
                break
            elif msg_type == "closed":
                print("\n连接关闭")
                break
            else:
                print("\n[未知消息]", data)

    async def start(self):
        """启动录音和识别"""
        self.running = True
        async with websockets.connect(self.ws_url, max_size=2**23) as ws:
            ready = await ws.recv()
            print("connected:", ready)
            await ws.send(json.dumps({"event": "start"}))

            send_task = asyncio.create_task(self._sender(ws))
            recv_task = asyncio.create_task(self._receiver(ws))
            global_loop = asyncio.get_running_loop()

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.frames,
               # callback=lambda *args: self._audio_callback(*args),
                callback=lambda *args: self._audio_callback(*args, loop=global_loop)

            ):
                print("🎙️ 开始录音，按 Ctrl+C 停止")
                try:
                    while self.running:
                        await asyncio.sleep(0.1)
                except KeyboardInterrupt:
                    print("手动中断")

            # 停止录音时
            await self.audio_queue.put(None)
            await ws.send(json.dumps({"event": "end"}))
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    def stop(self):
        """手动停止"""
        self.running = False
        print("已停止语音识别客户端")



# async def handle_final(text):
#     print("\n[外部接收到最终识别结果]", text)
#   #  await asyncio.sleep(0.1)
    
    
    
# if __name__ == "__main__":
#     client = ASRClient(on_final=handle_final)
#     asyncio.run(client.start())


















