import asyncio
import json
import threading
import websockets


class TranscriptStreamServer:
    def __init__(self, port: int = 8766):
        self._port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run, daemon=True, name="TranscriptStream").start()
        print(f"[TranscriptStream] Listening on ws://localhost:{self._port}")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, "localhost", self._port):
            await asyncio.Future()

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    def broadcast(self, text: str, end_time: float) -> None:
        if not self._clients or not self._loop:
            return
        msg = json.dumps({"type": "transcript", "text": text, "end_time": end_time})
        asyncio.run_coroutine_threadsafe(self._broadcast_all(msg), self._loop)

    async def _broadcast_all(self, msg: str) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead
