"""Fake PersonaPlex/Moshi server for GPU-free integration testing.

Provides a minimal aiohttp server that mimics the PersonaPlex websocket
protocol (/api/chat duplex and /api/avatar/audio mirror) without any
real model inference.

Usage in tests:
    async with FakeMoshiServer() as server:
        ws_url = f"ws://127.0.0.1:{server.port}/api/chat?text_prompt=hi&voice_prompt=test.pt"
        # ...connect your bridge to ws_url...
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Optional

import numpy as np
from aiohttp import web

# Attempt to use sphn for Opus encoding (same codec as real PersonaPlex).
# Falls back to raw PCM framing if sphn is unavailable.
try:
    import sphn

    SPHN_AVAILABLE = True
except ImportError:
    SPHN_AVAILABLE = False


def _generate_tone_pcm(
    duration_ms: float = 40.0,
    freq_hz: float = 440.0,
    sample_rate: int = 24000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a pure sine tone chunk at 24kHz mono float32."""
    n_samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


class FakeMoshiServer:
    """Minimal fake PersonaPlex server for testing bridge/mirror clients.

    Configurable behaviors:
    - handshake_delay_ms: simulate slow system-prompt processing before handshake.
    - drop_after_n_packets: simulate abrupt websocket close mid-stream.
    - audio_chunk_ms: duration of each downstream audio chunk.
    - audio_cadence_ms: interval between sending chunks (simulates real-time).
    - stream_duration_ms: total audio to stream before cleanly closing (0 = infinite).
    - require_voice_prompt: return 400 if voice_prompt query param missing.
    - reject_connections: immediately close any websocket (simulates overload).
    """

    def __init__(
        self,
        *,
        handshake_delay_ms: float = 0.0,
        drop_after_n_packets: int = -1,
        audio_chunk_ms: float = 40.0,
        audio_cadence_ms: float = 40.0,
        stream_duration_ms: float = 0.0,
        require_voice_prompt: bool = False,
        reject_connections: bool = False,
        tone_freq_hz: float = 440.0,
    ):
        self.handshake_delay_ms = handshake_delay_ms
        self.drop_after_n_packets = drop_after_n_packets
        self.audio_chunk_ms = audio_chunk_ms
        self.audio_cadence_ms = audio_cadence_ms
        self.stream_duration_ms = stream_duration_ms
        self.require_voice_prompt = require_voice_prompt
        self.reject_connections = reject_connections
        self.tone_freq_hz = tone_freq_hz

        # Telemetry (readable by tests after run)
        self.connections_total = 0
        self.packets_sent_total = 0
        self.uplink_bytes_received = 0
        self.last_text_prompt: Optional[str] = None
        self.last_voice_prompt: Optional[str] = None

        # Track open websockets so stop() can close them deterministically
        # instead of waiting for each handler's stream loop to notice client close.
        self._open_websockets: set = set()

        self._app = web.Application()
        self._app.router.add_get("/api/chat", self._handle_chat)
        self._app.router.add_get("/api/avatar/audio", self._handle_mirror)
        self._app.router.add_get("/api/voices", self._handle_voices)
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self.port: int = 0

    async def start(self) -> int:
        """Start the fake server on a random available port. Returns port number."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host="127.0.0.1", port=0)
        await self._site.start()
        # Extract the actual bound port
        sock = self._site._server.sockets[0]
        self.port = sock.getsockname()[1]
        return self.port

    async def stop(self) -> None:
        """Gracefully shut down the fake server."""
        # Close any in-flight websockets first so handler tasks can exit promptly.
        for ws in list(self._open_websockets):
            with __import__("contextlib").suppress(Exception):
                await ws.close()
        self._open_websockets.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    def _encode_audio_packet(self, pcm24k: np.ndarray) -> bytes:
        """Encode PCM to wire format: kind=1 byte + opus payload."""
        if SPHN_AVAILABLE:
            writer = sphn.OpusStreamWriter(24000)
            writer.append_pcm(pcm24k)
            opus_bytes = writer.read_bytes()
            if opus_bytes:
                return b"\x01" + opus_bytes
            return b""
        else:
            # Fallback: raw f32le framing (only for unit tests without sphn)
            return b"\x01" + pcm24k.astype(np.float32).tobytes()

    async def _handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        """Handle duplex /api/chat websocket (mirrors PersonaPlex protocol)."""
        self.connections_total += 1
        self.last_text_prompt = request.query.get("text_prompt")
        self.last_voice_prompt = request.query.get("voice_prompt")

        if self.require_voice_prompt and not self.last_voice_prompt:
            return web.Response(status=400, text="voice_prompt required")

        if self.reject_connections:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close()
            return ws

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._open_websockets.add(ws)

        try:
            # Simulate handshake delay (system prompt / voice loading)
            if self.handshake_delay_ms > 0:
                await asyncio.sleep(self.handshake_delay_ms / 1000.0)

            # Send handshake (kind=0, empty payload)
            await ws.send_bytes(b"\x00")

            # Start concurrent recv (consume uplink) and send (stream audio)
            recv_task = asyncio.create_task(self._consume_uplink(ws))
            try:
                await self._stream_audio(ws)
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
        finally:
            self._open_websockets.discard(ws)

        return ws

    async def _handle_mirror(self, request: web.Request) -> web.WebSocketResponse:
        """Handle read-only /api/avatar/audio websocket (mirror mode)."""
        self.connections_total += 1
        self.last_text_prompt = request.query.get("text_prompt")
        self.last_voice_prompt = request.query.get("voice_prompt")

        if self.reject_connections:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close()
            return ws

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._open_websockets.add(ws)

        try:
            # Handshake
            if self.handshake_delay_ms > 0:
                await asyncio.sleep(self.handshake_delay_ms / 1000.0)
            await ws.send_bytes(b"\x00")

            # Stream audio (no uplink expected in mirror mode)
            await self._stream_audio(ws)
        finally:
            self._open_websockets.discard(ws)
        return ws

    async def _handle_voices(self, request: web.Request) -> web.Response:
        """Return fake voice list (for testing /api/voices proxy)."""
        return web.json_response({"voices": ["default.pt", "myvoice.pt"]})

    async def _consume_uplink(self, ws: web.WebSocketResponse) -> None:
        """Read and discard uplink audio from browser mic."""
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    self.uplink_bytes_received += len(msg.data)
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass

    async def _stream_audio(self, ws: web.WebSocketResponse) -> None:
        """Stream synthetic audio chunks at configured cadence.

        Responsive to client close: when ws.send_bytes raises (peer closed)
        we exit promptly. We also break the cadence sleep into small slices
        so a graceful close handshake can complete in <100ms instead of
        having to wait out the full audio_cadence_ms tick.
        """
        packets_sent = 0
        start = time.monotonic()

        while not ws.closed:
            # Check packet limit
            if self.drop_after_n_packets >= 0 and packets_sent >= self.drop_after_n_packets:
                await ws.close()
                return

            # Check duration limit
            if self.stream_duration_ms > 0:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                if elapsed_ms >= self.stream_duration_ms:
                    await ws.close()
                    return

            # Generate and send one audio chunk
            pcm = _generate_tone_pcm(
                duration_ms=self.audio_chunk_ms,
                freq_hz=self.tone_freq_hz,
            )
            packet = self._encode_audio_packet(pcm)
            if packet:
                try:
                    await ws.send_bytes(packet)
                    packets_sent += 1
                    self.packets_sent_total += 1
                except (ConnectionResetError, asyncio.CancelledError):
                    return
                except Exception:
                    # aiohttp can raise generic errors when the peer is
                    # mid-close-handshake. Treat as graceful exit.
                    return

            # Pace at configured cadence, but in slices so we notice ws.closed
            # promptly when the client closes mid-cadence.
            target = self.audio_cadence_ms / 1000.0
            slice_size = 0.02  # 20ms slices
            slept = 0.0
            while slept < target and not ws.closed:
                await asyncio.sleep(min(slice_size, target - slept))
                slept += slice_size
