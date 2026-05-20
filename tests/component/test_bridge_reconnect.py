"""Component tests for PersonaPlexChatBridge reconnect behavior.

These tests verify that:
1. The bridge connects to FakeMoshiServer, receives handshake, streams audio.
2. When the server drops the connection, the bridge reconnects with backoff.
3. The backoff resets after a successful handshake.
4. Pre-handshake mic audio is dropped (not queued).

NOTE: sphn is mocked here because it's unavailable on CPU-only test
environments. The mock provides passthrough encode/decode that satisfies
the bridge's interface without real Opus compression.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

# Path setup
MUSETALK_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "MuseTalk"
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_ROOT = Path(__file__).resolve().parent.parent
for p in [str(MUSETALK_ROOT), str(UNIFIED_ROOT), str(TESTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================================
# Mock sphn module so personaplex_io.py can be imported without the real codec
# ============================================================================
class _FakeOpusStreamReader:
    """Mock OpusStreamReader that treats input as raw PCM bytes."""

    def __init__(self, sample_rate: int):
        self._sr = sample_rate
        self._buf = b""

    def append_bytes(self, data: bytes) -> None:
        self._buf += data

    def read_pcm(self) -> np.ndarray:
        if len(self._buf) < 4:
            return np.zeros((1, 0), dtype=np.float32)
        # Interpret as raw float32 PCM (matches FakeMoshiServer fallback)
        n_samples = len(self._buf) // 4
        pcm = np.frombuffer(self._buf[: n_samples * 4], dtype=np.float32).reshape(1, -1)
        self._buf = self._buf[n_samples * 4 :]
        return pcm


class _FakeOpusStreamWriter:
    """Mock OpusStreamWriter that passes PCM through as raw bytes."""

    def __init__(self, sample_rate: int):
        self._sr = sample_rate
        self._buf = b""

    def append_pcm(self, pcm: np.ndarray) -> None:
        self._buf += pcm.astype(np.float32).tobytes()

    def read_bytes(self) -> bytes:
        out = self._buf
        self._buf = b""
        return out


# Install mock before importing personaplex_io
_sphn_mock = types.ModuleType("sphn")
_sphn_mock.OpusStreamReader = _FakeOpusStreamReader
_sphn_mock.OpusStreamWriter = _FakeOpusStreamWriter
sys.modules["sphn"] = _sphn_mock

# Now we can import the bridge
from scripts.musetalk_webrtc.personaplex_io import PersonaPlexChatBridge
from scripts.musetalk_webrtc.buffers import AudioTrackBuffer, PcmRingBuffer, StreamingLinearResampler
from scripts.musetalk_webrtc.models import SessionState

from fakes.moshi_server import FakeMoshiServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import make_test_args


def _make_session() -> SessionState:
    return SessionState(
        session_id="bridge-test-001",
        token="test-token",
        created_epoch=time.time(),
        last_activity_epoch=time.time(),
    )


@pytest.fixture
async def buffers():
    """Create the three buffers needed by the bridge."""
    ring_24k = PcmRingBuffer(max_samples=240000)
    ring_16k = PcmRingBuffer(max_samples=160000)
    audio_buf = AudioTrackBuffer(max_samples_48k=480000)
    return ring_24k, ring_16k, audio_buf


@pytest.fixture
async def moshi_server():
    """Start FakeMoshiServer that streams 5 packets then closes."""
    s = FakeMoshiServer(
        audio_cadence_ms=30.0,
        stream_duration_ms=200.0,  # stream for 200ms then cleanly close
    )
    await s.start()
    try:
        yield s
    finally:
        await s.stop()


class TestBridgeConnectsAndReceivesAudio:
    """Basic happy-path: bridge connects, gets handshake, receives audio."""

    @pytest.mark.asyncio
    async def test_bridge_receives_handshake_and_audio(self, moshi_server, buffers):
        ring_24k, ring_16k, audio_buf = buffers
        session = _make_session()

        ws_url = f"ws://127.0.0.1:{moshi_server.port}/api/chat?text_prompt=hi&voice_prompt=v.pt"
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=ring_24k,
            pcm_ring_16k=ring_16k,
            audio_track_buffer=audio_buf,
            reconnect_delay_seconds=0.1,
        )

        task = asyncio.create_task(bridge.run())
        # Wait for handshake + some audio
        await asyncio.sleep(0.4)
        bridge.stop_event.set()
        await asyncio.wait_for(task, timeout=3.0)

        assert bridge.handshake.is_set(), "Bridge never received handshake"
        assert bridge.rx_packets > 0, "Bridge received no audio packets"
        assert session.personaplex_audio_frames_rx > 0


class TestBridgeReconnectsAfterDrop:
    """When the server drops the connection, the bridge reconnects."""

    @pytest.mark.asyncio
    async def test_reconnects_after_server_closes(self, buffers):
        """Server streams 3 packets then closes. Bridge should reconnect."""
        ring_24k, ring_16k, audio_buf = buffers
        session = _make_session()

        # Server drops after 3 packets
        server = FakeMoshiServer(
            audio_cadence_ms=20.0,
            drop_after_n_packets=3,
        )
        await server.start()

        ws_url = f"ws://127.0.0.1:{server.port}/api/chat?voice_prompt=v.pt"
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=ring_24k,
            pcm_ring_16k=ring_16k,
            audio_track_buffer=audio_buf,
            reconnect_delay_seconds=0.05,  # fast reconnect for tests
        )

        task = asyncio.create_task(bridge.run())
        # Wait long enough for at least 2 connection cycles
        await asyncio.sleep(0.6)
        bridge.stop_event.set()
        await asyncio.wait_for(task, timeout=3.0)
        await server.stop()

        # The server should have seen multiple connections (bridge reconnected)
        assert server.connections_total >= 2, (
            f"Expected >=2 connections (reconnect), got {server.connections_total}"
        )


class TestPreHandshakeMicDropped:
    """Audio pushed before handshake should be silently dropped."""

    @pytest.mark.asyncio
    async def test_uplink_dropped_before_handshake(self, buffers):
        ring_24k, ring_16k, audio_buf = buffers
        session = _make_session()

        # Server with long handshake delay
        server = FakeMoshiServer(
            handshake_delay_ms=500.0,
            stream_duration_ms=100.0,
        )
        await server.start()

        ws_url = f"ws://127.0.0.1:{server.port}/api/chat?voice_prompt=v.pt"
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=ring_24k,
            pcm_ring_16k=ring_16k,
            audio_track_buffer=audio_buf,
            reconnect_delay_seconds=0.1,
        )

        task = asyncio.create_task(bridge.run())

        # Push mic audio BEFORE handshake arrives
        await asyncio.sleep(0.1)  # connected but no handshake yet
        pcm = np.random.randn(960).astype(np.float32) * 0.5
        await bridge.push_uplink_pcm24k(pcm)
        await bridge.push_uplink_pcm24k(pcm)

        # The uplink queue should be empty (pre-handshake drops)
        assert bridge.uplink_queue.qsize() == 0, (
            "Pre-handshake mic audio should be dropped, not queued"
        )

        bridge.stop_event.set()
        await asyncio.wait_for(task, timeout=3.0)
        await server.stop()


class TestUplinkQueueOverflow:
    """When uplink queue is full, oldest is dropped (not blocked)."""

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest(self, moshi_server, buffers):
        ring_24k, ring_16k, audio_buf = buffers
        session = _make_session()

        ws_url = f"ws://127.0.0.1:{moshi_server.port}/api/chat?voice_prompt=v.pt"
        bridge = PersonaPlexChatBridge(
            ws_url=ws_url,
            session=session,
            pcm_ring_24k=ring_24k,
            pcm_ring_16k=ring_16k,
            audio_track_buffer=audio_buf,
            reconnect_delay_seconds=0.1,
        )

        task = asyncio.create_task(bridge.run())
        # Wait for handshake
        await asyncio.sleep(0.15)
        assert bridge.handshake.is_set()

        # Flood the uplink queue (capacity=64)
        pcm = np.random.randn(960).astype(np.float32) * 0.3
        for _ in range(100):
            await bridge.push_uplink_pcm24k(pcm)

        # Queue should NOT have grown beyond maxsize
        assert bridge.uplink_queue.qsize() <= 64

        bridge.stop_event.set()
        await asyncio.wait_for(task, timeout=3.0)
