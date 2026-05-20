"""Component tests for FakeMoshiServer.

The FakeMoshiServer is test infrastructure: future bridge / E2E tests will
rely on it to simulate PersonaPlex without booting Moshi. So it deserves
its own coverage to lock in its protocol contract.

Protocol contract (mirrors real PersonaPlex):
- WebSocket endpoints: /api/chat (duplex), /api/avatar/audio (mirror).
- First downstream binary message is the handshake (kind=0).
- Subsequent binary messages are audio frames (kind=1 + opus payload, or
  kind=1 + raw f32le PCM if sphn is unavailable).
- Server reads uplink audio and discards it (just counts bytes).
- /api/voices returns a JSON list (used by the runtime route proxy).

Configurable failure modes verified here:
- handshake_delay_ms: tests can simulate slow voice loading.
- drop_after_n_packets: tests can simulate mid-call websocket drop.
- reject_connections: tests can simulate Moshi being overloaded.
- require_voice_prompt: tests can simulate missing query params.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import aiohttp
import pytest

# Path setup so `tests.fakes.moshi_server` resolves
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_ROOT = Path(__file__).resolve().parent.parent
for p in [str(UNIFIED_ROOT), str(TESTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fakes.moshi_server import FakeMoshiServer


@pytest.fixture
async def server():
    """Start a FakeMoshiServer on a random port; teardown after each test."""
    s = FakeMoshiServer(audio_chunk_ms=40.0, audio_cadence_ms=40.0)
    await s.start()
    try:
        yield s
    finally:
        await s.stop()


class TestStartupShutdown:
    """Lifecycle: bind random port, accept connections, clean teardown."""

    @pytest.mark.asyncio
    async def test_start_returns_port(self):
        s = FakeMoshiServer()
        port = await s.start()
        try:
            assert port > 0
            assert s.port == port
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with FakeMoshiServer() as s:
            assert s.port > 0


class TestChatHandshake:
    """The /api/chat endpoint must send the handshake byte before any audio."""

    @pytest.mark.asyncio
    async def test_first_message_is_handshake(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/chat?text_prompt=hi&voice_prompt=v.pt"
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                assert msg.type == aiohttp.WSMsgType.BINARY
                assert msg.data == b"\x00", "First message should be handshake (kind=0)"

    @pytest.mark.asyncio
    async def test_records_query_params(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/chat?text_prompt=hello&voice_prompt=v.pt"
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                # Receive handshake, then close
                await asyncio.wait_for(ws.receive(), timeout=2.0)
                await ws.close()
        assert server.last_text_prompt == "hello"
        assert server.last_voice_prompt == "v.pt"
        assert server.connections_total >= 1


class TestHandshakeDelay:
    """A configurable handshake delay simulates slow voice loading."""

    @pytest.mark.asyncio
    async def test_delay_blocks_handshake(self):
        """With a 200ms delay, the handshake should arrive after ~200ms."""
        s = FakeMoshiServer(handshake_delay_ms=200.0)
        await s.start()
        try:
            url = f"ws://127.0.0.1:{s.port}/api/chat"
            t0 = asyncio.get_event_loop().time()
            async with aiohttp.ClientSession() as cs:
                async with cs.ws_connect(url) as ws:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    elapsed = asyncio.get_event_loop().time() - t0
            assert msg.data == b"\x00"
            # Generous bound: at least 100ms (allow scheduler jitter), well
            # under the 2s test timeout.
            assert elapsed >= 0.1, f"Handshake arrived in {elapsed:.3f}s, expected >=0.1s"
        finally:
            await s.stop()


class TestAudioStreaming:
    """After handshake, the server should stream audio packets at the configured cadence."""

    @pytest.mark.asyncio
    async def test_streams_audio_packets_after_handshake(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/chat"
        received_audio = 0
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                # Burn the handshake
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                assert msg.data == b"\x00"
                # Now collect audio packets for ~150ms
                deadline = asyncio.get_event_loop().time() + 0.15
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type == aiohttp.WSMsgType.BINARY and msg.data and msg.data[0] == 1:
                        received_audio += 1
        # 150ms at 40ms cadence → ~3-4 packets
        assert received_audio >= 2, f"Expected >=2 audio packets in 150ms, got {received_audio}"

    @pytest.mark.asyncio
    async def test_packet_count_matches_telemetry(self, server):
        """The server's `packets_sent_total` should match what the client receives."""
        url = f"ws://127.0.0.1:{server.port}/api/chat"
        client_received = 0
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                # Skip handshake
                await asyncio.wait_for(ws.receive(), timeout=2.0)
                deadline = asyncio.get_event_loop().time() + 0.15
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type == aiohttp.WSMsgType.BINARY and msg.data and msg.data[0] == 1:
                        client_received += 1
        # Allow off-by-one; the server counts at send time and the client
        # may exit the loop before receiving the very last in-flight packet.
        assert abs(server.packets_sent_total - client_received) <= 1


class TestUplinkConsumption:
    """The duplex /api/chat endpoint should accept uplink audio bytes."""

    @pytest.mark.asyncio
    async def test_uplink_bytes_received(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/chat"
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                # Send some fake uplink "audio" (raw bytes; server discards)
                payload = b"\x01" + b"\x00" * 1024
                await ws.send_bytes(payload)
                await ws.send_bytes(payload)
                # Give the server a moment to drain
                await asyncio.sleep(0.05)
                await ws.close()
        assert server.uplink_bytes_received >= 2 * 1025


class TestDropAfterNPackets:
    """The drop_after_n_packets knob simulates mid-call connection loss."""

    @pytest.mark.asyncio
    async def test_drop_after_3_packets(self):
        s = FakeMoshiServer(audio_cadence_ms=20.0, drop_after_n_packets=3)
        await s.start()
        try:
            url = f"ws://127.0.0.1:{s.port}/api/chat"
            audio_packets = 0
            close_seen = False
            async with aiohttp.ClientSession() as cs:
                async with cs.ws_connect(url) as ws:
                    # Burn handshake
                    await asyncio.wait_for(ws.receive(), timeout=2.0)
                    # Read until close
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        except asyncio.TimeoutError:
                            break
                        if msg.type == aiohttp.WSMsgType.BINARY and msg.data and msg.data[0] == 1:
                            audio_packets += 1
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                            close_seen = True
                            break
            assert audio_packets <= 3, f"Got {audio_packets} packets, expected <=3"
            assert close_seen or audio_packets == 3
        finally:
            await s.stop()


class TestRejectConnections:
    """The reject_connections knob simulates Moshi being overloaded."""

    @pytest.mark.asyncio
    async def test_rejected_immediately(self):
        s = FakeMoshiServer(reject_connections=True)
        await s.start()
        try:
            url = f"ws://127.0.0.1:{s.port}/api/chat"
            async with aiohttp.ClientSession() as cs:
                async with cs.ws_connect(url) as ws:
                    # Server should close almost immediately. We may receive
                    # a CLOSE frame or just see the connection drop.
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    assert msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ), f"Unexpected msg type: {msg.type}"
        finally:
            await s.stop()


class TestRequireVoicePrompt:
    """When require_voice_prompt=True, missing the query param should fail."""

    @pytest.mark.asyncio
    async def test_missing_voice_prompt_returns_400(self):
        s = FakeMoshiServer(require_voice_prompt=True)
        await s.start()
        try:
            # Use a plain HTTP GET to verify the 400 (the websocket upgrade
            # will not happen because the response is a regular Response).
            async with aiohttp.ClientSession() as cs:
                # No voice_prompt in query → should 400
                async with cs.get(f"http://127.0.0.1:{s.port}/api/chat") as resp:
                    assert resp.status == 400
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_with_voice_prompt_succeeds(self):
        s = FakeMoshiServer(require_voice_prompt=True)
        await s.start()
        try:
            url = f"ws://127.0.0.1:{s.port}/api/chat?voice_prompt=test.pt"
            async with aiohttp.ClientSession() as cs:
                async with cs.ws_connect(url) as ws:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    assert msg.data == b"\x00"  # handshake
        finally:
            await s.stop()


class TestVoicesEndpoint:
    """The /api/voices proxy returns a JSON list."""

    @pytest.mark.asyncio
    async def test_voices_returns_list(self, server):
        async with aiohttp.ClientSession() as cs:
            async with cs.get(f"http://127.0.0.1:{server.port}/api/voices") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "voices" in data
                assert isinstance(data["voices"], list)
                assert len(data["voices"]) > 0


class TestMirrorEndpoint:
    """The /api/avatar/audio (mirror) endpoint streams audio without uplink."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_mirror_handshake_and_audio(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/avatar/audio"
        got_handshake = False
        got_audio = False
        async with aiohttp.ClientSession() as cs:
            async with cs.ws_connect(url) as ws:
                # First two binary frames should be handshake then audio.
                # We bound this with explicit per-receive timeouts and an
                # outer break to avoid blocking on the server's infinite stream.
                for _ in range(20):
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.3)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.BINARY:
                        continue
                    if msg.data == b"\x00":
                        got_handshake = True
                        continue
                    if msg.data and msg.data[0] == 1:
                        got_audio = True
                        break
                # Explicit close so the server-side _stream_audio loop exits.
                await ws.close()
        assert got_handshake, "Mirror endpoint did not send handshake byte"
        assert got_audio, "Mirror endpoint did not stream any audio packets"


class TestMultipleConcurrentConnections:
    """The server should handle several simultaneous clients."""

    @pytest.mark.asyncio
    async def test_three_concurrent_clients(self, server):
        url = f"ws://127.0.0.1:{server.port}/api/chat"

        async def client():
            async with aiohttp.ClientSession() as cs:
                async with cs.ws_connect(url) as ws:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    return msg.data == b"\x00"

        results = await asyncio.gather(client(), client(), client())
        assert all(results)
        assert server.connections_total >= 3
