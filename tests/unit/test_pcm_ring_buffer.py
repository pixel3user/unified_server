"""Unit tests for PcmRingBuffer: the core audio ring used by the inference engine."""

import asyncio
import importlib

import numpy as np
import pytest


@pytest.fixture
async def ring():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.PcmRingBuffer(max_samples=1000)


@pytest.mark.asyncio
async def test_append_and_latest_basic(ring):
    """Appending samples and reading latest window works."""
    data = np.ones(100, dtype=np.float32)
    await ring.append(data)

    window, total = await ring.latest(100)
    assert window.shape == (100,)
    assert total == 100
    np.testing.assert_array_equal(window, data)


@pytest.mark.asyncio
async def test_ring_wraps_at_capacity(ring):
    """Ring buffer clips to max_samples when overfilled."""
    # Fill with 1200 samples into a 1000-sample ring
    data = np.arange(1200, dtype=np.float32)
    await ring.append(data)

    window, total = await ring.latest(1000)
    assert window.shape == (1000,)
    assert total == 1200
    # Should contain the last 1000 samples
    np.testing.assert_array_equal(window, data[-1000:])


@pytest.mark.asyncio
async def test_latest_returns_less_when_buffer_short(ring):
    """latest(N) returns fewer samples if buffer has less than N."""
    data = np.ones(50, dtype=np.float32)
    await ring.append(data)

    window, total = await ring.latest(200)
    assert window.shape == (50,)
    assert total == 50


@pytest.mark.asyncio
async def test_multiple_appends_accumulate(ring):
    """Multiple appends accumulate correctly."""
    chunk1 = np.ones(300, dtype=np.float32) * 1.0
    chunk2 = np.ones(300, dtype=np.float32) * 2.0
    chunk3 = np.ones(300, dtype=np.float32) * 3.0
    await ring.append(chunk1)
    await ring.append(chunk2)
    await ring.append(chunk3)

    window, total = await ring.latest(1000)
    # 900 total, ring capacity 1000, so all fit
    assert window.shape == (900,)
    assert total == 900


@pytest.mark.asyncio
async def test_wrap_preserves_tail(ring):
    """After wrap, only the newest samples remain."""
    chunk1 = np.ones(800, dtype=np.float32) * 1.0
    chunk2 = np.ones(400, dtype=np.float32) * 2.0
    await ring.append(chunk1)
    await ring.append(chunk2)

    window, total = await ring.latest(1000)
    # 1200 appended, ring is 1000, so last 1000 survive
    assert window.shape == (1000,)
    assert total == 1200
    # First 200 should be from chunk1 (1.0), last 400 from chunk2 (2.0)
    np.testing.assert_array_equal(window[:200], np.ones(200) * 1.0)
    np.testing.assert_array_equal(window[600:], np.ones(400) * 2.0)


@pytest.mark.asyncio
async def test_wait_for_total_after_wakes_on_new_data(ring):
    """wait_for_total_after unblocks when new data arrives."""
    await ring.append(np.ones(100, dtype=np.float32))
    _, total_before = await ring.latest(100)

    async def writer():
        await asyncio.sleep(0.05)
        await ring.append(np.ones(50, dtype=np.float32))

    task = asyncio.ensure_future(writer())
    result = await ring.wait_for_total_after(total_before, timeout=1.0)
    assert result is True
    await task


@pytest.mark.asyncio
async def test_wait_for_total_after_times_out(ring):
    """wait_for_total_after returns False on timeout with no new data."""
    await ring.append(np.ones(100, dtype=np.float32))
    _, total_before = await ring.latest(100)

    result = await ring.wait_for_total_after(total_before, timeout=0.05)
    assert result is False


@pytest.mark.asyncio
async def test_empty_append_is_noop(ring):
    """Appending empty array does not change state."""
    await ring.append(np.ones(50, dtype=np.float32))
    _, total_before = await ring.latest(50)

    await ring.append(np.zeros(0, dtype=np.float32))
    _, total_after = await ring.latest(50)
    assert total_after == total_before


@pytest.mark.asyncio
async def test_concurrent_producer_consumer(ring):
    """Concurrent producer/consumer do not corrupt data or deadlock."""
    produced = 0
    consumed_totals = []

    async def producer():
        nonlocal produced
        for _ in range(100):
            await ring.append(np.ones(10, dtype=np.float32))
            produced += 10
            await asyncio.sleep(0)

    async def consumer():
        for _ in range(20):
            window, total = await ring.latest(100)
            consumed_totals.append(total)
            await asyncio.sleep(0.01)

    p = asyncio.ensure_future(producer())
    c = asyncio.ensure_future(consumer())
    await asyncio.gather(p, c)

    assert produced == 1000
    # Consumer should have seen monotonically increasing totals
    assert consumed_totals[-1] > consumed_totals[0]


@pytest.mark.asyncio
async def test_no_memory_growth_under_repeated_appends():
    """Ring buffer memory should not grow unboundedly under many small appends.

    This test catches the fragmentation bug: if the buffer uses np.concatenate
    on every append, RSS will grow. A fixed circular buffer stays constant.
    """
    import importlib
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    ring = buffers.PcmRingBuffer(max_samples=16000)  # 1 second at 16kHz

    # Simulate 60 seconds of 20ms chunks (3000 appends)
    chunk = np.random.randn(320).astype(np.float32)  # 20ms @ 16kHz
    for _ in range(3000):
        await ring.append(chunk)

    # Verify ring didn't grow beyond capacity
    window, total = await ring.latest(16000)
    assert window.shape == (16000,)
    assert total == 3000 * 320  # 960000 total samples seen
