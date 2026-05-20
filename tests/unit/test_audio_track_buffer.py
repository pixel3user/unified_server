"""Unit tests for AudioTrackBuffer: 48kHz output buffer for WebRTC audio track."""

import asyncio
import importlib

import numpy as np
import pytest


@pytest.fixture
def atbuf():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.AudioTrackBuffer(max_samples_48k=48000)  # 1 second


@pytest.mark.asyncio
async def test_append_and_pop(atbuf):
    """append_from_24k upsamples and pop_48k returns correct frame."""
    # 960 samples at 24kHz → should become ~1920 at 48kHz
    mono24k = np.ones(960, dtype=np.float32) * 0.5
    await atbuf.append_from_24k(mono24k)

    # Pop standard WebRTC frame (960 samples = 20ms @ 48kHz)
    frame = await atbuf.pop_48k(960)
    assert frame.shape == (960,)
    assert frame.dtype == np.float32
    # Should contain non-zero values from the upsampled signal
    assert np.any(frame != 0.0)


@pytest.mark.asyncio
async def test_pop_underflow_zero_pads(atbuf):
    """pop_48k zero-pads when buffer has fewer samples than requested."""
    # Append just 100 samples worth of 24k (→ ~200 at 48k)
    mono24k = np.ones(100, dtype=np.float32) * 0.3
    await atbuf.append_from_24k(mono24k)

    frame = await atbuf.pop_48k(960)
    assert frame.shape == (960,)
    # First part should be non-zero, tail should be zero-padded
    assert np.any(frame[:150] != 0.0)
    assert np.all(frame[250:] == 0.0)


@pytest.mark.asyncio
async def test_pop_empty_returns_zeros(atbuf):
    """pop_48k on empty buffer returns all zeros."""
    frame = await atbuf.pop_48k(960)
    assert frame.shape == (960,)
    np.testing.assert_array_equal(frame, np.zeros(960, dtype=np.float32))


@pytest.mark.asyncio
async def test_capacity_clipping(atbuf):
    """Buffer clips to max_samples when overfilled."""
    # Append 2 seconds of 24kHz (48000 samples) → 96000 at 48kHz, but cap is 48000
    big = np.ones(48000, dtype=np.float32) * 0.1
    await atbuf.append_from_24k(big)

    # Buffer should be clipped
    async with atbuf.lock:
        assert atbuf.buf.size <= atbuf.max_samples


@pytest.mark.asyncio
async def test_empty_append_noop(atbuf):
    """Appending empty array doesn't change buffer."""
    await atbuf.append_from_24k(np.zeros(0, dtype=np.float32))
    frame = await atbuf.pop_48k(960)
    np.testing.assert_array_equal(frame, np.zeros(960, dtype=np.float32))
