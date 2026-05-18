"""Unit tests for StreamingLinearResampler: chunked audio resampling without clicks."""

import importlib

import numpy as np
import pytest


@pytest.fixture
def resampler_24k_to_16k():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.StreamingLinearResampler(src_rate=24000, dst_rate=16000)


@pytest.fixture
def resampler_24k_to_48k():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.StreamingLinearResampler(src_rate=24000, dst_rate=48000)


def test_empty_input_returns_empty(resampler_24k_to_16k):
    """Empty input produces empty output."""
    out = resampler_24k_to_16k.process(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,)


def test_24k_to_16k_ratio(resampler_24k_to_16k):
    """1 second of 24kHz input produces approximately 16000 samples of output."""
    one_second = np.random.randn(24000).astype(np.float32)
    out = resampler_24k_to_16k.process(one_second)
    # Allow +-10 samples tolerance for interpolation boundary effects
    assert abs(out.shape[0] - 16000) < 10


def test_24k_to_48k_ratio(resampler_24k_to_48k):
    """1 second of 24kHz input produces approximately 48000 samples of output."""
    one_second = np.random.randn(24000).astype(np.float32)
    out = resampler_24k_to_48k.process(one_second)
    assert abs(out.shape[0] - 48000) < 10


def test_chunked_equals_single_pass(resampler_24k_to_16k):
    """Processing in small chunks produces same result as one large chunk.

    This verifies phase continuity across chunk boundaries — the property
    that prevents audible clicks when websocket packets are resampled
    independently.
    """
    import importlib
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")

    # Generate 500ms of a 440Hz tone at 24kHz
    t = np.arange(12000, dtype=np.float32) / 24000.0
    signal = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    # Single-pass
    single_resampler = buffers.StreamingLinearResampler(src_rate=24000, dst_rate=16000)
    single_out = single_resampler.process(signal)

    # Chunked (40ms chunks = 960 samples at 24kHz, matching typical websocket packets)
    chunked_resampler = buffers.StreamingLinearResampler(src_rate=24000, dst_rate=16000)
    chunks = [signal[i : i + 960] for i in range(0, len(signal), 960)]
    chunked_parts = [chunked_resampler.process(c) for c in chunks]
    chunked_out = np.concatenate(chunked_parts)

    # They should be very close (linear interpolation is deterministic)
    min_len = min(len(single_out), len(chunked_out))
    np.testing.assert_allclose(single_out[:min_len], chunked_out[:min_len], atol=1e-6)


def test_small_chunks_no_zero_output():
    """Even very small chunks (160 samples) should eventually produce output."""
    import importlib
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    resampler = buffers.StreamingLinearResampler(src_rate=24000, dst_rate=16000)
    total_out = 0
    for _ in range(100):
        chunk = np.random.randn(160).astype(np.float32)
        out = resampler.process(chunk)
        total_out += out.shape[0]

    # 100 * 160 = 16000 input samples → should produce ~10666 output samples
    assert total_out > 10000


def test_dtype_is_float32(resampler_24k_to_16k):
    """Output is always float32 regardless of input dtype."""
    out = resampler_24k_to_16k.process(np.ones(960, dtype=np.float64))
    assert out.dtype == np.float32
