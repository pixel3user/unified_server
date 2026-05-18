"""Unit tests for VideoFrameBuffer: the FIFO frame queue feeding WebRTC video."""

import asyncio
import importlib

import numpy as np
import pytest


@pytest.fixture
async def vbuf():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.VideoFrameBuffer(maxsize=4)


@pytest.mark.asyncio
async def test_publish_and_get(vbuf):
    """Basic publish/get cycle works."""
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    await vbuf.publish(frame)

    got = await vbuf.get(timeout=0.5)
    np.testing.assert_array_equal(got, frame)


@pytest.mark.asyncio
async def test_get_timeout_returns_last_frame(vbuf):
    """When queue is empty, get() returns last_frame after timeout."""
    # Set a known last_frame
    custom = np.ones((720, 1280, 3), dtype=np.uint8) * 42
    vbuf.last_frame = custom

    got = await vbuf.get(timeout=0.02)
    np.testing.assert_array_equal(got, custom)


@pytest.mark.asyncio
async def test_drop_oldest_when_full(vbuf):
    """When queue is full, publish drops oldest frame."""
    frames = [np.ones((720, 1280, 3), dtype=np.uint8) * i for i in range(6)]
    for f in frames:
        await vbuf.publish(f)

    # Queue capacity is 4, so first 2 should be dropped
    # We should get frames[2], frames[3], frames[4], frames[5]
    got = await vbuf.get(timeout=0.01)
    assert got[0, 0, 0] == 2  # oldest surviving frame


@pytest.mark.asyncio
async def test_last_frame_updated_on_publish(vbuf):
    """last_frame is always the most recently published frame."""
    frame1 = np.ones((720, 1280, 3), dtype=np.uint8) * 10
    frame2 = np.ones((720, 1280, 3), dtype=np.uint8) * 20
    await vbuf.publish(frame1)
    await vbuf.publish(frame2)

    np.testing.assert_array_equal(vbuf.last_frame, frame2)


@pytest.mark.asyncio
async def test_get_nowait_returns_oldest_or_last(vbuf):
    """get_nowait returns oldest queued frame or last_frame if empty."""
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 77
    await vbuf.publish(frame)

    got = await vbuf.get_nowait()
    np.testing.assert_array_equal(got, frame)

    # Now queue is empty, should return last_frame
    got2 = await vbuf.get_nowait()
    np.testing.assert_array_equal(got2, frame)  # last_frame was set to frame


@pytest.mark.asyncio
async def test_snapshot_jpeg_produces_valid_jpeg(vbuf):
    """snapshot_jpeg returns valid JPEG bytes."""
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    vbuf.last_frame = frame

    jpeg = vbuf.snapshot_jpeg()
    assert len(jpeg) > 0
    # JPEG magic bytes
    assert jpeg[:2] == b"\xff\xd8"
