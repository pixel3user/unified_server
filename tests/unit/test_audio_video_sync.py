"""Property tests documenting the 'avatar speed up' bug.

Bug summary
-----------
The user reports that during voice replies, the avatar's lip-sync visually
"skips ahead" or "speeds up" relative to the audio. Two root causes were
identified by reading the engine source:

1. **Silent audio drops on bursts**.
   When a burst of audio arrives in `pcm_ring_16k` (e.g., PersonaPlex flushes
   its TTS queue after a long reply, or the bridge reconnects with backlog),
   the engine's `_run()` loop:
     - reads `total = pcm_ring.total_samples` (now includes the whole burst);
     - sets `last_total_samples = total` immediately, marking the whole burst
       as "consumed" from the engine's perspective;
     - then clamps `new_samples` to `max_advance_samples` (default 240ms).
   The dropped portion never gets inferred over — but it DOES still play
   through the separate `audio_track_buffer` (which has no such clamp).
   Result: the user hears words for which no lip-sync was generated, and
   the avatar's mouth "jumps ahead" to the position matching the last 240ms.

2. **Frame-cap drift**.
   With defaults `max_advance_ms=240`, `max_tail_frames=5`, `fps=25`:
     - `raw_new_frames = round(0.240 * 25) = 6`
     - `capped_new_frames = min(6, 5) = 5`
   Each inference cycle thus consumes 240ms of audio but only publishes
   5*(1/25)=200ms of video → 40ms of drift per cycle, accumulating.

Why xfail
---------
Both fixes touch the GPU inference path (`_infer_window_frames`) and need to
be validated against real model outputs before deploying — silently changing
the per-cycle frame count could destabilize the temporal mouth-smoothing
filter (`mouth_smoothing_alpha`) or break the avatar timeline.

These tests are written against `FakeInferenceEngine`, which faithfully
mirrors the buggy engine behavior. When the real engine is patched (and
mirrored in FakeInferenceEngine), these xfail markers should be removed.
At that point pytest will report XPASS, signaling the fix landed correctly.

Suggested fix direction
-----------------------
- Replace `self.last_total_samples = total` with
  `self.last_total_samples += new_samples` (after clamp) so leftover audio
  is processed on the next cycle instead of being silently dropped.
- Remove `max_tail_frames` cap (or auto-derive it from
  `max_advance_ms * fps / 1000`) so video frame count matches consumed audio.
"""

from __future__ import annotations

import asyncio
import importlib

import numpy as np
import pytest

import sys
from pathlib import Path

MUSETALK_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "MuseTalk"
if str(MUSETALK_ROOT) not in sys.path:
    sys.path.insert(0, str(MUSETALK_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import make_test_args


@pytest.fixture
async def ring():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.PcmRingBuffer(max_samples=16000 * 30)  # 30s capacity


@pytest.fixture
async def video_buffer():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    # Large enough to NOT lose frames to drop-oldest during the test, so we
    # can measure exactly how many the engine published.
    return buffers.VideoFrameBuffer(maxsize=4096)


def _frames_drained_count(video_buffer) -> int:
    """Count frames currently sitting in the queue, draining as we go."""
    n = 0
    while True:
        try:
            video_buffer.queue.get_nowait()
            n += 1
        except asyncio.QueueEmpty:
            break
    return n


@pytest.mark.asyncio
async def test_no_silent_audio_drops_on_burst(ring, video_buffer):
    """A 1-second audio burst should be fully inferred over, not silently dropped.

    Today's behavior: dropped_audio_ms_total > 0 when a burst exceeds
    max_advance_ms (which is the entire premise of the clamp). After the fix,
    the engine should consume the burst gradually across multiple cycles
    instead of dropping the overflow.
    """
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    args = make_test_args(window_ms=640, hop_ms=80, max_advance_ms=240, fps=25)
    engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=2.0)

    # Feed a 1-second burst all at once (16000 samples)
    burst = np.random.randn(16000).astype(np.float32) * 0.5
    await ring.append(burst)

    # Run engine long enough for the burst to be fully processed under any
    # reasonable per-cycle budget.
    task = asyncio.create_task(engine.run())
    await asyncio.sleep(1.5)
    engine.stop_event.set()
    await task

    # Today this fails with dropped_audio_ms_total ≈ 760ms.
    # After fix: should be 0 (or very small from rounding).
    assert engine.dropped_audio_ms_total < 50.0, (
        f"Engine silently dropped {engine.dropped_audio_ms_total:.1f}ms of audio "
        "but the audio_track_buffer played all of it — this is the visible 'speed up' bug."
    )


@pytest.mark.asyncio
async def test_published_frames_match_consumed_audio(ring, video_buffer):
    """Per-cycle, published frame count should equal the frames implied by the
    audio consumed — i.e., (consumed_audio_ms / 1000) * fps.

    Today: capped at max_tail_frames=5, but max_advance_ms=240 implies 6 frames
    at fps=25. So we publish 5/6 of what we should, every cycle.
    """
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    fps = 25
    max_advance_ms = 240
    max_tail_frames = 5
    args = make_test_args(
        window_ms=640,
        hop_ms=80,
        max_advance_ms=max_advance_ms,
        fps=fps,
        max_tail_frames=max_tail_frames,
    )
    engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=2.0)

    # Compute expected: a single 240ms burst should produce ceil(240/1000*25)=6 frames
    burst_ms = max_advance_ms
    burst_samples = int((burst_ms / 1000.0) * 16000)
    expected_frames_per_burst = int(round((burst_ms / 1000.0) * fps))

    # Run a single burst → single inference cycle
    burst = np.random.randn(burst_samples).astype(np.float32) * 0.5
    await ring.append(burst)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.3)
    engine.stop_event.set()
    await task

    actual_frames = engine.frames_published
    # Today: actual_frames == 5, expected_frames_per_burst == 6.
    # After fix: actual_frames should == expected_frames_per_burst.
    assert actual_frames >= expected_frames_per_burst, (
        f"Published {actual_frames} frames for {burst_ms}ms of audio; "
        f"expected at least {expected_frames_per_burst} at {fps}fps. "
        "This is the frame-cap drift component of the speed-up bug."
    )


@pytest.mark.asyncio
async def test_audio_video_drift_stays_bounded_under_load(ring, video_buffer):
    """Over multiple bursts, accumulated drift should stay below 100ms.

    Today: drift grows monotonically because every burst-cycle drops a
    fractional frame's worth (frame-cap) AND silently discards audio above
    max_advance_ms (last_total_samples=total bug).

    After fix: drift should stay bounded by one inference cycle's worth
    (~one max_advance_ms = 240ms worst case, but ideally near zero).
    """
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    fps = 25
    args = make_test_args(
        window_ms=640,
        hop_ms=80,
        max_advance_ms=240,
        fps=fps,
        max_tail_frames=5,
    )
    engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=2.0)

    # Feed 5 bursts of 500ms each, paced 200ms apart.
    # Each burst (8000 samples = 500ms) exceeds max_advance_ms=240, so the
    # bug fires once per burst.
    async def feeder():
        for _ in range(5):
            await ring.append(np.random.randn(8000).astype(np.float32) * 0.5)
            await asyncio.sleep(0.2)

    feed_task = asyncio.ensure_future(feeder())
    engine_task = asyncio.create_task(engine.run())
    await asyncio.sleep(2.0)
    engine.stop_event.set()
    await asyncio.gather(engine_task, feed_task, return_exceptions=True)

    # Compute drift: how many seconds of audio did we silently drop?
    # If the engine were perfect, dropped should be near zero and frames_published
    # should match the time we ran (~50 frames in 2s at 25fps).
    drift_ms = engine.dropped_audio_ms_total
    # Today this is hundreds of ms; after fix should be near zero.
    assert drift_ms < 100.0, (
        f"Audio/video drift accumulated to {drift_ms:.0f}ms over a 2-second test. "
        "On real workloads this manifests as the avatar visibly running ahead "
        "of (or behind) the spoken audio."
    )


# -----------------------------------------------------------------------------
# Non-xfail tests: verify that the FakeEngine's cap math is at least internally
# CONSISTENT, even though the policy itself is buggy. These should keep passing
# both before and after the fix; they document the wiring of the policy knobs.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_respects_max_advance_clamp(ring, video_buffer):
    """When fed audio > max_advance_ms, the engine defers excess to next iteration."""
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    args = make_test_args(window_ms=640, hop_ms=80, max_advance_ms=100, fps=25)
    engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=2.0)

    # 500ms of audio = 5x the 100ms max_advance budget
    burst = np.random.randn(8000).astype(np.float32) * 0.5
    await ring.append(burst)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.5)
    engine.stop_event.set()
    await task

    # With the fix: audio is NOT dropped, it's consumed across multiple iterations.
    # The engine should process all 500ms across ~5 iterations of 100ms each.
    assert engine.jobs >= 4, (
        f"Expected multiple iterations to consume the burst, got {engine.jobs}"
    )
    # All audio consumed — frames should match total duration
    expected_frames = int(round(0.5 * 25))  # 500ms at 25fps = ~12-13 frames
    assert engine.frames_published >= expected_frames - 2


@pytest.mark.asyncio
async def test_engine_publishes_frames_proportional_to_audio(ring, video_buffer):
    """Frame count is derived from consumed audio duration, not artificially capped."""
    from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

    args = make_test_args(
        window_ms=640,
        hop_ms=80,
        max_advance_ms=1000,  # large advance budget — consume all at once
        fps=25,
        max_tail_frames=3,    # this cap is now ignored
    )
    engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=2.0)

    # 640ms of audio → should yield ~16 frames at 25fps in one cycle
    burst = np.random.randn(int(16000 * 0.64)).astype(np.float32) * 0.5
    await ring.append(burst)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.1)
    engine.stop_event.set()
    await task

    assert engine.jobs == 1
    # Should publish frames proportional to audio, NOT capped at max_tail_frames
    expected = int(round(0.64 * 25))  # ~16 frames
    assert engine.frames_published >= expected - 1, (
        f"Published {engine.frames_published} frames, expected ~{expected}"
    )
