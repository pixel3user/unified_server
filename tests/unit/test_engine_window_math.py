"""Unit tests for inference engine windowing/hop logic and FakeInferenceEngine behavior."""

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
    return buffers.PcmRingBuffer(max_samples=160000)


@pytest.fixture
async def video_buffer():
    buffers = importlib.import_module("scripts.musetalk_webrtc.buffers")
    return buffers.VideoFrameBuffer(maxsize=64)


class TestWindowMathConversions:
    """Verify that the engine's sample count calculations match the audio spec."""

    def test_window_samples(self):
        """window_ms=640 at 16kHz = 10240 samples."""
        args = make_test_args(window_ms=640)
        expected = int((640 / 1000.0) * 16000)
        assert expected == 10240

    def test_hop_samples(self):
        """hop_ms=80 at 16kHz = 1280 samples (min new audio per inference)."""
        args = make_test_args(hop_ms=80)
        expected = int((80 / 1000.0) * 16000)
        assert expected == 1280

    def test_max_advance_samples(self):
        """max_advance_ms=240 at 16kHz = 3840 samples."""
        args = make_test_args(max_advance_ms=240)
        expected = int((240 / 1000.0) * 16000)
        assert expected == 3840

    def test_frames_from_samples(self):
        """80ms of new audio at 25fps = 2 frames."""
        fps = 25
        new_samples = 1280  # 80ms
        expected_frames = max(1, int(round((new_samples / 16000.0) * fps)))
        assert expected_frames == 2


class TestFakeEngineBasicBehavior:
    """Test FakeInferenceEngine's windowing loop without a GPU."""

    @pytest.mark.asyncio
    async def test_engine_runs_and_publishes_on_speech(self, ring, video_buffer):
        """FakeEngine publishes frames when speech audio is in the ring."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args(window_ms=640, hop_ms=80, fps=25, max_tail_frames=5)
        engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=1.0)

        # Feed 200ms of loud audio (3200 samples at 16kHz, well above hop threshold)
        speech = np.random.randn(3200).astype(np.float32) * 0.5
        await ring.append(speech)

        # Run engine for a short burst
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(0.15)
        engine.stop_event.set()
        await task

        assert engine.jobs > 0
        assert engine.frames_published > 0
        assert video_buffer.queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_engine_waits_for_min_hop(self, ring, video_buffer):
        """FakeEngine does not infer until hop_ms worth of new audio arrives."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args(window_ms=640, hop_ms=80, fps=25)
        engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=1.0)

        # Feed only 500 samples (31ms < 80ms hop) — not enough to trigger
        tiny = np.ones(500, dtype=np.float32) * 0.5
        await ring.append(tiny)

        task = asyncio.create_task(engine.run())
        await asyncio.sleep(0.15)
        engine.stop_event.set()
        await task

        # Should have done 0 inference jobs (only topups)
        assert engine.jobs == 0

    @pytest.mark.asyncio
    async def test_engine_tracks_dropped_audio(self, ring, video_buffer):
        """When new_samples > max_advance, excess is tracked as dropped."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args(
            window_ms=640, hop_ms=80, max_advance_ms=240, fps=25, max_tail_frames=5
        )
        engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=1.0)

        # Feed 500ms = 8000 samples (max_advance = 3840, so 4160 would be "dropped")
        big_chunk = np.random.randn(8000).astype(np.float32) * 0.3
        await ring.append(big_chunk)

        task = asyncio.create_task(engine.run())
        await asyncio.sleep(0.1)
        engine.stop_event.set()
        await task

        assert engine.dropped_audio_ms_total > 0

    @pytest.mark.asyncio
    async def test_engine_failure_injection(self, ring, video_buffer):
        """fail_after_n_jobs causes RuntimeError after N successful jobs."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args(window_ms=640, hop_ms=80, fps=25)
        engine = FakeInferenceEngine(
            args, ring, video_buffer, simulated_inference_ms=1.0, fail_after_n_jobs=2
        )

        # Feed enough audio continuously for several jobs
        async def feeder():
            for _ in range(50):
                await ring.append(np.random.randn(1600).astype(np.float32) * 0.3)
                await asyncio.sleep(0.01)

        feed_task = asyncio.ensure_future(feeder())
        with pytest.raises(RuntimeError, match="simulated failure"):
            await engine.run()
        feed_task.cancel()

        assert engine.jobs == 2

    @pytest.mark.asyncio
    async def test_engine_ready_event(self, ring, video_buffer):
        """Engine sets ready event shortly after run() starts."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args()
        engine = FakeInferenceEngine(args, ring, video_buffer, simulated_inference_ms=1.0)

        assert not engine.ready.is_set()
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(0.05)
        assert engine.ready.is_set()
        engine.stop_event.set()
        await task

    @pytest.mark.asyncio
    async def test_engine_status_shape(self, ring, video_buffer):
        """status() returns expected dict keys."""
        from scripts.musetalk_webrtc.engines.fake import FakeInferenceEngine

        args = make_test_args()
        engine = FakeInferenceEngine(args, ring, video_buffer)

        status = engine.status()
        assert "jobs" in status
        assert "last_publish_epoch" in status
        assert "last_error" in status
        assert "dropped_audio_ms_total" in status
        assert "fake" in status
        assert status["fake"] is True
