"""Shared pytest fixtures for the unified server test suite.

All fixtures here are GPU-free and construct the system using FakeInferenceEngine
and FakeMoshiServer, enabling full pipeline testing on a laptop.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

# Ensure MuseTalk and unified_server are importable.
# We add the MuseTalk root so that `scripts.musetalk_webrtc.buffers` etc. resolve,
# BUT we need to handle the case where `sphn` / `aiortc` / `torch` are unavailable
# (those are only needed by some submodules, not by buffers/models).
MUSETALK_ROOT = Path(__file__).resolve().parent.parent.parent / "MuseTalk"
UNIFIED_ROOT = Path(__file__).resolve().parent.parent

for p in [str(MUSETALK_ROOT), str(UNIFIED_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Patch: prevent the musetalk_webrtc __init__.py from importing cli→server→sphn
# when we only need buffers/models. We do this by pre-loading the package with
# an empty module if needed, then importing submodules directly.
_webrtc_pkg = "scripts.musetalk_webrtc"
if _webrtc_pkg not in sys.modules:
    import types

    pkg = types.ModuleType(_webrtc_pkg)
    pkg.__path__ = [str(MUSETALK_ROOT / "scripts" / "musetalk_webrtc")]
    pkg.__package__ = _webrtc_pkg
    sys.modules[_webrtc_pkg] = pkg
    # Also register parent package
    if "scripts" not in sys.modules:
        scripts_pkg = types.ModuleType("scripts")
        scripts_pkg.__path__ = [str(MUSETALK_ROOT / "scripts")]
        scripts_pkg.__package__ = "scripts"
        sys.modules["scripts"] = scripts_pkg


def _import_buffers():
    """Import buffers module without triggering full package __init__."""
    return importlib.import_module("scripts.musetalk_webrtc.buffers")


def _import_models():
    """Import models module without triggering full package __init__."""
    return importlib.import_module("scripts.musetalk_webrtc.models")


def make_test_args(**overrides) -> "AppArgs":
    """Create a minimal AppArgs with sensible defaults for testing.

    All values are chosen to avoid external dependencies (no model paths,
    no real ports, no ICE servers).
    """
    models = _import_models()
    AppArgs = models.AppArgs

    defaults = dict(
        host="127.0.0.1",
        port=0,
        ice_servers=[],
        ice_transport_policy="all",
        ice_username="",
        ice_credential="",
        personaplex_host="127.0.0.1",
        personaplex_port=9999,
        personaplex_path="/api/chat",
        personaplex_text_prompt="You are helpful.",
        personaplex_voice_prompt="test.pt",
        personaplex_extra_query=[],
        avatar_id="test_avatar",
        version="v15",
        gpu_id=0,
        use_fp16=False,
        require_mmpose=False,
        fps=25,
        avatar_fps=25,
        batch_size=4,
        bbox_shift=0,
        unet_model_path="",
        unet_config="",
        vae_type="sd-vae-ft-mse",
        whisper_dir="",
        ffmpeg_path="ffmpeg",
        parsing_mode="jaw",
        extra_margin=10,
        left_cheek_width=90,
        right_cheek_width=90,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        ring_buffer_seconds=10.0,
        window_ms=640,
        hop_ms=80,
        min_window_ms=320,
        max_advance_ms=240,
        max_tail_frames=5,
        mouth_smoothing_alpha=0.75,
        video_queue_size=64,
        status_json=None,
        reconnect_delay_seconds=1.0,
        input_source="mirror",
        webrtc_audio_loopback=False,
        musetalk_only=False,
        enable_api_auth=False,
        api_token="test-token-123",
        session_offer_timeout_seconds=30.0,
        session_max_age_seconds=3600.0,
        session_cleanup_interval_seconds=10.0,
        session_disconnect_grace_seconds=10.0,
        ice_gather_timeout_seconds=5.0,
        single_session_mode=True,
        web_test_only=False,
        debug=True,
        debug_events_limit=100,
    )
    defaults.update(overrides)
    return AppArgs(**defaults)


@pytest.fixture
def test_args():
    """Provide default test AppArgs."""
    return make_test_args()


@pytest.fixture
def pcm_ring_16k():
    """Provide a fresh 16kHz PcmRingBuffer sized for 10s."""
    buffers = _import_buffers()
    return buffers.PcmRingBuffer(max_samples=160000)


@pytest.fixture
def pcm_ring_24k():
    """Provide a fresh 24kHz PcmRingBuffer sized for 10s."""
    buffers = _import_buffers()
    return buffers.PcmRingBuffer(max_samples=240000)


@pytest.fixture
def video_buffer():
    """Provide a fresh VideoFrameBuffer."""
    buffers = _import_buffers()
    return buffers.VideoFrameBuffer(maxsize=64)


@pytest.fixture
def audio_track_buffer():
    """Provide a fresh AudioTrackBuffer sized for 10s @ 48kHz."""
    buffers = _import_buffers()
    return buffers.AudioTrackBuffer(max_samples_48k=480000)


@pytest.fixture
def fake_engine(test_args, pcm_ring_16k, video_buffer):
    """Provide a FakeInferenceEngine ready to run."""
    fake_mod = importlib.import_module("scripts.musetalk_webrtc.engines.fake")
    return fake_mod.FakeInferenceEngine(
        test_args,
        pcm_ring_16k,
        video_buffer,
        simulated_inference_ms=5.0,
    )
