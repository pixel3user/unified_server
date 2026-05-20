"""Shared fixtures for HTTP integration tests.

These tests use aiohttp's TestClient to exercise the full route layer
without needing a real WebRTC peer or GPU. They mock sphn/aiortc at module
level and use web_test_only mode.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# Mock heavy deps
for mod_name in ["sphn", "aiortc", "aiortc.contrib", "aiortc.contrib.media"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
if "av" not in sys.modules:
    av_mock = types.ModuleType("av")
    av_mock.VideoFrame = type("VideoFrame", (), {"from_ndarray": staticmethod(lambda *a, **kw: None)})
    av_mock.AudioFrame = type("AudioFrame", (), {})
    sys.modules["av"] = av_mock

# Evict any previously-loaded stub modules
for stub in ["scripts.musetalk_webrtc.server", "scripts.musetalk_webrtc.cli", "scripts.musetalk_webrtc", "scripts"]:
    mod = sys.modules.get(stub)
    if mod is not None and not getattr(mod, "__file__", None):
        del sys.modules[stub]

MUSETALK_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "MuseTalk"
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(MUSETALK_ROOT), str(UNIFIED_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import make_test_args


@pytest.fixture
async def client():
    """Create a TestClient backed by a WebRtcApp in web_test_only mode."""
    from scripts.musetalk_webrtc.server import WebRtcApp

    args = make_test_args(
        web_test_only=True,
        single_session_mode=True,
        enable_api_auth=False,
    )
    app_state = WebRtcApp(args)
    app = app_state.build_app()
    # Attach app_state so tests can inspect it
    app["_app_state"] = app_state

    server = TestServer(app)
    tc = TestClient(server)
    await tc.start_server()
    try:
        yield tc
    finally:
        await tc.close()


@pytest.fixture
async def auth_client():
    """Create a TestClient with API auth enabled."""
    from scripts.musetalk_webrtc.server import WebRtcApp

    args = make_test_args(
        web_test_only=True,
        single_session_mode=True,
        enable_api_auth=True,
        api_token="secret-test-token",
    )
    app_state = WebRtcApp(args)
    app = app_state.build_app()
    app["_app_state"] = app_state

    server = TestServer(app)
    tc = TestClient(server)
    await tc.start_server()
    try:
        yield tc
    finally:
        await tc.close()
