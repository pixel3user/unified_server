"""Unit tests for the WebRTC disconnect grace period.

Production bug this prevents: mobile users on WiFi switching to LTE (or
walking through a tunnel) cause WebRTC to enter 'disconnected' state for a
few seconds. Without a grace period, the session either:
  - Lingers forever (if last_activity_epoch keeps getting touched), or
  - Gets killed by the broad pc_disconnected idle rule (~20s+) which is
    way too long to wait for the user to re-offer.

The grace period gives a tight, dedicated knob (`session_disconnect_grace_seconds`,
default 10s) that closes a stuck session promptly so the client can reconnect.

These tests verify _session_should_expire's new branch behavior.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

import pytest

MUSETALK_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "MuseTalk"
if str(MUSETALK_ROOT) not in sys.path:
    sys.path.insert(0, str(MUSETALK_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import make_test_args


@pytest.fixture(autouse=True)
def _mock_heavy_imports():
    """Mock sphn/aiortc/av and evict any cloudflare-style stubs."""
    for stub in [
        "scripts.musetalk_webrtc.server",
        "scripts.musetalk_webrtc.cli",
        "scripts.musetalk_webrtc",
        "scripts",
    ]:
        mod = sys.modules.get(stub)
        if mod is not None and not getattr(mod, "__file__", None):
            del sys.modules[stub]
    for mod_name in ["sphn", "aiortc", "aiortc.contrib", "aiortc.contrib.media"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    if "av" not in sys.modules:
        av_mock = types.ModuleType("av")
        av_mock.VideoFrame = type("VideoFrame", (), {"from_ndarray": staticmethod(lambda *a, **kw: None)})
        av_mock.AudioFrame = type("AudioFrame", (), {})
        sys.modules["av"] = av_mock
    # Ensure torch mock has Tensor (cloudflare test may have installed a bare mock)
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "Tensor"):
        sys.modules["torch"].Tensor = type("Tensor", (), {})


@pytest.fixture
def app():
    """Build a WebRtcApp in web_test_only mode with grace=10s."""
    from scripts.musetalk_webrtc.server import WebRtcApp

    args = make_test_args(
        web_test_only=True,
        session_disconnect_grace_seconds=10.0,
    )
    return WebRtcApp(args)


def _make_session(
    *,
    created_epoch: float = None,
    last_activity_epoch: float = None,
    disconnected_at: float = 0.0,
    pcid: str = "pc_test",
    pc_value=None,
):
    """Build a SessionState that looks like a real connected one by default."""
    models = importlib.import_module("scripts.musetalk_webrtc.models")
    now = time.time()
    return models.SessionState(
        session_id="sess-001",
        token="test-token",
        created_epoch=created_epoch if created_epoch is not None else now,
        last_activity_epoch=last_activity_epoch if last_activity_epoch is not None else now,
        pc=pc_value if pc_value is not None else "mock_pc",  # non-None → not offer_timeout
        pcid=pcid,
        disconnected_at=disconnected_at,
    )


class TestDisconnectGraceField:
    """The new field is wired through correctly."""

    def test_app_args_carries_default(self, app):
        assert app.args.session_disconnect_grace_seconds == 10.0

    def test_session_disconnected_at_defaults_to_zero(self):
        session = _make_session()
        assert session.disconnected_at == 0.0


class TestGracePeriodNotElapsed:
    """While within the grace window, the session should NOT be expired."""

    def test_just_disconnected_does_not_expire(self, app):
        now = time.time()
        session = _make_session(disconnected_at=now)  # disconnected this instant
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is False, f"unexpected expiry reason: {reason}"

    def test_disconnected_5s_with_10s_grace_does_not_expire(self, app):
        now = time.time()
        session = _make_session(disconnected_at=now - 5.0)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is False
        assert reason == ""

    def test_disconnected_exactly_at_grace_does_not_expire(self, app):
        """Boundary: exactly equal to grace should NOT expire (strict > comparison)."""
        now = time.time()
        session = _make_session(disconnected_at=now - 10.0)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        # The check is `(now - disconnected_at) > grace`, so equality stays alive.
        assert should is False


class TestGracePeriodElapsed:
    """Past the grace window, the session must be expired with the right reason."""

    def test_disconnected_15s_with_10s_grace_expires(self, app):
        now = time.time()
        session = _make_session(disconnected_at=now - 15.0)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is True
        assert reason == "disconnect_grace_exceeded"

    def test_grace_exceeded_takes_precedence_over_pc_idle(self, app):
        """Verify the new grace branch fires before the broader pc_disconnected rule."""
        # Disconnected 11s ago AND idle 100s ago (both rules would trigger).
        # The grace rule runs first in the function, so we should see its reason.
        now = time.time()
        session = _make_session(
            disconnected_at=now - 11.0,
            last_activity_epoch=now - 100.0,
        )
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is True
        assert reason == "disconnect_grace_exceeded"


class TestGracePeriodWithCustomDuration:
    """Verify the configurable knob works."""

    def test_short_grace_expires_quickly(self):
        """A 1-second grace should expire after 1.5s."""
        from scripts.musetalk_webrtc.server import WebRtcApp

        args = make_test_args(
            web_test_only=True,
            session_disconnect_grace_seconds=1.0,
        )
        app = WebRtcApp(args)

        now = time.time()
        session = _make_session(disconnected_at=now - 1.5)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is True
        assert reason == "disconnect_grace_exceeded"

    def test_long_grace_keeps_session_alive(self):
        """A 60-second grace should keep the session alive for 30s of disconnect."""
        from scripts.musetalk_webrtc.server import WebRtcApp

        args = make_test_args(
            web_test_only=True,
            session_disconnect_grace_seconds=60.0,
        )
        app = WebRtcApp(args)

        now = time.time()
        session = _make_session(disconnected_at=now - 30.0)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "disconnected"

        should, reason = app._session_should_expire(session, now)
        assert should is False


class TestNoFalsePositives:
    """Sessions that never disconnected should never trigger the grace rule."""

    def test_connected_session_with_zero_disconnected_at(self, app):
        """disconnected_at=0 means 'never disconnected' — must never expire on grace."""
        now = time.time()
        session = _make_session(disconnected_at=0.0)
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "connected"

        should, reason = app._session_should_expire(session, now)
        assert should is False
        # Importantly: even if the absolute now is very large, we must not
        # interpret (now - 0) as "disconnected forever ago".
        assert reason == ""

    def test_recovered_session_after_disconnected_at_cleared(self, app):
        """When the connectionstatechange handler clears disconnected_at to 0
        on reconnect, the grace rule must immediately stop applying."""
        now = time.time()
        session = _make_session(disconnected_at=0.0)  # reconnected → cleared
        app.sessions[session.session_id] = session
        app.pc_states[session.pcid] = "connected"

        # Even if some other counter is stale, grace shouldn't fire
        should, reason = app._session_should_expire(session, now + 1000.0)
        assert reason != "disconnect_grace_exceeded"
