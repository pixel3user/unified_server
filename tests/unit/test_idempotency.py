"""Unit tests for the /offer idempotency cache.

The bug this prevents: when a browser's `fetch('/offer', ...)` retries due
to a transient network glitch, every retry creates a new session and (in
single-session mode) immediately closes the previous one. Users see "the
connection drops randomly" — but it's actually their own retried fetch
killing the in-flight session.

The fix: an `x-idempotency-key` header lets the server return the cached
answer SDP for retries within IDEMPOTENCY_TTL_SECONDS instead of creating
a new session.

These tests cover the cache layer directly. Full HTTP route tests come in
the next layer of the pyramid (Layer 3).
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
    """Mock sphn/aiortc/av so server.py can be imported without those deps.

    Also evicts any stub `scripts.musetalk_webrtc.server` module that earlier
    tests (e.g. test_cloudflare_turn) may have inserted, so we get the real
    implementation here.
    """
    # Evict stubs from previous test modules
    for stub in [
        "scripts.musetalk_webrtc.server",
        "scripts.musetalk_webrtc.cli",
        "scripts.musetalk_webrtc",
        "scripts",
    ]:
        mod = sys.modules.get(stub)
        # Only evict modules that look like our test stubs (no __file__).
        # Real modules have __file__ set; the stub types.ModuleType ones don't.
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


@pytest.fixture
def app():
    """Build a WebRtcApp in web_test_only mode (no engine, no models)."""
    from scripts.musetalk_webrtc.server import WebRtcApp

    args = make_test_args(web_test_only=True, single_session_mode=True)
    return WebRtcApp(args)


@pytest.fixture
def constants():
    return importlib.import_module("scripts.musetalk_webrtc.constants")


class TestIdempotencyLookup:
    """Read path: stored entries are returned, expired entries are not."""

    def test_empty_cache_returns_none(self, app):
        assert app._idempotency_lookup("any-key") is None

    def test_empty_key_returns_none(self, app):
        """An empty/missing idempotency key should never match anything."""
        app._idempotency_cache[""] = {"payload": {"x": 1}, "expires_at": time.time() + 10}
        assert app._idempotency_lookup("") is None

    def test_stored_entry_is_returned(self, app):
        payload = {"sdp": "v=0...", "session_id": "abc"}
        app._idempotency_store("key-1", payload)
        result = app._idempotency_lookup("key-1")
        assert result == payload

    def test_expired_entry_is_not_returned(self, app):
        """An entry past its expires_at must be returned as None and evicted."""
        app._idempotency_cache["stale"] = {
            "payload": {"x": 1},
            "expires_at": time.time() - 1.0,  # expired 1s ago
        }
        assert app._idempotency_lookup("stale") is None
        # Side effect: should be evicted
        assert "stale" not in app._idempotency_cache


class TestIdempotencyStore:
    """Write path: TTL is set correctly, cache stays bounded."""

    def test_store_sets_ttl(self, app, constants):
        before = time.time()
        app._idempotency_store("k", {"x": 1})
        after = time.time()
        entry = app._idempotency_cache["k"]
        # expires_at should be roughly now + TTL
        assert before + constants.IDEMPOTENCY_TTL_SECONDS - 0.5 <= entry["expires_at"]
        assert entry["expires_at"] <= after + constants.IDEMPOTENCY_TTL_SECONDS + 0.5

    def test_store_with_empty_key_is_noop(self, app):
        app._idempotency_store("", {"x": 1})
        assert "" not in app._idempotency_cache

    def test_store_overwrites_existing_key(self, app):
        app._idempotency_store("k", {"version": 1})
        app._idempotency_store("k", {"version": 2})
        assert app._idempotency_lookup("k") == {"version": 2}

    def test_cache_evicts_expired_entries_when_full(self, app, constants):
        """When at capacity, expired entries are reclaimed before insertion."""
        cap = constants.IDEMPOTENCY_CACHE_MAX_ENTRIES

        # Fill cache, half expired, half fresh
        now = time.time()
        for i in range(cap):
            expires = now - 1 if i < cap // 2 else now + 100
            app._idempotency_cache[f"k{i}"] = {"payload": {"i": i}, "expires_at": expires}

        # Now insert one more — should trigger eviction of expired entries
        app._idempotency_store("new-key", {"x": "new"})

        # The new key should be present
        assert app._idempotency_lookup("new-key") == {"x": "new"}
        # Expired keys should be gone
        for i in range(cap // 2):
            assert f"k{i}" not in app._idempotency_cache

    def test_cache_drops_oldest_when_no_expired_to_reclaim(self, app, constants):
        """If cache is full and nothing is expired, drop a portion of oldest."""
        cap = constants.IDEMPOTENCY_CACHE_MAX_ENTRIES
        now = time.time()
        # Fill cache with all-fresh entries, sorted by expires_at
        for i in range(cap):
            app._idempotency_cache[f"k{i}"] = {
                "payload": {"i": i},
                # Earlier i → earlier expires_at → "older"
                "expires_at": now + 100 + i,
            }
        app._idempotency_store("new-key", {"x": "new"})

        assert "new-key" in app._idempotency_cache
        # We should have dropped at least one of the oldest entries.
        # The exact eviction count is an implementation detail; just verify
        # the cache hasn't grown unboundedly.
        assert len(app._idempotency_cache) <= constants.IDEMPOTENCY_CACHE_MAX_ENTRIES


class TestEndToEndSemantics:
    """Verify the full lookup→hit→store cycle behaves correctly."""

    def test_store_then_lookup_returns_same_payload(self, app):
        payload = {"session_id": "x", "sdp": "v=0\r\n", "type": "answer"}
        app._idempotency_store("retry-key-1", payload)
        assert app._idempotency_lookup("retry-key-1") == payload

    def test_different_keys_isolated(self, app):
        app._idempotency_store("a", {"x": 1})
        app._idempotency_store("b", {"x": 2})
        assert app._idempotency_lookup("a") == {"x": 1}
        assert app._idempotency_lookup("b") == {"x": 2}

    def test_lookup_after_manual_expiry(self, app):
        """Simulate clock skew by manually setting expires_at to the past."""
        app._idempotency_store("k", {"x": 1})
        # Force expiry
        app._idempotency_cache["k"]["expires_at"] = time.time() - 0.001
        assert app._idempotency_lookup("k") is None
