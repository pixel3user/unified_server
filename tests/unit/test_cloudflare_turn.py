"""Unit tests for CloudflareTurnProvider: caching, fallback, error handling."""

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))

# Mock heavy dependencies that unified_server imports at module level
_heavy_mocks = [
    "huggingface_hub", "sentencepiece", "torch", "torch.cuda",
    "moshi", "moshi.models", "moshi.models.loaders", "moshi.server",
    "scripts", "scripts.musetalk_webrtc", "scripts.musetalk_webrtc.cli",
    "scripts.musetalk_webrtc.server",
]
for mod_name in _heavy_mocks:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m

# Add required attributes that unified_server.py references
sys.modules["torch"].cuda = types.ModuleType("torch.cuda")
sys.modules["torch"].cuda.is_available = lambda: False
sys.modules["torch"].cuda.current_device = lambda: 0
sys.modules["torch"].cuda.mem_get_info = lambda d: (0, 0)
sys.modules["torch"].cuda.memory_allocated = lambda d: 0
sys.modules["torch"].cuda.memory_reserved = lambda d: 0
sys.modules["torch"].cuda.max_memory_allocated = lambda d: 0
sys.modules["torch"].cuda.max_memory_reserved = lambda d: 0
sys.modules["torch"].cuda.get_device_name = lambda d: "mock"
sys.modules["torch"].no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **kw: "/dev/null"
sys.modules["moshi.models.loaders"].DEFAULT_REPO = "mock/repo"
sys.modules["moshi.models.loaders"].MIMI_NAME = "mimi.bin"
sys.modules["moshi.models.loaders"].TEXT_TOKENIZER_NAME = "tok.bin"
sys.modules["moshi.models.loaders"].MOSHI_NAME = "moshi.bin"
sys.modules["moshi.server"].ServerState = type("ServerState", (), {})
sys.modules["moshi.server"]._get_voice_prompt_dir = lambda *a: "/tmp"
sys.modules["moshi.server"].seed_all = lambda x: None
sys.modules["moshi.server"].torch_auto_device = lambda x: "cpu"
sys.modules["scripts.musetalk_webrtc.cli"].parse_args = lambda: None
sys.modules["scripts.musetalk_webrtc.server"].WebRtcApp = type("WebRtcApp", (), {})

from unified_server import CloudflareTurnProvider


@pytest.fixture
def provider():
    return CloudflareTurnProvider(
        token_id="abcdef1234567890abcdef1234567890",
        api_token="test-api-token-for-unit-tests",
        ttl_seconds=3600,
        fallback_policy="all",
    )


def _mock_response(data: dict, status: int = 200):
    """Create a mock urllib response context manager."""
    import io

    class FakeResp:
        def __init__(self):
            self.status = status
            self._body = json.dumps(data).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return FakeResp()


class TestCaching:
    """Cache TTL and refresh behavior."""

    def test_first_call_fetches(self, provider):
        """First get_config() makes a network request."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            config = provider.get_config()
        assert len(config["iceServers"]) == 1
        assert "turn:example.com:3478" in config["iceServers"][0]["urls"]

    def test_second_call_uses_cache(self, provider):
        """Second call within TTL returns cached config without fetch."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)) as mock_open:
            provider.get_config()
            provider.get_config()
        # Should only have called urlopen once
        assert mock_open.call_count == 1

    def test_cache_expires_after_ttl(self, provider):
        """After TTL window passes, a new fetch is made."""
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)) as mock_open:
            provider.get_config()
            # Force cache expiration
            provider._cached_until = time.time() - 1
            provider.get_config()
        assert mock_open.call_count == 2


class TestErrorHandling:
    """Error paths and fallback behavior."""

    def test_http_error_raises(self, provider):
        """HTTP error from Cloudflare raises RuntimeError."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://example.com",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=__import__("io").BytesIO(b'{"error": "forbidden"}'),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="Cloudflare TURN HTTP 403"):
                provider.get_config()

    def test_last_error_populated_on_failure(self, provider):
        """last_error and last_status_code are set on failure."""
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=__import__("io").BytesIO(b"server error"),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError):
                provider.get_config()
        assert provider.last_status_code == 500


class TestFiltering:
    """URL filtering (e.g., removing :53 DNS ports)."""

    def test_filters_port_53_urls(self, provider):
        """TURN URLs with :53 are filtered out (Cloudflare DNS port).

        NOTE: The current filter uses substring ':53' which also matches ':5349'.
        This is a known bug — the filter is overly aggressive. This test documents
        the actual behavior. When fixed, update this test to keep :5349.
        """
        mock_data = {
            "iceServers": [
                {
                    "urls": [
                        "turn:turn.example.com:3478",
                        "turn:dns.example.com:53",  # should be filtered
                        "turns:turn.example.com:5349",  # BUG: also filtered by ":53" substring
                    ],
                    "username": "user",
                    "credential": "pass",
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            config = provider.get_config()

        urls = config["iceServers"][0]["urls"]
        assert "turn:dns.example.com:53" not in urls
        assert "turn:turn.example.com:3478" in urls
        # BUG: :5349 is incorrectly filtered because of naive ":53" substring match
        # TODO: Fix filter to use regex or exact port match
        assert "turns:turn.example.com:5349" not in urls  # documents the bug


class TestDebugIdentity:
    """Debug identity output for diagnostics."""

    def test_debug_identity_structure(self, provider):
        """debug_identity returns expected keys with masked values."""
        identity = provider.debug_identity()
        assert "token_id_prefix" in identity
        assert "token_id_suffix" in identity
        assert "api_token_len" in identity
        assert "api_token_sha256_prefix" in identity
        assert len(identity["token_id_prefix"]) == 8
        assert len(identity["token_id_suffix"]) == 6




class _FakeClock:
    """Manually-advanced clock for deterministic circuit breaker tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestCircuitBreaker:
    """Verify open/closed/half-open transitions and fast-fail behavior.

    These tests exist because the production bug was: every new WebRTC
    /offer would block ~15s on an HTTP timeout when Cloudflare was down.
    The breaker must skip the network call entirely while in 'open' state.
    """

    @pytest.fixture
    def clock(self):
        return _FakeClock(t=1000.0)

    @pytest.fixture
    def provider(self, clock):
        return CloudflareTurnProvider(
            token_id="abcdef1234567890abcdef1234567890",
            api_token="test-api-token-for-unit-tests",
            ttl_seconds=3600,
            fallback_policy="all",
            failure_threshold=3,
            circuit_open_duration_seconds=60.0,
            clock=clock,
        )

    def _http_500(self):
        import urllib.error
        import io
        return urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"server error"),
        )

    def test_starts_closed(self, provider):
        """A fresh provider has a closed circuit."""
        assert provider.circuit_state == "closed"

    def test_one_failure_does_not_open_circuit(self, provider):
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            with pytest.raises(RuntimeError):
                provider.get_config()
        assert provider.circuit_state == "closed"
        assert provider._consecutive_failures == 1

    def test_threshold_failures_open_circuit(self, provider):
        """After failure_threshold failures, the circuit opens."""
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    provider.get_config()
        assert provider.circuit_state == "open"
        assert provider._consecutive_failures == 3

    def test_open_circuit_fails_fast_without_network_call(self, provider):
        """While open, get_config raises immediately with no urlopen call.

        This is the production bug fix: no more 15s blocking timeouts.
        """
        # Trip the breaker
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    provider.get_config()
        assert provider.circuit_state == "open"

        # Now subsequent calls must NOT touch the network.
        with patch("urllib.request.urlopen") as mock_open:
            with pytest.raises(RuntimeError, match="circuit_open"):
                provider.get_config()
            assert mock_open.call_count == 0

    def test_open_to_half_open_after_cooldown(self, provider, clock):
        """After circuit_open_duration_seconds, state transitions to half_open."""
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    provider.get_config()
        assert provider.circuit_state == "open"

        clock.advance(61.0)
        assert provider.circuit_state == "half_open"

    def test_half_open_success_closes_circuit(self, provider, clock):
        """A successful probe in half_open state closes the circuit."""
        # Trip
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    provider.get_config()

        clock.advance(61.0)
        assert provider.circuit_state == "half_open"

        # Probe succeeds
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            config = provider.get_config()
        assert config["iceServers"]
        assert provider.circuit_state == "closed"
        assert provider._consecutive_failures == 0

    def test_half_open_failure_re_opens_circuit(self, provider, clock):
        """A failed probe in half_open state re-opens the circuit."""
        # Trip
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            for _ in range(3):
                with pytest.raises(RuntimeError):
                    provider.get_config()

        clock.advance(61.0)
        assert provider.circuit_state == "half_open"

        # Probe fails again — circuit re-opens, cooldown restarts.
        with patch("urllib.request.urlopen", side_effect=self._http_500()):
            with pytest.raises(RuntimeError):
                provider.get_config()
        assert provider.circuit_state == "open"

    def test_cached_config_bypasses_breaker(self, provider, clock):
        """A valid cache entry should be returned even if the breaker is open.

        Rationale: the breaker protects against DOWNSTREAM outages, but if we
        already have a recent valid config in cache, we should serve it.
        """
        # Populate cache with a successful call
        mock_data = {"iceServers": [{"urls": "turn:example.com:3478"}]}
        with patch("urllib.request.urlopen", return_value=_mock_response(mock_data)):
            provider.get_config()

        # Manually trip the breaker (simulating a later refresh failure burst)
        provider._consecutive_failures = 999
        provider._circuit_opened_at = clock.t

        # Cached call should still return the cached config without raising.
        # We verify this by NOT advancing the clock past TTL.
        cfg = provider.get_config()
        assert cfg["iceServers"]


class TestFastFailWallTime:
    """Sanity check: an open breaker rejects in microseconds, not seconds."""

    def test_open_circuit_rejects_in_under_10ms(self):
        """The whole point of the breaker is sub-millisecond rejection."""
        clock = _FakeClock(t=1000.0)
        provider = CloudflareTurnProvider(
            token_id="abcdef1234567890abcdef1234567890",
            api_token="test-api-token-for-unit-tests",
            ttl_seconds=3600,
            fallback_policy="all",
            failure_threshold=1,
            circuit_open_duration_seconds=60.0,
            clock=clock,
        )
        # Trip with a single failure
        import urllib.error
        import io
        exc = urllib.error.HTTPError("u", 500, "x", {}, io.BytesIO(b""))
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError):
                provider.get_config()
        assert provider.circuit_state == "open"

        # Now measure how long an open-circuit call takes
        import time as _real_time
        start = _real_time.perf_counter()
        with pytest.raises(RuntimeError, match="circuit_open"):
            provider.get_config()
        elapsed_ms = (_real_time.perf_counter() - start) * 1000
        assert elapsed_ms < 10.0, f"Open circuit took {elapsed_ms:.2f}ms, expected <10ms"
